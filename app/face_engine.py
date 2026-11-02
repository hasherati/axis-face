import os
import time
from datetime import datetime
from typing import Optional, Dict, Any, Tuple
import cv2
import numpy as np
import onnxruntime as ort
from insightface.app import FaceAnalysis

from database import get_connection, save_or_prune_hourly_image, IMAGES_DIR

SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", 0.45))
MIN_DET_SCORE = float(os.getenv("MIN_DET_SCORE", 0.68))
MIN_QUALITY_SCORE = float(os.getenv("MIN_QUALITY_SCORE", 0.65))
MAX_EMBEDDINGS_PER_PERSON = 5


class FaceEngine:
    def __init__(self):
        self.face_app: Optional[FaceAnalysis] = None
        self.gpu_status: Dict[str, Any] = {
            "requested_mode": os.getenv("EXECUTION_MODE", "auto").lower(),
            "active_mode": "CPU",
            "onnx_providers": [],
            "cuda_available": False,
            "model_loaded": False,
            "warmup_inference_ms": None,
            "error": None,
        }

    def start(self) -> None:
        requested = os.getenv("EXECUTION_MODE", "auto").strip().lower()
        self.gpu_status["requested_mode"] = requested

        try:
            avail_providers = ort.get_available_providers()
            self.gpu_status["onnx_providers"] = avail_providers

            # Check if NVIDIA GPU device is actually exposed inside the container
            gpu_device_present = os.path.exists("/dev/nvidiactl") or os.path.exists("/dev/nvidia0")
            cuda_supported = ("CUDAExecutionProvider" in avail_providers) and gpu_device_present
            self.gpu_status["cuda_available"] = cuda_supported

            if requested == "cpu":
                use_gpu = False
            elif requested == "gpu":
                use_gpu = True
            else:  # "auto"
                use_gpu = cuda_supported

            if use_gpu:
                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
                ctx_id = 0
                self.gpu_status["active_mode"] = "GPU"
            else:
                providers = ["CPUExecutionProvider"]
                ctx_id = -1
                self.gpu_status["active_mode"] = "CPU"

            print(
                f"=== Initializing InsightFace | Requested: {requested.upper()} "
                f"| Active: {self.gpu_status['active_mode']} ==="
            )

            self.face_app = FaceAnalysis(name="buffalo_l", providers=providers)
            self.face_app.prepare(ctx_id=ctx_id, det_size=(320, 320))
            self.gpu_status["model_loaded"] = True

            dummy_img = np.zeros((320, 320, 3), dtype=np.uint8)
            self.face_app.get(dummy_img)

            t0 = time.perf_counter()
            self.face_app.get(dummy_img)
            t1 = time.perf_counter()
            self.gpu_status["warmup_inference_ms"] = round((t1 - t0) * 1000, 2)
            print(
                f"=== {self.gpu_status['active_mode']} Warm-up Complete: "
                f"{self.gpu_status['warmup_inference_ms']} ms ==="
            )
        except Exception as e:
            self.gpu_status["error"] = str(e)
            print(f"ERROR initializing FaceEngine pipeline: {e}")

    def _compute_quality_score(self, img: np.ndarray, face: Any) -> float:
        """Computes a 0.0 - 1.0 quality score from sharpness, frontal pose, and resolution."""
        h, w = img.shape[:2]
        x1, y1, x2, y2 = [int(v) for v in face.bbox]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        face_crop = img[y1:y2, x1:x2]
        if face_crop.size == 0:
            face_crop = img

        # 1. Sharpness via Laplacian variance (normalized roughly to 0..1)
        gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        sharpness_score = min(1.0, lap_var / 250.0)

        # 2. Pose Frontalness from 5-point landmarks (left_eye, right_eye, nose, left_mouth, right_mouth)
        pose_score = 0.5
        if hasattr(face, "kps") and face.kps is not None and len(face.kps) == 5:
            kps = face.kps
            left_eye, right_eye, nose = kps[0], kps[1], kps[2]
            eye_dist = float(np.linalg.norm(right_eye - left_eye))
            if eye_dist > 1.0:
                # Yaw symmetry: nose should be horizontally centered between eyes
                left_to_nose = abs(nose[0] - left_eye[0])
                right_to_nose = abs(right_eye[0] - nose[0])
                yaw_asym = abs(left_to_nose - right_to_nose) / eye_dist
                yaw_score = max(0.0, 1.0 - yaw_asym)

                # Roll penalty: eyes should be roughly level
                eye_dy = abs(right_eye[1] - left_eye[1]) / eye_dist
                roll_score = max(0.0, 1.0 - eye_dy)
                pose_score = 0.65 * yaw_score + 0.35 * roll_score

        # 3. Resolution & detection confidence
        face_width = max(1, x2 - x1)
        res_score = min(1.0, face_width / 140.0)
        det_score = float(getattr(face, "det_score", 0.8))

        composite = (
            (0.40 * sharpness_score)
            + (0.35 * pose_score)
            + (0.15 * res_score)
            + (0.10 * det_score)
        )
        return round(float(np.clip(composite, 0.01, 0.99)), 4)

    def _is_valid_face_geometry(self, face: Any) -> Tuple[bool, str]:
        """
        Verifies that the detection has a high confidence score and anatomically valid
        5-point facial landmarks (prevents back-of-head and extreme profile false positives).
        """
        det_score = float(getattr(face, "det_score", 0.0))
        if det_score < MIN_DET_SCORE:
            return False, f"Low face detection confidence ({det_score:.2f} < {MIN_DET_SCORE})"

        if not hasattr(face, "kps") or face.kps is None or len(face.kps) != 5:
            return False, "Missing 5-point facial landmarks"

        x1, y1, x2, y2 = [float(v) for v in face.bbox]
        box_w = max(1.0, x2 - x1)
        box_h = max(1.0, y2 - y1)

        left_eye, right_eye, nose, left_mouth, right_mouth = face.kps

        # 1. Left eye must be to the left of right eye, and separated by at least 18% of bbox width
        eye_dx = float(right_eye[0] - left_eye[0])
        if eye_dx / box_w < 0.18:
            return False, f"Profile too steep or invalid eye spacing ({eye_dx / box_w:.2f})"

        # 2. Eyes must be roughly level (roll angle under ~35 degrees)
        eye_dy = abs(float(right_eye[1] - left_eye[1]))
        if eye_dy > eye_dx * 0.70:
            return False, "Invalid eye tilt/roll"

        # 3. Vertical anatomical order: eyes above nose, nose above mouth
        eye_mid_y = (left_eye[1] + right_eye[1]) / 2.0
        mouth_mid_y = (left_mouth[1] + right_mouth[1]) / 2.0
        if not (eye_mid_y < nose[1] < mouth_mid_y):
            return False, "Landmarks failed vertical eye-nose-mouth order"

        # Vertical span from eyes to mouth must be at least 20% of face height
        if (mouth_mid_y - eye_mid_y) / box_h < 0.20:
            return False, "Landmarks vertically compressed"

        # 4. Nose must sit horizontally between the left and right eyes (filters extreme yaw/back-of-head)
        if not (left_eye[0] <= nose[0] <= right_eye[0]):
            return False, "Extreme side profile (nose outside eye span)"

        return True, "Valid face"

    def extract_face(self, image_bytes: bytes) -> Tuple[Optional[Tuple[np.ndarray, float, bytes]], str]:
        """
        Decodes image bytes, detects/aligns the primary face, validates geometry & quality,
        and returns ((normalized_512d_embedding, quality_score, normalized_jpeg_bytes), reason).
        """
        if not image_bytes:
            return None, "Empty image payload"

        np_arr = np.frombuffer(image_bytes, np.uint8)
        if np_arr.size == 0:
            return None, "Empty buffer"

        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if img is None or self.face_app is None:
            return None, "Invalid image encoding"

        faces = self.face_app.get(img)
        working_img = img

        if not faces:
            h, w = img.shape[:2]
            pad_y, pad_x = int(h * 0.4), int(w * 0.4)
            working_img = cv2.copyMakeBorder(
                img, pad_y, pad_y, pad_x, pad_x, cv2.BORDER_CONSTANT, value=[0, 0, 0]
            )
            faces = self.face_app.get(working_img)

        if not faces:
            return None, "No face detected in Head crop"

        best_face = max(
            faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])
        )

        # Gate 1: Check detection score and 5-point anatomical geometry
        is_valid, reason = self._is_valid_face_geometry(best_face)
        if not is_valid:
            return None, reason

        # Gate 2: Check minimum composite quality score
        quality_score = self._compute_quality_score(working_img, best_face)
        if quality_score < MIN_QUALITY_SCORE:
            return None, f"Quality score too low ({quality_score:.2f} < {MIN_QUALITY_SCORE})"

        embedding = best_face.normed_embedding.astype(np.float32)

        ok, encoded_jpg = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        jpeg_bytes = encoded_jpg.tobytes() if ok else image_bytes

        return (embedding, quality_score, jpeg_bytes), "OK"

    def process_face_image(
        self,
        image_bytes: bytes,
        camera_id: str,
        captured_dt: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """
        Full pipeline: Extracts embedding + quality, matches or creates Person in DB,
        applies the Top-3 hourly image pruning rule, and records the sighting.
        """
        if not image_bytes:
            return {"detected": False, "reason": "Uploaded file or image payload was empty (0 bytes)"}

        if captured_dt is None:
            captured_dt = datetime.now()

        extracted, reason = self.extract_face(image_bytes)
        if extracted is None:
            return {"detected": False, "reason": reason}

        embedding, quality_score, jpeg_bytes = extracted
        now_iso = captured_dt.isoformat(timespec="seconds")

        conn = get_connection()
        try:
            with conn:
                conn.execute(
                    """
                    INSERT INTO cameras (camera_id, friendly_name, last_seen)
                    VALUES (?, ?, ?)
                    ON CONFLICT(camera_id) DO UPDATE SET last_seen = excluded.last_seen
                    """,
                    (camera_id, camera_id, now_iso),
                )

                rows = conn.execute(
                    "SELECT person_id, embedding FROM embeddings"
                ).fetchall()

                best_person_id = None
                best_similarity = -1.0

                for row in rows:
                    stored_vec = np.frombuffer(row["embedding"], dtype=np.float32)
                    sim = float(np.dot(embedding, stored_vec))
                    if sim > best_similarity:
                        best_similarity = sim
                        best_person_id = row["person_id"]

                is_new_person = False
                if best_person_id is not None and best_similarity >= SIMILARITY_THRESHOLD:
                    person_id = best_person_id
                    similarity = round(best_similarity, 4)
                else:
                    is_new_person = True
                    similarity = 1.0
                    cur = conn.execute(
                        """
                        INSERT INTO persons (name, person_type, first_seen, last_seen, last_camera, total_sightings)
                        VALUES ('Unknown', 'Unknown', ?, ?, ?, 0)
                        """,
                        (now_iso, now_iso, camera_id),
                    )
                    person_id = cur.lastrowid
                    conn.execute(
                        "UPDATE persons SET name = ? WHERE id = ?",
                        (f"Unknown #{person_id}", person_id),
                    )

                saved_rel_path = save_or_prune_hourly_image(
                    conn=conn,
                    person_id=person_id,
                    camera_id=camera_id,
                    captured_dt=captured_dt,
                    quality_score=quality_score,
                    image_bytes=jpeg_bytes,
                )

                person_row = conn.execute(
                    "SELECT * FROM persons WHERE id = ?", (person_id,)
                ).fetchone()

                best_image_path = person_row["best_image_path"]
                best_quality = person_row["best_quality_score"] or 0.0
                if saved_rel_path and (
                    best_image_path is None
                    or quality_score > best_quality
                    or not (IMAGES_DIR / best_image_path).exists()
                ):
                    best_image_path = saved_rel_path
                    best_quality = quality_score

                conn.execute(
                    """
                    UPDATE persons
                    SET last_seen = ?,
                        last_camera = ?,
                        total_sightings = total_sightings + 1,
                        best_image_path = ?,
                        best_quality_score = ?
                    WHERE id = ?
                    """,
                    (now_iso, camera_id, best_image_path, best_quality, person_id),
                )

                emb_rows = conn.execute(
                    "SELECT id, quality_score FROM embeddings WHERE person_id = ? ORDER BY quality_score ASC",
                    (person_id,),
                ).fetchall()
                if len(emb_rows) < MAX_EMBEDDINGS_PER_PERSON:
                    conn.execute(
                        "INSERT INTO embeddings (person_id, embedding, quality_score, created_at) VALUES (?, ?, ?, ?)",
                        (person_id, embedding.tobytes(), quality_score, now_iso),
                    )
                elif quality_score > emb_rows[0]["quality_score"]:
                    conn.execute(
                        "UPDATE embeddings SET embedding = ?, quality_score = ?, created_at = ? WHERE id = ?",
                        (embedding.tobytes(), quality_score, now_iso, emb_rows[0]["id"]),
                    )

                conn.execute(
                    """
                    INSERT INTO sightings (person_id, camera_id, captured_at, similarity, quality_score, image_path)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (person_id, camera_id, now_iso, similarity, quality_score, saved_rel_path),
                )

                updated_person = conn.execute(
                    "SELECT id, name, person_type, total_sightings, best_quality_score, best_image_path FROM persons WHERE id = ?",
                    (person_id,),
                ).fetchone()

                return {
                    "detected": True,
                    "is_new_person": is_new_person,
                    "person_id": person_id,
                    "name": updated_person["name"],
                    "person_type": updated_person["person_type"],
                    "similarity": similarity,
                    "quality_score": quality_score,
                    "saved_to_hourly_bucket": saved_rel_path is not None,
                    "saved_image_path": saved_rel_path,
                    "total_sightings": updated_person["total_sightings"],
                }
        finally:
            conn.close()


face_engine = FaceEngine()