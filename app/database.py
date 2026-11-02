import os
import sqlite3
import time
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List
import numpy as np

DATA_DIR = Path(os.getenv("DATA_DIR", "/data"))
DB_PATH = DATA_DIR / "db" / "faces.db"
IMAGES_DIR = DATA_DIR / "images"
MAX_IMAGES_PER_HOUR = int(os.getenv("MAX_IMAGES_PER_HOUR", 3))


def get_connection() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    return conn


def init_db() -> None:
    conn = get_connection()
    with conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS persons (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                person_type TEXT NOT NULL DEFAULT 'Unknown',
                alert_rule TEXT NOT NULL DEFAULT 'default',
                best_image_path TEXT,
                best_quality_score REAL DEFAULT 0.0,
                first_seen TEXT NOT NULL,
                last_seen TEXT NOT NULL,
                last_camera TEXT NOT NULL,
                total_sightings INTEGER NOT NULL DEFAULT 1,
                last_alert_sent_at REAL DEFAULT 0.0
            );

            CREATE TABLE IF NOT EXISTS embeddings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                person_id INTEGER NOT NULL,
                embedding BLOB NOT NULL,
                quality_score REAL NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (person_id) REFERENCES persons(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS hourly_images (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                person_id INTEGER NOT NULL,
                hour_bucket TEXT NOT NULL,
                image_path TEXT NOT NULL,
                quality_score REAL NOT NULL,
                camera_id TEXT NOT NULL,
                captured_at TEXT NOT NULL,
                FOREIGN KEY (person_id) REFERENCES persons(id) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_hourly_person_bucket
            ON hourly_images(person_id, hour_bucket);

            CREATE TABLE IF NOT EXISTS sightings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                person_id INTEGER NOT NULL,
                camera_id TEXT NOT NULL,
                captured_at TEXT NOT NULL,
                similarity REAL NOT NULL,
                quality_score REAL NOT NULL,
                image_path TEXT,
                FOREIGN KEY (person_id) REFERENCES persons(id) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_sightings_captured_at
            ON sightings(captured_at DESC);

            CREATE TABLE IF NOT EXISTS cameras (
                camera_id TEXT PRIMARY KEY,
                friendly_name TEXT NOT NULL,
                last_seen TEXT
            );
            
            CREATE TABLE IF NOT EXISTS settings (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS settings (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS custom_tags (
                name TEXT PRIMARY KEY,
                alert_enabled INTEGER NOT NULL DEFAULT 0
            );
            """
        )
        
        # Safe migration: add custom_tag column to persons if it doesn't exist yet
        cols = [r["name"] for r in conn.execute("PRAGMA table_info(persons)").fetchall()]
        if "custom_tag" not in cols:
            conn.execute("ALTER TABLE persons ADD COLUMN custom_tag TEXT DEFAULT NULL;")
    conn.close()


def save_or_prune_hourly_image(
    conn: sqlite3.Connection,
    person_id: int,
    camera_id: str,
    captured_dt: datetime,
    quality_score: float,
    image_bytes: bytes,
) -> Optional[str]:
    """
    Keeps at most MAX_IMAGES_PER_HOUR (3) best images for a person in a given hour.
    Returns relative image path if saved, or None if discarded.
    """
    hour_bucket = captured_dt.strftime("%Y-%m-%d_%H")
    person_folder = IMAGES_DIR / f"person_{person_id}" / hour_bucket
    person_folder.mkdir(parents=True, exist_ok=True)

    existing = conn.execute(
        """
        SELECT id, image_path, quality_score
        FROM hourly_images
        WHERE person_id = ? AND hour_bucket = ?
        ORDER BY quality_score ASC
        """,
        (person_id, hour_bucket),
    ).fetchall()

    # If we already have 3 images in this hour, check if the new one beats the worst of the 3
    if len(existing) >= MAX_IMAGES_PER_HOUR:
        worst = existing[0]
        if quality_score <= worst["quality_score"]:
            return None  # Not better than existing 3 images; skip saving to disk

        # Remove the lowest-quality image file from disk and DB
        old_full_path = IMAGES_DIR / worst["image_path"]
        if old_full_path.exists():
            try:
                old_full_path.unlink()
            except OSError:
                pass
        conn.execute("DELETE FROM hourly_images WHERE id = ?", (worst["id"],))
        # Clear image_path reference in any sightings pointing to the deleted file
        conn.execute(
            "UPDATE sightings SET image_path = NULL WHERE image_path = ?",
            (worst["image_path"],),
        )

    # Save the new higher-quality image
    ts_ms = int(time.time() * 1000)
    filename = f"{captured_dt.strftime('%M%S')}_{ts_ms}_q{int(quality_score * 100)}.jpg"
    rel_path = f"person_{person_id}/{hour_bucket}/{filename}"
    full_path = IMAGES_DIR / rel_path
    full_path.write_bytes(image_bytes)

    conn.execute(
        """
        INSERT INTO hourly_images (person_id, hour_bucket, image_path, quality_score, camera_id, captured_at)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            person_id,
            hour_bucket,
            rel_path,
            quality_score,
            camera_id,
            captured_dt.isoformat(timespec="seconds"),
        ),
    )
    return rel_path


def get_db_stats() -> Dict[str, Any]:
    conn = get_connection()
    stats = {
        "db_file_exists": DB_PATH.exists(),
        "total_persons": conn.execute("SELECT COUNT(*) FROM persons").fetchone()[0],
        "total_sightings": conn.execute("SELECT COUNT(*) FROM sightings").fetchone()[0],
        "stored_hourly_images": conn.execute("SELECT COUNT(*) FROM hourly_images").fetchone()[0],
    }
    conn.close()
    return stats


def update_person_details(
    person_id: int,
    name: str,
    person_type: str,
    alert_rule: str,
    custom_tag: Optional[str] = None,
) -> None:
    conn = get_connection()
    try:
        with conn:
            if not name.startswith("Unknown") and person_type == "Unknown":
                person_type = "Known"
            clean_tag = custom_tag.strip() if (custom_tag and custom_tag.strip()) else None
            conn.execute(
                """
                UPDATE persons
                SET name = ?, person_type = ?, alert_rule = ?, custom_tag = ?
                WHERE id = ?
                """,
                (name.strip(), person_type, alert_rule, clean_tag, person_id),
            )
    finally:
        conn.close()


def merge_persons(source_id: int, target_id: int) -> bool:
    """
    Merges source_id into target_id: moves embeddings, sightings, and hourly images,
    updates stats, and deletes source_id.
    """
    if source_id == target_id:
        return False

    conn = get_connection()
    try:
        with conn:
            src = conn.execute("SELECT * FROM persons WHERE id = ?", (source_id,)).fetchone()
            tgt = conn.execute("SELECT * FROM persons WHERE id = ?", (target_id,)).fetchone()
            if not src or not tgt:
                return False

            # Move embeddings and sightings to target
            conn.execute("UPDATE embeddings SET person_id = ? WHERE person_id = ?", (target_id, source_id))
            conn.execute("UPDATE sightings SET person_id = ? WHERE person_id = ?", (target_id, source_id))

            # Move hourly_images files on disk and update paths
            hr_rows = conn.execute("SELECT * FROM hourly_images WHERE person_id = ?", (source_id,)).fetchall()
            for row in hr_rows:
                old_rel = row["image_path"]
                old_full = IMAGES_DIR / old_rel
                new_rel = old_rel.replace(f"person_{source_id}/", f"person_{target_id}/", 1)
                new_full = IMAGES_DIR / new_rel
                new_full.parent.mkdir(parents=True, exist_ok=True)
                if old_full.exists():
                    shutil.move(str(old_full), str(new_full))

                conn.execute(
                    "UPDATE hourly_images SET person_id = ?, image_path = ? WHERE id = ?",
                    (target_id, new_rel, row["id"]),
                )
                conn.execute(
                    "UPDATE sightings SET image_path = ? WHERE image_path = ?",
                    (new_rel, old_rel),
                )

            # Update target person stats
            first_seen = min(src["first_seen"], tgt["first_seen"])
            if src["last_seen"] > tgt["last_seen"]:
                last_seen = src["last_seen"]
                last_camera = src["last_camera"]
            else:
                last_seen = tgt["last_seen"]
                last_camera = tgt["last_camera"]

            total_sightings = src["total_sightings"] + tgt["total_sightings"]

            best_img = tgt["best_image_path"]
            best_q = tgt["best_quality_score"] or 0.0
            if (src["best_quality_score"] or 0.0) > best_q and src["best_image_path"]:
                best_img = src["best_image_path"].replace(f"person_{source_id}/", f"person_{target_id}/", 1)
                best_q = src["best_quality_score"]

            conn.execute(
                """
                UPDATE persons
                SET first_seen = ?, last_seen = ?, last_camera = ?,
                    total_sightings = ?, best_image_path = ?, best_quality_score = ?
                WHERE id = ?
                """,
                (first_seen, last_seen, last_camera, total_sightings, best_img, best_q, target_id),
            )

            # Delete source person row and empty folder
            conn.execute("DELETE FROM persons WHERE id = ?", (source_id,))
            src_folder = IMAGES_DIR / f"person_{source_id}"
            if src_folder.exists():
                shutil.rmtree(src_folder, ignore_errors=True)

            return True
    finally:
        conn.close()


def delete_person(person_id: int) -> None:
    conn = get_connection()
    try:
        with conn:
            conn.execute("DELETE FROM persons WHERE id = ?", (person_id,))
        folder = IMAGES_DIR / f"person_{person_id}"
        if folder.exists():
            shutil.rmtree(folder, ignore_errors=True)
    finally:
        conn.close()

def delete_hourly_image(image_id: int) -> Optional[int]:
    """
    Permanently deletes an hourly image from disk and DB, clears sighting references,
    and updates the person's best_image_path avatar if needed.
    Returns person_id if found and deleted, else None.
    """
    conn = get_connection()
    try:
        with conn:
            row = conn.execute(
                "SELECT id, person_id, image_path FROM hourly_images WHERE id = ?",
                (image_id,),
            ).fetchone()
            if not row:
                return None

            person_id = row["person_id"]
            rel_path = row["image_path"]

            # 1. Permanently delete file from disk
            full_path = IMAGES_DIR / rel_path
            if full_path.exists():
                try:
                    full_path.unlink()
                except OSError:
                    pass

            # 2. Remove from hourly_images and clear any sighting reference
            conn.execute("DELETE FROM hourly_images WHERE id = ?", (image_id,))
            conn.execute(
                "UPDATE sightings SET image_path = NULL WHERE image_path = ?",
                (rel_path,),
            )

            # 3. If this was the person's best_image_path, promote next best remaining image
            person = conn.execute(
                "SELECT best_image_path FROM persons WHERE id = ?", (person_id,)
            ).fetchone()

            if person and person["best_image_path"] == rel_path:
                next_best = conn.execute(
                    """
                    SELECT image_path, quality_score
                    FROM hourly_images
                    WHERE person_id = ?
                    ORDER BY quality_score DESC
                    LIMIT 1
                    """,
                    (person_id,),
                ).fetchone()

                if next_best:
                    conn.execute(
                        "UPDATE persons SET best_image_path = ?, best_quality_score = ? WHERE id = ?",
                        (next_best["image_path"], next_best["quality_score"], person_id),
                    )
                else:
                    conn.execute(
                        "UPDATE persons SET best_image_path = NULL, best_quality_score = 0.0 WHERE id = ?",
                        (person_id,),
                    )

            return person_id
    finally:
        conn.close()


def get_custom_tags() -> List[Dict[str, Any]]:
    conn = get_connection()
    try:
        rows = conn.execute(
            "SELECT name, alert_enabled FROM custom_tags ORDER BY name ASC"
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def upsert_custom_tag(name: str, alert_enabled: bool) -> None:
    clean_name = name.strip()
    if not clean_name:
        return
    conn = get_connection()
    try:
        with conn:
            conn.execute(
                """
                INSERT INTO custom_tags (name, alert_enabled)
                VALUES (?, ?)
                ON CONFLICT(name) DO UPDATE SET alert_enabled = excluded.alert_enabled
                """,
                (clean_name, 1 if alert_enabled else 0),
            )
    finally:
        conn.close()


def delete_custom_tag(name: str) -> None:
    conn = get_connection()
    try:
        with conn:
            conn.execute("UPDATE persons SET custom_tag = NULL WHERE custom_tag = ?", (name,))
            conn.execute("DELETE FROM custom_tags WHERE name = ?", (name,))
    finally:
        conn.close()