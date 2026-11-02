import asyncio
import base64
import json
import os
from datetime import datetime
from typing import Optional, Tuple, Dict, Any, List
from email_alerter import check_and_send_alert
import aiomqtt

from face_engine import face_engine

MQTT_BROKER_HOST = os.getenv("MQTT_BROKER_HOST", "mosquitto")
MQTT_BROKER_PORT = int(os.getenv("MQTT_BROKER_PORT", 1883))
MQTT_USERNAME = os.getenv("MQTT_USERNAME", "") or None
MQTT_PASSWORD = os.getenv("MQTT_PASSWORD", "") or None

mqtt_stats: Dict[str, Any] = {
    "connected": False,
    "broker": f"{MQTT_BROKER_HOST}:{MQTT_BROKER_PORT}",
    "messages_received": 0,
    "skipped_non_face": 0,
    "faces_matched_or_added": 0,
    "last_topic": None,
    "last_camera_id": None,
    "seen_class_types": [],
    "last_payload_structure": None,
    "last_message_at": None,
    "last_error": None,
}


def redact_base64_for_debug(obj: Any) -> Any:
    """Replaces long base64 strings with a short placeholder so we can inspect the exact JSON schema."""
    if isinstance(obj, dict):
        return {k: redact_base64_for_debug(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [redact_base64_for_debug(i) for i in obj]
    elif isinstance(obj, str) and len(obj) > 120:
        return f"<STRING_LEN_{len(obj)}>"
    return obj


def extract_camera_id_from_topic(topic: str) -> str:
    marker = "/scene/object_snapshot"
    if marker in topic:
        prefix = topic.split(marker)[0].strip("/")
        if prefix:
            return prefix.replace("/", "_")
    parts = topic.strip("/").split("/")
    return parts[0] if parts else "unknown_camera"


def extract_all_class_types(obj: Any) -> List[str]:
    """
    Collects ALL 'type' values inside 'class' or 'classes' anywhere in the JSON payload.
    """
    found: List[str] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in ("class", "classes"):
                if isinstance(v, dict) and "type" in v:
                    found.append(str(v["type"]))
                elif isinstance(v, list):
                    for item in v:
                        if isinstance(item, dict) and "type" in item:
                            found.append(str(item["type"]))
                elif isinstance(v, str):
                    found.append(v)
            elif isinstance(v, (dict, list)):
                found.extend(extract_all_class_types(v))
    elif isinstance(obj, list):
        for item in obj:
            found.extend(extract_all_class_types(item))
    return found


def find_base64_image_in_json(obj: Any) -> Optional[str]:
    if isinstance(obj, dict):
        for key in ("image", "data", "snapshot", "jpeg", "base64", "picture", "crop"):
            val = obj.get(key)
            if isinstance(val, str) and len(val) > 100:
                return val
        for val in obj.values():
            found = find_base64_image_in_json(val)
            if found:
                return found
    elif isinstance(obj, list):
        for item in obj:
            found = find_base64_image_in_json(item)
            if found:
                return found
    elif isinstance(obj, str) and len(obj) > 200:
        if obj.startswith("/9j/") or obj.startswith("iVBOR") or "base64," in obj:
            return obj
    return None


def parse_axis_payload(payload: bytes) -> Tuple[Optional[bytes], Optional[datetime], List[str]]:
    if not payload:
        return None, None, []

    try:
        text = payload.decode("utf-8").strip()
        data = json.loads(text)
    except Exception:
        return None, None, ["non_json"]

    # Save redacted JSON structure so we can see exact Axis keys at http://localhost:8000/
    redacted = redact_base64_for_debug(data)
    mqtt_stats["last_payload_structure"] = redacted
    print(f"[MQTT DEBUG] Incoming JSON structure: {json.dumps(redacted)}")

    class_types = extract_all_class_types(data)
    for ct in class_types:
        if ct not in mqtt_stats["seen_class_types"]:
            mqtt_stats["seen_class_types"].append(ct)

    # Check if 'face' is in any of the extracted class types
    has_face = any(ct.lower() in ("face", "head") for ct in class_types)
    if not has_face:
        return None, None, class_types

    b64_str = find_base64_image_in_json(data)
    if not b64_str:
        return None, None, class_types

    if "base64," in b64_str:
        b64_str = b64_str.split("base64,", 1)[1]

    try:
        image_bytes = base64.b64decode(b64_str)
    except Exception:
        return None, None, class_types

    captured_dt = datetime.now()
    if isinstance(data, dict):
        ts_val = (
            data.get("timestamp")
            or data.get("time")
            or (data.get("frame", {}) if isinstance(data.get("frame"), dict) else {}).get("timestamp")
        )
        if isinstance(ts_val, str):
            try:
                captured_dt = datetime.fromisoformat(ts_val.replace("Z", "+00:00")).astimezone().replace(tzinfo=None)
            except Exception:
                pass

    return image_bytes, captured_dt, class_types


async def mqtt_listener_loop() -> None:
    while True:
        try:
            print(f"[MQTT] Connecting to broker at {MQTT_BROKER_HOST}:{MQTT_BROKER_PORT}...")
            async with aiomqtt.Client(
                hostname=MQTT_BROKER_HOST,
                port=MQTT_BROKER_PORT,
                username=MQTT_USERNAME,
                password=MQTT_PASSWORD,
            ) as client:
                mqtt_stats["connected"] = True
                mqtt_stats["last_error"] = None

                await client.subscribe("+/scene/object_snapshot/v1/#")
                await client.subscribe("+/+/scene/object_snapshot/v1/#")
                await client.subscribe("scene/object_snapshot/v1/#")
                print("[MQTT] Connected and subscribed to */scene/object_snapshot/v1/#")

                async for message in client.messages:
                    topic_str = str(message.topic)
                    camera_id = extract_camera_id_from_topic(topic_str)

                    mqtt_stats["messages_received"] += 1
                    mqtt_stats["last_topic"] = topic_str
                    mqtt_stats["last_camera_id"] = camera_id
                    mqtt_stats["last_message_at"] = datetime.now().isoformat(timespec="seconds")

                    image_bytes, captured_dt, class_types = parse_axis_payload(message.payload)

                    if not any(ct.lower() in ("face", "head") for ct in class_types):
                        mqtt_stats["skipped_non_face"] += 1
                        continue

                    if not image_bytes:
                        mqtt_stats["last_error"] = f"Face message on {topic_str} had no valid image data"
                        continue

                    result = await asyncio.to_thread(
                        face_engine.process_face_image,
                        image_bytes,
                        camera_id,
                        captured_dt,
                    )

                    if result.get("detected"):
                        mqtt_stats["faces_matched_or_added"] += 1
                        print(
                            f"[MQTT] Camera={camera_id} | Person={result['name']} "
                            f"(sim={result['similarity']}, q={result['quality_score']}) "
                            f"| Saved={result['saved_to_hourly_bucket']}"
                        )
                        asyncio.create_task(check_and_send_alert(result, image_bytes))
                    else:
                        print(f"[MQTT] Camera={camera_id} | Face crop rejected by InsightFace: {result.get('reason')}")

        except asyncio.CancelledError:
            mqtt_stats["connected"] = False
            break
        except Exception as e:
            mqtt_stats["connected"] = False
            mqtt_stats["last_error"] = str(e)
            await asyncio.sleep(5)