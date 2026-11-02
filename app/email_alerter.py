import os
import time
from email.message import EmailMessage
from email.utils import make_msgid
from typing import Dict, Any, Optional, Tuple
import aiosmtplib

from database import get_connection, IMAGES_DIR


DEFAULT_SETTINGS = {
    "alert_enabled": os.getenv("ALERT_ENABLED", "false").lower(),
    "smtp_host": os.getenv("SMTP_HOST", ""),
    "smtp_port": os.getenv("SMTP_PORT", "587"),
    "smtp_use_tls": os.getenv("SMTP_USE_TLS", "true").lower(),
    "smtp_username": os.getenv("SMTP_USERNAME", ""),
    "smtp_password": os.getenv("SMTP_PASSWORD", ""),
    "alert_from_email": os.getenv("ALERT_FROM_EMAIL", ""),
    "alert_to_email": os.getenv("ALERT_TO_EMAIL", ""),
    "alert_on_known": os.getenv("ALERT_ON_KNOWN", "true").lower(),
    "alert_on_unknown": os.getenv("ALERT_ON_UNKNOWN", "true").lower(),
    "alert_cooldown_minutes": os.getenv("ALERT_COOLDOWN_MINUTES", "15"),
}


def get_alert_settings() -> Dict[str, str]:
    conn = get_connection()
    try:
        rows = conn.execute("SELECT key, value FROM settings").fetchall()
        db_vals = {r["key"]: r["value"] for r in rows}
        merged = dict(DEFAULT_SETTINGS)
        merged.update(db_vals)
        return merged
    finally:
        conn.close()


def should_send_alert(person_id: int) -> Tuple[bool, str, Dict[str, Any]]:
    """
    Checks if an email alert should be triggered for this person_id right now.
    Evaluates:
      1. Per-person rule ('never' or 'always')
      2. Custom tag alert rule (if person has a custom_tag with alert_enabled=1)
      3. Standard tag global rule (Known vs Unknown)
    """
    cfg = get_alert_settings()
    if cfg["alert_enabled"] != "true":
        return False, "Alerts disabled globally", {}

    if not cfg["smtp_host"] or not cfg["alert_to_email"]:
        return False, "SMTP host or recipient email not configured", {}

    conn = get_connection()
    try:
        person = conn.execute(
            "SELECT * FROM persons WHERE id = ?", (person_id,)
        ).fetchone()
        if not person:
            return False, "Person not found", {}

        cam_row = conn.execute(
            "SELECT friendly_name FROM cameras WHERE camera_id = ?",
            (person["last_camera"],),
        ).fetchone()
        camera_name = cam_row["friendly_name"] if cam_row else person["last_camera"]

        rule = (person["alert_rule"] or "default").lower()
        p_type = (person["person_type"] or "Unknown").lower()
        custom_tag = person["custom_tag"]

        # Check if person's custom_tag has email alerts enabled
        custom_tag_alert = False
        if custom_tag:
            tag_row = conn.execute(
                "SELECT alert_enabled FROM custom_tags WHERE name = ?",
                (custom_tag,),
            ).fetchone()
            if tag_row and int(tag_row["alert_enabled"]) == 1:
                custom_tag_alert = True

        if rule == "never":
            return False, "Person alert_rule is set to 'never'", {}
        elif rule == "default":
            if not custom_tag_alert:
                if p_type == "known" and cfg["alert_on_known"] != "true":
                    return False, "Global alert_on_known is false and no custom tag alert active", {}
                if p_type != "known" and cfg["alert_on_unknown"] != "true":
                    return False, "Global alert_on_unknown is false and no custom tag alert active", {}

        cooldown_sec = float(cfg.get("alert_cooldown_minutes", 15)) * 60.0
        now_ts = time.time()
        last_sent = float(person["last_alert_sent_at"] or 0.0)
        if (now_ts - last_sent) < cooldown_sec:
            remaining = int(cooldown_sec - (now_ts - last_sent))
            return False, f"In cooldown ({remaining}s remaining)", {}

        with conn:
            conn.execute(
                "UPDATE persons SET last_alert_sent_at = ? WHERE id = ?",
                (now_ts, person_id),
            )

        return True, "Triggered", {
            "person": dict(person),
            "camera_name": camera_name,
            "cfg": cfg,
        }
    finally:
        conn.close()


async def send_alert_email(
    person_name: str,
    person_type: str,
    camera_name: str,
    captured_at: str,
    quality_score: float,
    similarity: float,
    custom_tag: Optional[str] = None,
    image_bytes: Optional[bytes] = None,
    cfg: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    if cfg is None:
        cfg = get_alert_settings()

    tag_label = f" [{custom_tag}]" if custom_tag else ""
    msg = EmailMessage()
    msg["Subject"] = f"[Axis Face Alert] {person_type}{tag_label}: {person_name} seen on {camera_name}"
    msg["From"] = cfg["alert_from_email"] or cfg["smtp_username"]
    msg["To"] = cfg["alert_to_email"]

    cid = make_msgid(domain="axis-face.local")
    cid_clean = cid[1:-1]

    img_html = (
        f'<p><img src="cid:{cid_clean}" alt="Face Crop" '
        f'style="max-width:280px; border-radius:8px; border:2px solid #334155;"/></p>'
        if image_bytes
        else "<p><em>(No image attached)</em></p>"
    )

    custom_tag_row = (
        f'<tr><td style="padding: 4px 12px 4px 0;"><strong>Custom Tag:</strong></td><td>{custom_tag}</td></tr>'
        if custom_tag
        else ""
    )

    html_body = f"""
    <html>
      <body style="font-family: Arial, sans-serif; color: #1e293b;">
        <h3 style="margin-bottom: 8px;">Face Detection Alert</h3>
        <table style="border-collapse: collapse; margin-bottom: 12px;">
          <tr><td style="padding: 4px 12px 4px 0;"><strong>Identity:</strong></td><td>{person_name} ({person_type})</td></tr>
          {custom_tag_row}
          <tr><td style="padding: 4px 12px 4px 0;"><strong>Camera:</strong></td><td>{camera_name}</td></tr>
          <tr><td style="padding: 4px 12px 4px 0;"><strong>Time:</strong></td><td>{captured_at}</td></tr>
          <tr><td style="padding: 4px 12px 4px 0;"><strong>Quality / Match:</strong></td><td>Q={quality_score:.2f} | Sim={similarity:.2f}</td></tr>
        </table>
        {img_html}
      </body>
    </html>
    """

    msg.set_content(
        f"Face Detection Alert: {person_name} ({person_type}{tag_label}) seen on {camera_name} at {captured_at}."
    )
    msg.add_alternative(html_body, subtype="html")

    if image_bytes:
        msg.get_payload()[1].add_related(
            image_bytes, maintype="image", subtype="jpeg", cid=cid
        )

    port = int(cfg.get("smtp_port", 587))
    use_starttls = cfg.get("smtp_use_tls", "true") == "true" and port != 465
    use_tls = port == 465

    await aiosmtplib.send(
        msg,
        hostname=cfg["smtp_host"],
        port=port,
        username=cfg["smtp_username"] or None,
        password=cfg["smtp_password"] or None,
        start_tls=use_starttls,
        use_tls=use_tls,
        timeout=15,
    )
    return {"sent": True, "to": cfg["alert_to_email"], "subject": msg["Subject"]}


async def check_and_send_alert(result: Dict[str, Any], image_bytes: bytes) -> None:
    if not result.get("detected"):
        return

    person_id = result["person_id"]
    ok, reason, ctx = should_send_alert(person_id)
    if not ok:
        return

    person = ctx["person"]
    try:
        res = await send_alert_email(
            person_name=person["name"],
            person_type=person["person_type"],
            custom_tag=person.get("custom_tag"),
            camera_name=ctx["camera_name"],
            captured_at=person["last_seen"],
            quality_score=result["quality_score"],
            similarity=result["similarity"],
            image_bytes=image_bytes,
            cfg=ctx["cfg"],
        )
        print(f"[EMAIL ALERT] Sent alert for {person['name']} to {res['to']}")
    except Exception as e:
        print(f"[EMAIL ALERT ERROR] Failed to send email: {e}")