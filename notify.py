from datetime import datetime
from email.mime.text import MIMEText
from pathlib import Path
import os
import smtplib
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
if (BASE_DIR / '.env').exists():
    load_dotenv(BASE_DIR / '.env', override=True)

LOG_DIR = Path(os.getenv('LOG_DIR', '/tmp/logs'))
LOG_DIR.mkdir(parents=True, exist_ok=True)
ALERT_LOG = LOG_DIR / 'alerts.log'


def _append(message: str):
    with ALERT_LOG.open('a', encoding='utf-8') as handle:
        handle.write(f"{datetime.utcnow().isoformat()}Z {message}\n")


def send_email_alert(query: str):
    sender = os.getenv('EMAIL_USER', '').strip()
    password = os.getenv('EMAIL_PASS', '').strip()
    receiver = os.getenv('ALERT_EMAIL', '').strip() or sender
    if not sender or not password or not receiver:
        message = f'logged escalation for query={query}'
        _append(message)
        return {'mode': 'log_only', 'message': message, 'log_file': str(ALERT_LOG)}
    msg = MIMEText(f"Escalation Alert\n\nQuery: {query}")
    msg['Subject'] = 'ServiceDesk Escalation'
    msg['From'] = sender
    msg['To'] = receiver
    try:
        with smtplib.SMTP('smtp.gmail.com', 587, timeout=20) as server:
            server.starttls()
            server.login(sender, password)
            server.sendmail(sender, [receiver], msg.as_string())
        message = f'email sent to {receiver}'
        _append(message)
        return {'mode': 'email', 'message': message, 'log_file': str(ALERT_LOG)}
    except Exception as exc:
        message = f'email failed and was logged: {exc}'
        _append(message)
        return {'mode': 'log_only', 'message': message, 'log_file': str(ALERT_LOG)}
