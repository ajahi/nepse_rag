# notifier.py — pluggable human-notification channels (email now; SMS/Telegram/WhatsApp later)
import os
import ssl
import smtplib
import asyncio
import logging
from abc import ABC, abstractmethod
from email.message import EmailMessage
from typing import List

logger = logging.getLogger("MessengerRAG.notifier")


class Notifier(ABC):
    name = "base"

    @abstractmethod
    async def send(self, subject: str, body: str) -> bool:
        ...


class EmailNotifier(Notifier):
    name = "email"

    def __init__(self, host=None, port=None, user=None,
                 password=None, sender=None, recipients=None):
        self.host = host or os.environ.get("SMTP_HOST", "smtp.gmail.com")
        self.port = int(port or os.environ.get("SMTP_PORT", "587"))
        self.user = user or os.environ.get("SMTP_USER", "hachuwakhikhi@gmail.com")
        self.password = password or os.environ.get("SMTP_PASS", "eblm chxo iixj xeav")
        self.sender = sender or os.environ.get("ALERT_FROM", "hachuwakhikhi@gmail.com")
        if recipients is None:
            recipients = os.environ.get("ALERT_TO", "himaliamit1@gmail.com")
        if isinstance(recipients, str):
            recipients = [r.strip() for r in recipients.split(",") if r.strip()]
        self.recipients = recipients

    def _enabled(self) -> bool:
        return bool(self.user and self.password and self.recipients)

    def _send_blocking(self, subject: str, body: str):
        msg = EmailMessage()
        msg["Subject"] = subject
        msg["From"] = self.sender
        msg["To"] = ", ".join(self.recipients)
        msg.set_content(body)
        ctx = ssl.create_default_context()
        with smtplib.SMTP(self.host, self.port, timeout=15) as s:
            s.starttls(context=ctx)
            s.login(self.user, self.password)
            s.send_message(msg)

    async def send(self, subject: str, body: str) -> bool:
        if not self._enabled():
            logger.warning("EmailNotifier not configured (SMTP_USER/SMTP_PASS/ALERT_TO missing)")
            return False
        try:
            # smtplib is blocking -> push to a thread so it doesn't stall the event loop
            await asyncio.to_thread(self._send_blocking, subject, body)
            logger.info(f"Email alert sent to {self.recipients}")
            return True
        except Exception as e:
            logger.error(f"Email alert failed: {e}", exc_info=True)
            return False


class NotificationManager:
    """Fan a single alert out to every configured channel. Add SMS/Telegram later."""
    def __init__(self, notifiers: List[Notifier]):
        self.notifiers = notifiers

    async def notify(self, subject: str, body: str) -> bool:
        results = await asyncio.gather(
            *(n.send(subject, body) for n in self.notifiers),
            return_exceptions=True,
        )
        return any(r is True for r in results)
