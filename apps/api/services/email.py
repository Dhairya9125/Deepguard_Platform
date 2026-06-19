import os
import smtplib
from email.message import EmailMessage
import logging

logger = logging.getLogger(__name__)

def send_reset_password_email(to_email: str, reset_link: str):
    smtp_host = os.environ.get("SMTP_HOST", "smtp.gmail.com")
    smtp_port = int(os.environ.get("SMTP_PORT", 587))
    smtp_user = os.environ.get("SMTP_USER")
    smtp_pass = os.environ.get("SMTP_PASS")
    
    if not smtp_user or not smtp_pass:
        logger.error("SMTP_USER or SMTP_PASS not set. Cannot send email.")
        return False

    msg = EmailMessage()
    msg['Subject'] = "TRUX - Password Reset Request"
    msg['From'] = f"TRUX <{smtp_user}>"
    msg['To'] = to_email

    html_content = f"""
    <div style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto; background-color: #050816; color: #fff; padding: 40px; border-radius: 12px; border: 1px solid #1a1f3c;">
        <div style="text-align: center; border-bottom: 1px solid #1a1f3c; padding-bottom: 20px; margin-bottom: 20px;">
            <h1 style="color: #66E3FF; margin: 0;">TRUX</h1>
            <p style="color: #888; font-size: 14px; margin-top: 5px;">DeepGuard Platform</p>
        </div>
        
        <h2>Password Reset</h2>
        <p>We received a request to reset your password. Click the button below to choose a new password.</p>
        
        <div style="text-align: center; margin: 30px 0;">
            <a href="{reset_link}" style="display: inline-block; padding: 12px 30px; background: linear-gradient(to right, #4D7CFE, #66E3FF); color: #000; text-decoration: none; border-radius: 6px; font-weight: bold; font-size: 14px;">Reset Password</a>
        </div>
        
        <p style="color: #888; font-size: 12px;">If you did not request this, please ignore this email. This link will expire in 15 minutes.</p>
        <p style="color: #888; font-size: 12px;">For security reasons, this link can only be used once.</p>
    </div>
    """
    msg.set_content("Please enable HTML to view this email.")
    msg.add_alternative(html_content, subtype='html')

    try:
        with smtplib.SMTP(smtp_host, smtp_port) as server:
            server.starttls()
            server.login(smtp_user, smtp_pass)
            server.send_message(msg)
        logger.info(f"Password reset email sent to {to_email}")
        return True
    except Exception as e:
        logger.error(f"Failed to send email to {to_email}: {e}")
        return False
