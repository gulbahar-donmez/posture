from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, EmailStr
import os
from datetime import datetime
import json
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from dotenv import load_dotenv

load_dotenv()

router = APIRouter(prefix="/contact", tags=["Contact"])

class ContactMessage(BaseModel):
    name: str
    email: EmailStr
    message: str

class ContactResponse(BaseModel):
    success: bool
    message: str
    timestamp: datetime

CONTACT_MESSAGES_FILE = "contact_messages.json"

def save_contact_message(contact_data: dict):
    try:
        if os.path.exists(CONTACT_MESSAGES_FILE):
            with open(CONTACT_MESSAGES_FILE, 'r', encoding='utf-8') as f:
                messages = json.load(f)
        else:
            messages = []

        contact_data['timestamp'] = datetime.now().isoformat()
        contact_data['id'] = len(messages) + 1
        messages.append(contact_data)

        with open(CONTACT_MESSAGES_FILE, 'w', encoding='utf-8') as f:
            json.dump(messages, f, ensure_ascii=False, indent=2)
        
        return True
    except Exception as e:
        print(f"Error saving contact message: {e}")
        return False

def send_email_notification(contact_data: dict):
    try:
        #kontrol için
        print("=== YENİ İLETİŞİM MESAJI ===")
        print(f"Ad Soyad: {contact_data['name']}")
        print(f"E-posta: {contact_data['email']}")
        print(f"Mesaj: {contact_data['message']}")
        print(f"Tarih: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        try:
            smtp_server = os.getenv("MAIL_SERVER", "smtp.gmail.com")
            smtp_port = int(os.getenv("MAIL_PORT", "587"))
            smtp_username = os.getenv("MAIL_USERNAME")
            smtp_password = os.getenv("MAIL_PASSWORD")
            support_email = os.getenv("MAIL_FROM", smtp_username)
            
            if not all([smtp_username, smtp_password]):
                print("Email ayarları eksik - sadece konsola yazdırılıyor")
                return True

            msg = MIMEMultipart('alternative')
            msg['Subject'] = f"PostureGuard İletişim Formu - {contact_data['name']}"
            msg['From'] = smtp_username
            msg['To'] = support_email
            msg['Reply-To'] = contact_data['email']
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <meta charset="UTF-8">
                <style>
                    body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); }}
                    .container {{ max-width: 600px; margin: 0 auto; background: white; border-radius: 15px; overflow: hidden; box-shadow: 0 20px 40px rgba(0,0,0,0.1); }}
                    .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 30px; text-align: center; }}
                    .header h1 {{ margin: 0; font-size: 28px; font-weight: 300; }}
                    .content {{ padding: 30px; }}
                    .field {{ margin-bottom: 20px; padding: 15px; background: #f8f9fa; border-radius: 8px; border-left: 4px solid #667eea; }}
                    .field strong {{ color: #333; font-weight: 600; }}
                    .message-box {{ background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%); padding: 20px; border-radius: 10px; margin: 20px 0; border: 1px solid #dee2e6; }}
                    .footer {{ background: #f8f9fa; padding: 20px; text-align: center; color: #6c757d; font-size: 14px; }}
                    .reply-button {{ display: inline-block; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 12px 25px; text-decoration: none; border-radius: 25px; margin: 10px 5px; font-weight: 500; }}
                </style>
            </head>
            <body>
                <div class="container">
                    <div class="header">
                        <h1>📧 PostureGuard İletişim Formu</h1>
                        <p>Yeni bir mesaj alındı</p>
                    </div>
                    <div class="content">
                        <div class="field">
                            <strong>👤 Gönderen:</strong> {contact_data['name']}
                        </div>
                        <div class="field">
                            <strong>📧 E-posta:</strong> {contact_data['email']}
                        </div>
                        <div class="field">
                            <strong>📅 Tarih:</strong> {datetime.now().strftime('%d.%m.%Y %H:%M:%S')}
                        </div>
                        <div class="message-box">
                            <strong>💬 Mesaj:</strong><br><br>
                            {contact_data['message'].replace('\n', '<br>')}
                        </div>
                        <div style="text-align: center; margin-top: 30px;">
                            <a href="mailto:{contact_data['email']}?subject=PostureGuard Destek - Yanıt" class="reply-button">
                                Hızlı Yanıt
                            </a>
                        </div>
                    </div>
                    <div class="footer">
                        <p>Bu mesaj PostureGuard iletişim formu aracılığıyla gönderilmiştir.</p>
                        <p><strong>PostureGuard</strong> - Duruş Analizi ve Sağlık Takip Sistemi</p>
                    </div>
                </div>
            </body>
            </html>
            """

            text_content = f"""
PostureGuard İletişim Formu - Yeni Mesaj

Gönderen: {contact_data['name']}
E-posta: {contact_data['email']}
Tarih: {datetime.now().strftime('%d.%m.%Y %H:%M:%S')}

Mesaj:
{contact_data['message']}

---
Bu mesaj PostureGuard iletişim formu aracılığıyla gönderilmiştir.
Yanıtlamak için: {contact_data['email']}
            """

            part1 = MIMEText(text_content, 'plain', 'utf-8')
            part2 = MIMEText(html_content, 'html', 'utf-8')
            
            msg.attach(part1)
            msg.attach(part2)
            with smtplib.SMTP(smtp_server, smtp_port) as server:
                server.starttls()
                server.login(smtp_username, smtp_password)
                server.send_message(msg)
            return True
            
        except Exception as email_error:
            return True
            
    except Exception as e:
        print(f"Error in email notification: {e}")
        return False

@router.post("/send-message", response_model=ContactResponse)
async def send_contact_message(contact_message: ContactMessage):
    try:
        contact_data = {
            "name": contact_message.name,
            "email": contact_message.email,
            "message": contact_message.message
        }

        saved = save_contact_message(contact_data)
        if not saved:
            raise HTTPException(status_code=500, detail="Mesaj kaydedilemedi")

        send_email_notification(contact_data)
        
        return ContactResponse(
            success=True,
            message="Mesajınız başarıyla alındı! En kısa sürede size dönüş yapacağız.",
            timestamp=datetime.now()
        )
        
    except Exception as e:
        print(f"Contact message error: {e}")
        raise HTTPException(status_code=500, detail="Mesaj gönderilemedi. Lütfen tekrar deneyin.")

@router.get("/messages")
async def get_contact_messages():
    try:
        if os.path.exists(CONTACT_MESSAGES_FILE):
            with open(CONTACT_MESSAGES_FILE, 'r', encoding='utf-8') as f:
                messages = json.load(f)
            return {"success": True, "messages": messages}
        else:
            return {"success": True, "messages": []}
    except Exception as e:
        raise HTTPException(status_code=500, detail="Mesajlar getirilemedi")

@router.get("/health")
async def contact_health_check():
    return {
        "service": "Contact Service",
        "status": "healthy",
        "timestamp": datetime.now(),
        "endpoints": [
            "/contact/send-message",
            "/contact/messages",
            "/contact/health"
        ]
    }
