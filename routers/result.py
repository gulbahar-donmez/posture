from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from datetime import datetime
from database import get_db
from models import LogRecord, User
from routers.auth import get_current_user
import schemas
from fastapi.responses import StreamingResponse
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import io
import os
import traceback
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from io import BytesIO

router = APIRouter(
    prefix="/results",
    tags=["Results"],
)

PRIMARY_COLOR = colors.HexColor('#7AB689')
LIGHT_COLOR = colors.HexColor('#AAD4AD')
LIGHTEST_COLOR = colors.HexColor('#F6FBF8')
DARK_COLOR = colors.HexColor('#3EA232')
DARKEST_COLOR = colors.HexColor('#5F8185')

def create_chart_image(analysis_results):
    try:
        if not analysis_results or len(analysis_results) < 2:
            return None

        from datetime import datetime, timedelta
        from sqlalchemy import func, cast, Date
        end_date = datetime.now()
        start_date = end_date - timedelta(days=6)
        daily_averages = []
        current_date = start_date.date()
        
        for i in range(7):
            day_results = [r for r in analysis_results 
                         if r.timestamp.date() == current_date]
            
            if day_results:
                avg_score = sum(r.personalized_score for r in day_results) / len(day_results)
                daily_averages.append({
                    'date': current_date.strftime('%d/%m'),
                    'score': round(avg_score, 1)
                })
            else:
                daily_averages.append({
                    'date': current_date.strftime('%d/%m'),
                    'score': 0
                })
            
            current_date += timedelta(days=1)
        
        dates = [day['date'] for day in daily_averages]
        scores = [day['score'] for day in daily_averages]

        plt.figure(figsize=(10, 6))
        plt.plot(dates, scores, marker='o', linewidth=2, markersize=6, color='#7AB689')
        plt.fill_between(dates, scores, alpha=0.3, color='#AAD4AD')
        
        plt.title('Son 7 Gün - Günlük Ortalama Skor', fontsize=16, fontweight='bold', color='#3EA232')
        plt.xlabel('Tarih', fontsize=12, color='#5F8185')
        plt.ylabel('Duruş Skoru (%)', fontsize=12, color='#5F8185')
        plt.grid(True, alpha=0.3, color='#AAD4AD')
        plt.ylim(0, 100)

        plt.xticks(rotation=45)
        plt.gca().set_facecolor('#F6FBF8')
        plt.tight_layout()
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight', 
                   facecolor='#F6FBF8', edgecolor='none')
        buffer.seek(0)
        plt.close()
        
        return buffer
    except Exception as e:
        print(f"Grafik oluşturma hatası: {e}")
        return None

def create_pdf_report(user_data, analysis_results, report_type="progress"):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    story = []
    try:
        pdfmetrics.registerFont(TTFont('Arial', 'arial.ttf'))
        turkish_font = 'Arial'
    except:
        try:
            pdfmetrics.registerFont(TTFont('DejaVuSans', 'DejaVuSans.ttf'))
            turkish_font = 'DejaVuSans'
        except:
            turkish_font = 'Helvetica'
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=20,
        spaceAfter=30,
        alignment=1,
        fontName=turkish_font,
        textColor=DARK_COLOR
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=14,
        spaceAfter=12,
        spaceBefore=12,
        fontName=turkish_font,
        textColor=PRIMARY_COLOR
    )
    
    normal_style = ParagraphStyle(
        'CustomNormal',
        parent=styles['Normal'],
        fontName=turkish_font,
        textColor=DARKEST_COLOR
    )

    try:
        logo_path = "frontend/public/report_logo.png"
        if os.path.exists(logo_path):
            logo = Image(logo_path, width=1*inch, height=1*inch)
            logo.hAlign = 'CENTER'
            story.append(logo)
            story.append(Spacer(1, 20))
    except Exception as e:
        print(f"Logo yükleme hatası: {e}")

    story.append(Paragraph("PostureGuard Analiz Raporu", title_style))
    story.append(Spacer(1, 20))
    story.append(Paragraph("Kullanıcı Bilgileri", heading_style))
    user_info = [
        ["Ad Soyad:", user_data.get('fullName', 'N/A')],
        ["Kullanıcı Adı:", user_data.get('username', 'N/A')],
        ["Rapor Tarihi:", datetime.now().strftime("%d/%m/%Y %H:%M")],
        ["Rapor Türü:", "İlerleme Raporu" if report_type == "progress" else "Analiz Raporu"]
    ]
    
    user_table = Table(user_info, colWidths=[2*inch, 4*inch])
    user_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), PRIMARY_COLOR),
        ('TEXTCOLOR', (0, 0), (0, -1), colors.white),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, -1), turkish_font),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
        ('BACKGROUND', (1, 0), (-1, -1), LIGHTEST_COLOR),
        ('GRID', (0, 0), (-1, -1), 1, LIGHT_COLOR)
    ]))
    story.append(user_table)
    story.append(Spacer(1, 20))

    if analysis_results and len(analysis_results) >= 2:
        story.append(Paragraph("İlerleme Grafiği", heading_style))
        
        chart_buffer = create_chart_image(analysis_results)
        if chart_buffer:
            chart_img = Image(chart_buffer, width=6*inch, height=3*inch)
            chart_img.hAlign = 'CENTER'
            story.append(chart_img)
            story.append(Spacer(1, 20))

    if analysis_results:
        story.append(PageBreak())
        story.append(Paragraph("Analiz Sonuçlarım", heading_style))
        story.append(Spacer(1, 10))

        recent_results = analysis_results[:10]
        if recent_results:
            result_data = [["Tarih", "Seviye", "Skor", "Analiz Türü"]]
            for result in recent_results:
                result_data.append([
                    result.timestamp.strftime("%d/%m/%Y") if hasattr(result.timestamp, 'strftime') else str(result.timestamp),
                    result.level,
                    f"{result.personalized_score:.1f}%",
                    result.analysis_type
                ])
            
            result_table = Table(result_data, colWidths=[1.8*inch, 1*inch, 1*inch, 1.2*inch])
            result_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), PRIMARY_COLOR),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, -1), turkish_font),
                ('FONTSIZE', (0, 0), (-1, -1), 8),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
                ('BACKGROUND', (0, 1), (-1, -1), LIGHTEST_COLOR),
                ('GRID', (0, 0), (-1, -1), 1, LIGHT_COLOR)
            ]))
            story.append(result_table)
            story.append(Spacer(1, 20))

    if analysis_results:
        story.append(Paragraph("İstatistikler", heading_style))
        
        total_analyses = len(analysis_results)
        avg_score = sum(r.personalized_score for r in analysis_results) / total_analyses if total_analyses > 0 else 0
        good_analyses = sum(1 for r in analysis_results if r.level == "İYİ")
        good_percentage = (good_analyses / total_analyses * 100) if total_analyses > 0 else 0
        
        stats_data = [
            ["Toplam Analiz:", str(total_analyses)],
            ["Ortalama Skor:", f"{avg_score:.1f}%"],
            ["İyi Duruş Oranı:", f"{good_percentage:.1f}%"]
        ]
        
        stats_table = Table(stats_data, colWidths=[2*inch, 2*inch])
        stats_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), PRIMARY_COLOR),
            ('TEXTCOLOR', (0, 0), (0, -1), colors.white),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, -1), turkish_font),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
            ('BACKGROUND', (1, 0), (-1, -1), LIGHTEST_COLOR),
            ('GRID', (0, 0), (-1, -1), 1, LIGHT_COLOR)
        ]))
        story.append(stats_table)
        story.append(Spacer(1, 20))
    story.append(Paragraph("Öneriler", heading_style))
    recommendations = [
        "• Düzenli analiz yaparak duruşunuzu takip edin",
        "• Kalibrasyon yaparak sistemin sizi daha iyi tanımasını sağlayın",
        "• Günlük hedeflerinizi gerçekleştirin",
        "• Risk faktörlerinizi azaltmaya odaklanın"
    ]
    
    for rec in recommendations:
        story.append(Paragraph(rec, normal_style))
    
    doc.build(story)
    buffer.seek(0)
    return buffer

@router.get("/daily-averages", status_code=status.HTTP_200_OK)
async def get_daily_averages(
    days: int = 7,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    try:
        from datetime import datetime, timedelta
        from sqlalchemy import func, cast, Date
        
        user_id = current_user.user_id
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days-1)
        daily_averages = (
            db.query(
                cast(LogRecord.timestamp, Date).label('date'),
                func.avg(LogRecord.confidence * 100).label('average_score'),
                func.count(LogRecord.log_id).label('analysis_count')
            )
            .filter(
                LogRecord.user_id == user_id,
                LogRecord.timestamp >= start_date,
                LogRecord.timestamp <= end_date
            )
            .group_by(cast(LogRecord.timestamp, Date))
            .order_by(cast(LogRecord.timestamp, Date))
            .all()
        )

        result_data = []
        current_date = start_date.date()
        
        for i in range(days):
            date_str = current_date.strftime('%Y-%m-%d')
            day_label = current_date.strftime('%d/%m')
            day_data = next((d for d in daily_averages if d.date == current_date), None)
            
            if day_data:
                result_data.append({
                    'date': date_str,
                    'day_label': day_label,
                    'average_score': round(float(day_data.average_score), 1),
                    'analysis_count': int(day_data.analysis_count)
                })
            else:
                result_data.append({
                    'date': date_str,
                    'day_label': day_label,
                    'average_score': 0,
                    'analysis_count': 0
                })
            
            current_date += timedelta(days=1)
        
        return {
            'daily_averages': result_data,
            'total_days': days,
            'days_with_analysis': len([d for d in result_data if d['analysis_count'] > 0])
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Günlük ortalamalar hesaplanırken hata: {str(e)}"
        )

@router.get("/time_series", response_model=schemas.PaginatedAnalysisResults, status_code=status.HTTP_200_OK)
async def get_time_series_results(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    page: int = 1,
    page_size: int = 20
):
    if page < 1 or page_size < 1:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Sayfa ve sayfa boyutu 1'den büyük olmalıdır."
        )

    user_id = current_user.user_id
    total_records = db.query(LogRecord).filter(LogRecord.user_id == user_id).count()

    results_db = (
        db.query(LogRecord)
        .filter(LogRecord.user_id == user_id)
        .order_by(LogRecord.timestamp.desc())
        .offset((page - 1) * page_size)
        .limit(page_size)
        .all()
    )
    results_transformed = []
    for log in results_db:
        body_type_parts = str(log.body_type).split('_')
        analysis_type = "Tam Vücut"
        body_type_display = str(log.body_type)
        
        if len(body_type_parts) >= 3:
            if body_type_parts[-1] == "full_body":
                analysis_type = "Tam Vücut"
            elif body_type_parts[-1] == "neck_upper":
                analysis_type = "Boyun-Üst Vücut"
            if len(body_type_parts) >= 3:
                body_type = body_type_parts[0]
                symmetry = body_type_parts[1]
                body_type_display = f"{body_type} ({symmetry} simetri)"

        results_transformed.append(schemas.AnalysisResult(
            log_id=int(log.log_id),
            timestamp=log.timestamp,
            level=str(log.level),
            confidence=float(log.confidence),
            personalized_score=round(float(log.confidence) * 100, 2),
            analysis_type=analysis_type,
            body_type_classification=body_type_display,
        ))

    return schemas.PaginatedAnalysisResults(
        total_records=total_records,
        page=page,
        page_size=page_size,
        results=results_transformed
    )

@router.get("/pdf-report", status_code=status.HTTP_200_OK)
async def generate_pdf_report(
    report_type: str = "progress",
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """PDF rapor oluştur ve indir"""
    try:
        user_logs = (
            db.query(LogRecord)
            .filter(LogRecord.user_id == current_user.user_id)
            .order_by(LogRecord.timestamp.desc())
            .all()
        )

        analysis_results = []
        for log in user_logs:
            body_type_parts = str(log.body_type).split('_')
            analysis_type = "Tam Vücut"
            body_type_display = str(log.body_type)
            
            if len(body_type_parts) >= 3:
                if body_type_parts[-1] == "full_body":
                    analysis_type = "Tam Vücut"
                elif body_type_parts[-1] == "neck_upper":
                    analysis_type = "Boyun-Üst Vücut"
                if len(body_type_parts) >= 3:
                    body_type = body_type_parts[0]
                    symmetry = body_type_parts[1]
                    body_type_display = f"{body_type} ({symmetry} simetri)"

            analysis_results.append(schemas.AnalysisResult(
                log_id=int(log.log_id),
                timestamp=log.timestamp,
                level=str(log.level),
                confidence=float(log.confidence),
                personalized_score=round(float(log.confidence) * 100, 2),
                analysis_type=analysis_type,
                body_type_classification=body_type_display,
            ))

        user_data = {
            'fullName': f"{current_user.firstname} {current_user.lastname}",
            'username': current_user.username,
            'email': current_user.email
        }

        pdf_buffer = create_pdf_report(user_data, analysis_results, report_type)
        pdf_data = pdf_buffer.getvalue()
        filename = f"posture_report_{current_user.username}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        
        return StreamingResponse(
            io.BytesIO(pdf_data),
            media_type='application/pdf',
            headers={
                'Content-Disposition': f'attachment; filename="{filename}"'
            }
        )
        
    except Exception as e:
        print("PDF Hatası:", str(e))
        traceback.print_exc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"PDF rapor oluşturulurken hata: {str(e)}"
        )