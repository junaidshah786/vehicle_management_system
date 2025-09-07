import logging
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader
from reportlab.lib import colors
from io import BytesIO
from PIL import Image
import qrcode
import os
from reportlab.lib.units import inch


def generate_vehicle_icard_pdf(vehicle_id: str, vehicle: dict, logo_path: str = "E:/workspace/poc/vehicle_management_system/app/image utils/i_card_logo.png") -> BytesIO:
    try:
        # Generate QR code
        qr_data = f"{vehicle_id}__{vehicle.get('registrationNumber', 'N/A')}__{vehicle.get('vehicleType', 'N/A')}"
        qr = qrcode.make(qr_data).convert("RGB")

        # PDF setup
        pdf_buffer = BytesIO()
        c = canvas.Canvas(pdf_buffer, pagesize=A4)
        page_width, page_height = A4

        # I-Card dimensions
        card_width = 3.5 * inch
        card_height = 5.5 * inch
        card_x = (page_width - card_width) / 2
        card_y = (page_height - card_height) / 2

        # I-Card Base
        c.setFillColor(colors.white)
        c.roundRect(card_x, card_y, card_width, card_height, radius=12, stroke=1, fill=1)

        # Top Band (Darker Blue Background)
        band_height = card_height * 0.28
        c.setFillColorRGB(0.2, 0.4, 0.8)  # Darker blue
        c.roundRect(card_x, card_y + card_height - band_height, card_width, band_height, radius=12, stroke=0, fill=1)

        # Watermark
        c.setFont("Helvetica-Bold", 24)
        c.setFillColorRGB(0.8, 0.8, 0.8)
        c.saveState()
        c.translate(card_x + card_width / 2, card_y + card_height / 2)
        c.rotate(0)
        c.drawCentredString(0, 0, "RaahSair Verified")
        c.restoreState()
        c.setFillColor(colors.black)

        # QR Code
        qr_size = 130
        qr_x = card_x + (card_width - qr_size) / 2
        qr_y = card_y + card_height - qr_size - 25
        c.drawInlineImage(qr, qr_x, qr_y, width=qr_size, height=qr_size)

        # Logo
        if os.path.exists(logo_path):
            logo = Image.open(logo_path).convert("RGBA")
            logo_reader = ImageReader(logo)
            logo_w, logo_h = 70, 70
            logo_x = card_x + (card_width - logo_w) / 2
            logo_y = qr_y - logo_h - 10
            c.drawImage(logo_reader, logo_x, logo_y, width=logo_w, height=logo_h, mask='auto')
        else:
            logo_y = qr_y - 80

        # Title
        title_y = logo_y - 20
        c.setFont("Helvetica-Bold", 12)
        c.drawCentredString(card_x + card_width / 2, title_y, "VEHICLE I-CARD")

        # Vehicle Info in Tabular Format
        labels = {
            "registrationNumber": "Registration No.",
            "vehicleType": "Vehicle Type",
            "vehicleShift": "Vehicle Shift",
            "ownerPhone": "Phone Number",
            "seatingCapacity": "Seating Capacity"
        }

        info_start_y = title_y - 30
        row_height = 18
        info_box_x = card_x + 15
        info_box_y = info_start_y - (len(labels) * row_height) + 8
        info_box_width = card_width - 30
        info_box_height = (len(labels) * row_height) + 8

        # Draw main border for info table
        c.setStrokeColor(colors.black)
        c.setLineWidth(0.7)
        c.roundRect(info_box_x, info_box_y, info_box_width, info_box_height, radius=6, stroke=1, fill=0)

        # Table column widths (adjust as needed)
        label_col_width = info_box_width * 0.45  # 45% for labels
        value_col_width = info_box_width * 0.55  # 55% for values
        
        # Draw table rows and content
        for i, (key, label) in enumerate(labels.items()):
            current_y = info_start_y - i * row_height
            
            # Draw horizontal lines between rows (except for the last row)
            if i > 0:
                c.line(info_box_x, current_y + 9, info_box_x + info_box_width, current_y + 9)
            
            # Draw vertical line to separate columns
            c.line(info_box_x + label_col_width, current_y + 9, info_box_x + label_col_width, current_y - 9)
            
            # Draw label (bold)
            c.setFont("Helvetica-Bold", 9.5)
            c.drawString(info_box_x + 8, current_y, label)
            
            # Draw value (regular)
            c.setFont("Helvetica", 9.5)
            value = str(vehicle.get(key, "N/A"))
            c.drawString(info_box_x + label_col_width + 8, current_y, value)

        # Footer
        c.setFont("Helvetica-Oblique", 8)
        c.setFillColor(colors.grey)
        c.drawCentredString(card_x + card_width / 2, card_y + 15, "Issued by RaahSair")

        # Finalize
        c.setFillColor(colors.black)
        c.save()
        pdf_buffer.seek(0)
        return pdf_buffer
    
    except Exception as e:
        logging.error(f"Error generating vehicle I-Card PDF: {e}")
        raise RuntimeError("Failed to generate vehicle I-Card PDF") from e




