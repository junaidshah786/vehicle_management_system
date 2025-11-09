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


def generate_vehicle_sticker_pdf(vehicle_id: str, vehicle: dict, logo_path: str = "E:/workspace/poc/vehicle_management_system/app/image utils/i_card_logo.png") -> BytesIO:
    try:
        # Generate QR code
        qr_data = f"{vehicle_id}__{vehicle.get('registrationNumber', 'N/A')}__{vehicle.get('vehicleType', 'N/A')}"
        qr = qrcode.make(qr_data).convert("RGB")

        # PDF setup
        pdf_buffer = BytesIO()
        c = canvas.Canvas(pdf_buffer, pagesize=A4)
        page_width, page_height = A4

        # Sticker dimensions (30% larger than I-Card)
        sticker_width = 4.55 * inch  # 3.5 * 1.3
        sticker_height = 7.15 * inch  # 5.5 * 1.3
        sticker_x = (page_width - sticker_width) / 2
        sticker_y = (page_height - sticker_height) / 2

        # Gradient Background Effect (using overlapping rectangles)
        c.setFillColorRGB(0.15, 0.35, 0.75)  # Deep blue
        c.roundRect(sticker_x, sticker_y, sticker_width, sticker_height, radius=20, stroke=0, fill=1)
        
        # Add lighter blue overlay for gradient effect
        c.setFillColorRGB(0.25, 0.45, 0.85)  # Lighter blue without transparency
        c.roundRect(sticker_x, sticker_y + sticker_height * 0.3, sticker_width, sticker_height * 0.7, radius=20, stroke=0, fill=1)

        # Decorative border
        c.setStrokeColorRGB(1, 1, 1)  # White border
        c.setLineWidth(3)
        c.roundRect(sticker_x + 10, sticker_y + 10, sticker_width - 20, sticker_height - 20, radius=15, stroke=1, fill=0)

        # Title at top (no logo)
        title_y = sticker_y + sticker_height - 50
        c.setFont("Helvetica-Bold", 20)
        c.setFillColorRGB(1, 1, 1)  # White text
        c.drawCentredString(sticker_x + sticker_width / 2, title_y, "RaahSair Verified")

        # QR Code (Larger size) - Better positioned
        qr_size = 220
        qr_x = sticker_x + (sticker_width - qr_size) / 2
        qr_y = title_y - qr_size - 30
        
        # White background for QR code
        c.setFillColorRGB(1, 1, 1)
        c.roundRect(qr_x - 12, qr_y - 12, qr_size + 24, qr_size + 24, radius=15, stroke=0, fill=1)
        
        c.drawInlineImage(qr, qr_x, qr_y, width=qr_size, height=qr_size)

        # "Scan Me" text under QR code - with good spacing
        scan_y = qr_y - 45
        c.setFont("Helvetica-Bold", 24)
        c.setFillColorRGB(1, 0.95, 0.3)  # Bright yellow
        c.drawCentredString(sticker_x + sticker_width / 2, scan_y, "SCAN ME")

        # Registration Number Section - Enhanced design with box
        reg_section_y = scan_y - 75
        box_width = sticker_width - 80
        box_height = 85
        box_x = sticker_x + 40
        box_y = reg_section_y - 65  # Adjusted for equal vertical spacing
        
        # Rounded box background for registration info
        c.setFillColorRGB(1, 1, 1)  # Solid white with reduced opacity
        c.setFillColorRGB(0.2, 0.5, 0.9)  # Solid blue background
        c.roundRect(box_x, box_y, box_width, box_height, radius=12, stroke=0, fill=1)
        
        # Decorative border for the box
        c.setStrokeColorRGB(1, 0.95, 0.3)  # Yellow border
        c.setLineWidth(2.5)
        c.roundRect(box_x, box_y, box_width, box_height, radius=12, stroke=1, fill=0)
        
        # Calculate center point of the box for equal spacing
        box_center_y = box_y + (box_height / 2)
        
        # Registration Number - centered in box
        c.setFont("Helvetica", 11)
        c.setFillColorRGB(1, 1, 1)  # White text
        c.drawCentredString(sticker_x + sticker_width / 2, box_center_y + 20, "VEHICLE REG. NO.")
        
        c.setFont("Helvetica-Bold", 26)
        c.setFillColorRGB(1, 0.95, 0.3)  # Bright yellow
        registration_number = vehicle.get('registrationNumber', 'N/A').upper()
        c.drawCentredString(sticker_x + sticker_width / 2, box_center_y - 10, registration_number)

        # Vehicle Shift with icon-like design - centered in box
        vehicle_shift = vehicle.get('vehicleShift', '')
        if vehicle_shift:
            c.setFont("Helvetica-Bold", 13)
            c.setFillColorRGB(1, 1, 1)
            shift_text = f"● {vehicle_shift.upper()} SHIFT"
            c.drawCentredString(sticker_x + sticker_width / 2, box_center_y - 35, shift_text)

        # Finalize
        c.save()
        pdf_buffer.seek(0)
        return pdf_buffer
    
    except Exception as e:
        logging.error(f"Error generating vehicle sticker PDF: {e}")
        raise RuntimeError("Failed to generate vehicle sticker PDF") from e


