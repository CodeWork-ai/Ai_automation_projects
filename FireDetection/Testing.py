import cv2
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from ultralytics import YOLO
from collections import defaultdict
import numpy as np
import os
from datetime import datetime
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak
from reportlab.platypus import Image as RLImage
from reportlab.lib.enums import TA_CENTER, TA_LEFT

# ===================== CONFIGURATION =====================
YOLO_MODEL_PATH = r"D:\CODEWORK_PROJECTS\FireDetection\best.pt"
CNN_MODEL_PATH = r"D:\CODEWORK_PROJECTS\FireDetection\models\best_model.pth"
VIDEO_PATH = r"D:\CODEWORK_PROJECTS\FireDetection\Predictions\1036-142621335_tiny_results\1036-142621335_tiny.mp4"
OUTPUT_BASE_DIR = r"D:\CODEWORK_PROJECTS\FireDetection\Predictions"
YOLO_CONFIDENCE_THRESHOLD = 0.5

# Class names matching your trained model
CLASS_NAMES = [
    'Electrical_Fire_Cls_C',
    'Gas_Fire_cls_B',
    'Wood_fire_cls_A',
    'cooking_fire_Cls_K',
    'metal_fire_cls_D'
]

CLASS_DISPLAY_NAMES = [
    'Electrical Fire (Class C)',
    'Gas Fire (Class B)',
    'Wood Fire (Class A)',
    'Cooking Fire (Class K)',
    'Metal Fire (Class D)'
]


# ===================== CNN MODEL DEFINITION =====================
class FireCNN(nn.Module):
    def __init__(self, num_classes):
        super(FireCNN, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.3),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# ===================== LOAD MODELS =====================
print("Loading models...")
yolo_model = YOLO(YOLO_MODEL_PATH)
print("✓ YOLO loaded (Fire Detection: YES/NO)")

cnn_model = FireCNN(num_classes=len(CLASS_NAMES))
cnn_model.load_state_dict(torch.load(CNN_MODEL_PATH, map_location='cpu'))
cnn_model.eval()
print("✓ CNN loaded (Fire Classification)")
print(f"  Classes: {CLASS_DISPLAY_NAMES}")

test_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# ===================== FIRE DETECTION (YOLO) =====================
def is_fire_present(frame):
    results = yolo_model(frame, verbose=False)
    fire_bboxes = []
    max_confidence = 0.0

    for result in results:
        boxes = result.boxes
        for box in boxes:
            cls = int(box.cls[0].item())
            if cls == 0:
                conf = box.conf[0].item()
                if conf >= YOLO_CONFIDENCE_THRESHOLD:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    fire_bboxes.append((x1, y1, x2, y2))
                    max_confidence = max(max_confidence, conf)

    fire_detected = len(fire_bboxes) > 0
    return fire_detected, max_confidence, fire_bboxes


# ===================== FIRE CLASSIFICATION (CNN) =====================
def classify_fire_type(frame):
    try:
        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        img_tensor = test_transform(pil_img).unsqueeze(0)

        with torch.no_grad():
            outputs = cnn_model(img_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)

        predicted_idx = predicted.item()

        return {
            'class': CLASS_NAMES[predicted_idx],
            'display_name': CLASS_DISPLAY_NAMES[predicted_idx],
            'class_index': predicted_idx,
            'confidence': confidence.item(),
            'all_probabilities': {CLASS_DISPLAY_NAMES[i]: probabilities[0][i].item() * 100
                                  for i in range(len(CLASS_NAMES))},
            'success': True
        }
    except Exception as e:
        return {'success': False, 'error': str(e)}


# ===================== PDF REPORT GENERATION =====================
def generate_pdf_report(results_data, output_pdf_path):
    """Generate comprehensive PDF report"""

    doc = SimpleDocTemplate(output_pdf_path, pagesize=letter,
                            rightMargin=72, leftMargin=72,
                            topMargin=72, bottomMargin=18)

    story = []
    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#1a1a1a'),
        spaceAfter=30,
        alignment=TA_CENTER,
        fontName='Helvetica-Bold'
    )

    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=16,
        textColor=colors.HexColor('#2c3e50'),
        spaceAfter=12,
        spaceBefore=12,
        fontName='Helvetica-Bold'
    )

    # Title
    story.append(Paragraph("Fire Detection & Classification Report", title_style))
    story.append(Spacer(1, 0.2 * inch))

    # Timestamp
    timestamp = datetime.now().strftime("%B %d, %Y at %H:%M:%S")
    story.append(Paragraph(f"<i>Generated: {timestamp}</i>", styles['Normal']))
    story.append(Spacer(1, 0.3 * inch))

    # ===== VIDEO INFORMATION =====
    story.append(Paragraph("Video Information", heading_style))

    video_info = [
        ['Parameter', 'Value'],
        ['Video Name', results_data['video_name']],
        ['Resolution', f"{results_data['width']}x{results_data['height']}"],
        ['Frame Rate', f"{results_data['fps']} FPS"],
        ['Total Frames', str(results_data['total_frames'])],
        ['Duration', f"{results_data['duration']:.2f} seconds"]
    ]

    video_table = Table(video_info, colWidths=[2.5 * inch, 3.5 * inch])
    video_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#3498db')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 10),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey])
    ]))
    story.append(video_table)
    story.append(Spacer(1, 0.3 * inch))

    # ===== DETECTION SUMMARY =====
    story.append(Paragraph("Detection Summary", heading_style))

    detection_info = [
        ['Metric', 'Value'],
        ['Frames with Fire Detected', f"{results_data['frames_with_fire']} ({results_data['fire_percentage']:.1f}%)"],
        ['Frames without Fire', f"{results_data['frames_without_fire']} ({results_data['no_fire_percentage']:.1f}%)"],
        ['Successfully Classified', f"{results_data['classified_frames']} frames"]
    ]

    detection_table = Table(detection_info, colWidths=[3 * inch, 3 * inch])
    detection_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#e74c3c')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 10),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey])
    ]))
    story.append(detection_table)
    story.append(Spacer(1, 0.4 * inch))

    # ===== FINAL WINNER =====
    story.append(Paragraph("🏆 Final Classification Result", heading_style))

    winner_data = [
        ['Classification', 'Details'],
        ['WINNER', results_data['winner_class']],
        ['Occurrences', f"{results_data['winner_count']} frames ({results_data['winner_percentage']:.1f}%)"],
        ['Average Confidence', f"{results_data['winner_confidence']:.2f}%"],
        ['Recommendation', results_data['recommendation']]
    ]

    winner_table = Table(winner_data, colWidths=[2 * inch, 4 * inch])
    winner_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#27ae60')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (0, -1), colors.lightgreen),
        ('BACKGROUND', (1, 1), (1, -1), colors.white),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 10),
        ('FONTNAME', (1, 1), (1, 1), 'Helvetica-Bold'),
        ('FONTSIZE', (1, 1), (1, 1), 14),
        ('TEXTCOLOR', (1, 1), (1, 1), colors.HexColor('#27ae60'))
    ]))
    story.append(winner_table)
    story.append(Spacer(1, 0.3 * inch))

    # ===== CLASSIFICATION DISTRIBUTION =====
    story.append(Paragraph("Classification Distribution", heading_style))

    distribution_data = [['Fire Type', 'Frames', 'Percentage', 'Avg Confidence']]
    for entry in results_data['distribution']:
        distribution_data.append([
            entry['class'],
            str(entry['count']),
            f"{entry['percentage']:.1f}%",
            f"{entry['avg_confidence']:.1f}%"
        ])

    dist_table = Table(distribution_data, colWidths=[2.5 * inch, 1 * inch, 1.2 * inch, 1.3 * inch])
    dist_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#34495e')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 11),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 9),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey])
    ]))
    story.append(dist_table)
    story.append(Spacer(1, 0.3 * inch))

    # ===== FINAL FRAME PROBABILITIES =====
    if results_data['final_probs']:
        story.append(Paragraph("Final Frame Probabilities", heading_style))

        probs_data = [['Fire Type', 'Probability']]
        for entry in results_data['final_probs']:
            probs_data.append([entry['class'], f"{entry['probability']:.2f}%"])

        probs_table = Table(probs_data, colWidths=[3.5 * inch, 2.5 * inch])
        probs_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#9b59b6')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 11),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 10),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey])
        ]))
        story.append(probs_table)

    story.append(Spacer(1, 0.5 * inch))

    # Footer
    footer_text = f"<i>Output video saved as: detected_video.mp4</i>"
    story.append(Paragraph(footer_text, styles['Normal']))

    # Build PDF
    doc.build(story)
    print(f"✓ PDF Report generated: {output_pdf_path}")


# ===================== VIDEO PROCESSING =====================
def process_video(video_path):
    """Process video and generate outputs"""

    # Extract video name
    video_name = os.path.splitext(os.path.basename(video_path))[0]

    # Create output directory
    output_dir = os.path.join(OUTPUT_BASE_DIR, f"{video_name}_results")
    os.makedirs(output_dir, exist_ok=True)

    output_video_path = os.path.join(output_dir, "detected_video.mp4")
    output_pdf_path = os.path.join(output_dir, "detection_result.pdf")

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Cannot open video {video_path}")
        return

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps

    print(f"\n{'=' * 70}")
    print(f"Video: {fps} FPS | {width}x{height} | {total_frames} frames")
    print(f"{'=' * 70}\n")

    # Video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    frame_number = 0
    frames_with_fire = 0
    classification_history = []
    last_classification = None

    print("Processing video...")
    print("Pipeline: YOLO (fire?) → CNN (direct classification)\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_number += 1

        # STAGE 1: YOLO checks - Is there fire?
        fire_detected, yolo_confidence, fire_bboxes = is_fire_present(frame)

        current_classification = None

        if fire_detected:
            frames_with_fire += 1

            # STAGE 2: CNN classifies
            classification = classify_fire_type(frame)

            if classification['success']:
                current_classification = classification
                last_classification = classification

                classification_history.append({
                    'frame': frame_number,
                    'class': classification['class'],
                    'confidence': classification['confidence']
                })

        # ===================== VISUALIZATION =====================

        # Draw fire bounding boxes
        for bbox in fire_bboxes:
            x1, y1, x2, y2 = bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
            cv2.putText(frame, f"Fire {yolo_confidence:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # Draw status banner
        if current_classification:
            fire_display_name = current_classification['display_name']
            fire_class_short = fire_display_name.split('(')[0].strip()

            banner_height = 120
            cv2.rectangle(frame, (0, 0), (width, banner_height), (0, 150, 0), -1)
            cv2.rectangle(frame, (0, 0), (width, banner_height), (0, 255, 0), 5)

            main_text = f"FIRE TYPE: {fire_class_short}"
            cv2.putText(frame, main_text, (20, 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 255, 255), 3)

            cv2.putText(frame, fire_display_name, (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            detail_text = f"Confidence: {current_classification['confidence'] * 100:.1f}%"
            cv2.putText(frame, detail_text, (20, 108),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        elif fire_detected:
            banner_height = 70
            cv2.rectangle(frame, (0, 0), (width, banner_height), (0, 180, 180), -1)
            cv2.rectangle(frame, (0, 0), (width, banner_height), (0, 255, 255), 4)

            cv2.putText(frame, "FIRE DETECTED - Classification Error", (20, 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 3)
        else:
            cv2.rectangle(frame, (0, 0), (width, 65), (80, 80, 80), -1)
            cv2.putText(frame, "No Fire Detected", (20, 43),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2)

        # Bottom info bar
        info_bg_y = height - 50
        cv2.rectangle(frame, (0, info_bg_y), (width, height), (30, 30, 30), -1)

        info_text = f"Frame: {frame_number}/{total_frames} | Fire frames: {frames_with_fire}"
        cv2.putText(frame, info_text, (10, height - 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        if current_classification:
            top3_probs = sorted(current_classification['all_probabilities'].items(),
                                key=lambda x: x[1], reverse=True)[:3]
            prob_text = " | ".join([f"{cls.split('(')[0].strip()}: {prob:.0f}%"
                                    for cls, prob in top3_probs])
            cv2.putText(frame, prob_text, (10, height - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        out.write(frame)

        # Progress updates
        if frame_number % 30 == 0:
            progress = frame_number / total_frames * 100
            print(f"Progress: {frame_number}/{total_frames} ({progress:.1f}%)", end='')
            if current_classification:
                print(
                    f" | Current: {current_classification['display_name']} ({current_classification['confidence'] * 100:.1f}%)")
            else:
                print()

    cap.release()
    out.release()

    # ===================== FINAL ANALYSIS =====================
    print("\n" + "=" * 70)
    print("PROCESSING COMPLETE")
    print("=" * 70)
    print(f"Total Frames: {frame_number}")
    print(f"Frames with Fire: {frames_with_fire} ({frames_with_fire / frame_number * 100:.1f}%)")
    print(f"Classified Frames: {len(classification_history)}")

    results_data = {
        'video_name': video_name,
        'width': width,
        'height': height,
        'fps': fps,
        'total_frames': total_frames,
        'duration': duration,
        'frames_with_fire': frames_with_fire,
        'frames_without_fire': frame_number - frames_with_fire,
        'fire_percentage': frames_with_fire / frame_number * 100,
        'no_fire_percentage': (frame_number - frames_with_fire) / frame_number * 100,
        'classified_frames': len(classification_history),
        'distribution': [],
        'final_probs': []
    }

    if classification_history:
        # Calculate statistics
        class_counts = defaultdict(int)
        total_confidence = defaultdict(float)

        for entry in classification_history:
            class_counts[entry['class']] += 1
            total_confidence[entry['class']] += entry['confidence']

        # Determine winner
        most_common_class = max(class_counts, key=class_counts.get)
        most_common_idx = CLASS_NAMES.index(most_common_class)
        most_common_display = CLASS_DISPLAY_NAMES[most_common_idx]

        winner_count = class_counts[most_common_class]
        winner_percentage = winner_count / len(classification_history) * 100
        winner_confidence = total_confidence[most_common_class] / winner_count * 100

        # Recommendation based on class
        recommendations = {
            'Electrical_Fire_Cls_C': 'Use CO2 or Dry Chemical extinguisher. Do NOT use water!',
            'Gas_Fire_cls_B': 'Use CO2, Dry Chemical, or Foam extinguisher.',
            'Wood_fire_cls_A': 'Use Water or Foam extinguisher.',
            'cooking_fire_Cls_K': 'Use Wet Chemical extinguisher. Do NOT use water!',
            'metal_fire_cls_D': 'Use specialized Dry Powder extinguisher for metal fires.'
        }

        results_data['winner_class'] = most_common_display
        results_data['winner_count'] = winner_count
        results_data['winner_percentage'] = winner_percentage
        results_data['winner_confidence'] = winner_confidence
        results_data['recommendation'] = recommendations.get(most_common_class, 'Consult fire safety expert.')

        print(f"\n🔥 OVERALL CLASSIFICATION:")
        print(f"   Most Common: {most_common_display}")
        print(f"   Occurrences: {winner_count} frames ({winner_percentage:.1f}%)")
        print(f"   Avg Confidence: {winner_confidence:.2f}%")

        print(f"\n🏆 FINAL WINNER: {most_common_display}")
        print(f"   Recommendation: {results_data['recommendation']}")

        # Distribution
        print(f"\n   Classification Distribution:")
        for cls in sorted(class_counts.keys(), key=lambda x: class_counts[x], reverse=True):
            count = class_counts[cls]
            percentage = count / len(classification_history) * 100
            avg_conf = total_confidence[cls] / count * 100
            cls_idx = CLASS_NAMES.index(cls)
            display_name = CLASS_DISPLAY_NAMES[cls_idx]
            print(f"      {display_name}: {count} frames ({percentage:.1f}%) - Avg Conf: {avg_conf:.1f}%")

            results_data['distribution'].append({
                'class': display_name,
                'count': count,
                'percentage': percentage,
                'avg_confidence': avg_conf
            })

        # Final frame probabilities
        if last_classification:
            print(f"\n   Final Frame Probabilities:")
            sorted_probs = sorted(last_classification['all_probabilities'].items(),
                                  key=lambda x: x[1], reverse=True)
            for cls, prob in sorted_probs:
                print(f"      {cls}: {prob:.2f}%")
                results_data['final_probs'].append({
                    'class': cls,
                    'probability': prob
                })
    else:
        print("\n✓ No fire detected in video")

    print(f"\n✓ Output video saved: {output_video_path}")

    # Generate PDF Report
    if classification_history:
        generate_pdf_report(results_data, output_pdf_path)

    print(f"✓ All outputs saved in: {output_dir}")
    print("=" * 70)


# ===================== MAIN =====================
if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("DIRECT FIRE DETECTION & CLASSIFICATION PIPELINE")
    print("=" * 70)
    print("Stage 1: YOLO - Detect fire presence (YES/NO)")
    print("Stage 2: CNN - Classify fire type (DIRECT, NO SMOOTHING):")
    for i, (class_name, display_name) in enumerate(zip(CLASS_NAMES, CLASS_DISPLAY_NAMES)):
        print(f"         {i}. {display_name}")
    print("=" * 70)

    process_video(VIDEO_PATH)