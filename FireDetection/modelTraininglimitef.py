import cv2
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from ultralytics import YOLO
from collections import defaultdict
import numpy as np

# ===================== CONFIGURATION =====================
YOLO_MODEL_PATH = r"D:\CODEWORK_PROJECTS\FireDetection\models\best.pt"
CNN_MODEL_PATH = r"D:\CODEWORK_PROJECTS\FireDetection\models\best_model.pth"
VIDEO_PATH = r"D:\CODEWORK_PROJECTS\FireDetection\Predictions\1036-142621335_tiny_results\1036-142621335_tiny.mp4"
YOLO_CONFIDENCE_THRESHOLD = 0.2  # YOLO: Is there fire? (YES/NO)

# Class names matching your trained model (exact order from training)
CLASS_NAMES = [
    'Electrical_Fire_Cls_C',
    'Gas_Fire_cls_B',
    'Wood_fire_cls_A',
    'cooking_fire_Cls_K',
    'metal_fire_cls_D'
]

# Display names for better visualization
CLASS_DISPLAY_NAMES = [
    'Electrical Fire (Class C)',
    'Gas Fire (Class B)',
    'Wood Fire (Class A)',
    'Cooking Fire (Class K)',
    'Metal Fire (Class D)'
]


# ===================== CNN MODEL DEFINITION =====================
class FireCNN(nn.Module):
    """Updated model architecture matching training script"""

    def __init__(self, num_classes):
        super(FireCNN, self).__init__()

        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.2),

            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.2),

            # Block 3
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
    """
    Use YOLO to check: Is there fire in this frame? (YES/NO)
    Returns: (fire_detected: bool, confidence: float, bboxes: list)
    """
    results = yolo_model(frame, verbose=False)

    fire_bboxes = []
    max_confidence = 0.0

    for result in results:
        boxes = result.boxes
        for box in boxes:
            cls = int(box.cls[0].item())
            if cls == 0:  # Fire class
                conf = box.conf[0].item()
                if conf >= YOLO_CONFIDENCE_THRESHOLD:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    fire_bboxes.append((x1, y1, x2, y2))
                    max_confidence = max(max_confidence, conf)

    fire_detected = len(fire_bboxes) > 0
    return fire_detected, max_confidence, fire_bboxes


# ===================== FIRE CLASSIFICATION (CNN) =====================
def classify_fire_type(frame):
    """
    Use CNN to classify: What type of fire?
    Input: ENTIRE frame
    """
    try:
        # Convert BGR to RGB
        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # Transform and classify
        img_tensor = test_transform(pil_img).unsqueeze(0)

        with torch.no_grad():
            outputs = cnn_model(img_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)

        predicted_idx = predicted.item()

        return {
            'class': CLASS_NAMES[predicted_idx],  # Original class name
            'display_name': CLASS_DISPLAY_NAMES[predicted_idx],  # Pretty name
            'class_index': predicted_idx,
            'confidence': confidence.item(),
            'all_probabilities': {CLASS_DISPLAY_NAMES[i]: probabilities[0][i].item() * 100
                                  for i in range(len(CLASS_NAMES))},
            'success': True
        }
    except Exception as e:
        return {'success': False, 'error': str(e)}


# ===================== VIDEO PROCESSING =====================
def process_video(video_path, output_path=None):
    """
    Direct 2-stage pipeline (NO temporal smoothing):
    1. YOLO: Is there fire? (YES/NO)
    2. If YES → CNN: What type? (Direct classification)
    """

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Cannot open video {video_path}")
        return

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"\n{'=' * 70}")
    print(f"Video: {fps} FPS | {width}x{height} | {total_frames} frames")
    print(f"{'=' * 70}\n")

    # Video writer
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_number = 0
    frames_with_fire = 0
    classification_history = []

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

            # STAGE 2: CNN classifies - What type of fire? (DIRECT, NO SMOOTHING)
            classification = classify_fire_type(frame)

            if classification['success']:
                current_classification = classification

                # Store for final analysis
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
            # Add YOLO confidence on bbox
            cv2.putText(frame, f"Fire {yolo_confidence:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # Draw status banner
        if current_classification:
            # FIRE CLASSIFIED - Green banner
            fire_display_name = current_classification['display_name']
            fire_class_short = fire_display_name.split('(')[0].strip()

            # Top banner - classification
            banner_height = 120
            cv2.rectangle(frame, (0, 0), (width, banner_height), (0, 150, 0), -1)
            cv2.rectangle(frame, (0, 0), (width, banner_height), (0, 255, 0), 5)

            # Main text
            main_text = f"FIRE TYPE: {fire_class_short}"
            cv2.putText(frame, main_text, (20, 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 255, 255), 3)

            # Full classification
            cv2.putText(frame, fire_display_name, (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            # Confidence
            detail_text = f"Confidence: {current_classification['confidence'] * 100:.1f}%"
            cv2.putText(frame, detail_text, (20, 108),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        elif fire_detected:
            # FIRE DETECTED BUT CLASSIFICATION FAILED - Yellow banner
            banner_height = 70
            cv2.rectangle(frame, (0, 0), (width, banner_height), (0, 180, 180), -1)
            cv2.rectangle(frame, (0, 0), (width, banner_height), (0, 255, 255), 4)

            cv2.putText(frame, "FIRE DETECTED - Classification Error", (20, 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 3)
        else:
            # NO FIRE - Gray banner
            cv2.rectangle(frame, (0, 0), (width, 65), (80, 80, 80), -1)
            cv2.putText(frame, "No Fire Detected", (20, 43),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2)

        # Bottom info bar with enhanced details
        info_bg_y = height - 50
        cv2.rectangle(frame, (0, info_bg_y), (width, height), (30, 30, 30), -1)

        info_text = f"Frame: {frame_number}/{total_frames} | Fire frames: {frames_with_fire}"
        cv2.putText(frame, info_text, (10, height - 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Show current classification probabilities if available
        if current_classification:
            top3_probs = sorted(current_classification['all_probabilities'].items(),
                                key=lambda x: x[1], reverse=True)[:3]
            prob_text = " | ".join([f"{cls.split('(')[0].strip()}: {prob:.0f}%"
                                    for cls, prob in top3_probs])
            cv2.putText(frame, prob_text, (10, height - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # Write output
        if output_path:
            out.write(frame)

        # Display
        cv2.imshow('Fire Detection & Classification', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

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
    if output_path:
        out.release()
    cv2.destroyAllWindows()

    # ===================== FINAL SUMMARY =====================
    print("\n" + "=" * 70)
    print("PROCESSING COMPLETE")
    print("=" * 70)
    print(f"Total Frames: {frame_number}")
    print(f"Frames with Fire: {frames_with_fire} ({frames_with_fire / frame_number * 100:.1f}%)")
    print(f"Classified Frames: {len(classification_history)}")

    if classification_history:
        # Calculate most common classification
        class_counts = defaultdict(int)
        total_confidence = defaultdict(float)

        for entry in classification_history:
            class_counts[entry['class']] += 1
            total_confidence[entry['class']] += entry['confidence']

        # Most frequent class
        most_common_class = max(class_counts, key=class_counts.get)
        most_common_idx = CLASS_NAMES.index(most_common_class)
        most_common_display = CLASS_DISPLAY_NAMES[most_common_idx]

        print(f"\n🔥 OVERALL CLASSIFICATION:")
        print(f"   Most Common: {most_common_display}")
        print(
            f"   Occurrences: {class_counts[most_common_class]} frames ({class_counts[most_common_class] / len(classification_history) * 100:.1f}%)")
        print(f"   Avg Confidence: {total_confidence[most_common_class] / class_counts[most_common_class] * 100:.2f}%")

        # Show distribution of all classifications
        print(f"\n   Classification Distribution:")
        for cls in sorted(class_counts.keys(), key=lambda x: class_counts[x], reverse=True):
            count = class_counts[cls]
            percentage = count / len(classification_history) * 100
            avg_conf = total_confidence[cls] / count * 100
            cls_idx = CLASS_NAMES.index(cls)
            display_name = CLASS_DISPLAY_NAMES[cls_idx]
            print(f"      {display_name}: {count} frames ({percentage:.1f}%) - Avg Conf: {avg_conf:.1f}%")

        # Show final frame probabilities
        if current_classification:
            print(f"\n   Final Frame Probabilities:")
            sorted_probs = sorted(current_classification['all_probabilities'].items(),
                                  key=lambda x: x[1], reverse=True)
            for cls, prob in sorted_probs:
                print(f"      {cls}: {prob:.2f}%")
    else:
        print("\n✓ No fire detected in video")

    if output_path:
        print(f"\n✓ Output saved to: {output_path}")
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

    output_video = VIDEO_PATH.replace('.mp4', '_classified.mp4')
    process_video(VIDEO_PATH, output_path=output_video)

# import cv2
# import torch
# import torch.nn as nn
# from torchvision import transforms
# from PIL import Image
# from ultralytics import YOLO
# from collections import deque, defaultdict
# import numpy as np
#
# # ===================== CONFIGURATION =====================
# YOLO_MODEL_PATH = r"D:\CODEWORK_PROJECTS\FireDetection\best.pt"
# CNN_MODEL_PATH = r"D:\CODEWORK_PROJECTS\FireDetection\models\best_model.pth"
# VIDEO_PATH = r"C:\Users\rukes\Downloads\12543-239934681_tiny.mp4"
# YOLO_CONFIDENCE_THRESHOLD = 0.5  # YOLO: Is there fire? (YES/NO)
# TEMPORAL_BUFFER_SIZE = 10  # Last 10 frames for voting
# MIN_PERSISTENT_FRAMES = 5  # Need 5 frames to confirm classification
#
# # Class names matching your trained model (exact order from training)
# CLASS_NAMES = [
#     'Electrical_Fire_Cls_C',
#     'Gas_Fire_cls_B',
#     'Wood_fire_cls_A',
#     'cooking_fire_Cls_K',
#     'metal_fire_cls_D'
# ]
#
# # Display names for better visualization
# CLASS_DISPLAY_NAMES = [
#     'Electrical Fire (Class C)',
#     'Gas Fire (Class B)',
#     'Wood Fire (Class A)',
#     'Cooking Fire (Class K)',
#     'Metal Fire (Class D)'
# ]
#
#
# # ===================== CNN MODEL DEFINITION =====================
# class FireCNN(nn.Module):
#     """Updated model architecture matching training script"""
#
#     def __init__(self, num_classes):
#         super(FireCNN, self).__init__()
#
#         self.features = nn.Sequential(
#             # Block 1
#             nn.Conv2d(3, 32, kernel_size=3, padding=1),
#             nn.BatchNorm2d(32),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(2),
#             nn.Dropout2d(0.2),
#
#             # Block 2
#             nn.Conv2d(32, 64, kernel_size=3, padding=1),
#             nn.BatchNorm2d(64),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(2),
#             nn.Dropout2d(0.2),
#
#             # Block 3
#             nn.Conv2d(64, 128, kernel_size=3, padding=1),
#             nn.BatchNorm2d(128),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(2),
#             nn.Dropout2d(0.3),
#         )
#
#         self.classifier = nn.Sequential(
#             nn.Flatten(),
#             nn.Linear(128 * 8 * 8, 128),
#             nn.BatchNorm1d(128),
#             nn.ReLU(inplace=True),
#             nn.Dropout(0.4),
#             nn.Linear(128, num_classes)
#         )
#
#     def forward(self, x):
#         x = self.features(x)
#         x = self.classifier(x)
#         return x
#
#
# # ===================== LOAD MODELS =====================
# print("Loading models...")
# yolo_model = YOLO(YOLO_MODEL_PATH)
# print("✓ YOLO loaded (Fire Detection: YES/NO)")
#
# cnn_model = FireCNN(num_classes=len(CLASS_NAMES))
# cnn_model.load_state_dict(torch.load(CNN_MODEL_PATH, map_location='cpu'))
# cnn_model.eval()
# print("✓ CNN loaded (Fire Classification)")
# print(f"  Classes: {CLASS_DISPLAY_NAMES}")
#
# test_transform = transforms.Compose([
#     transforms.Resize((64, 64)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# ])
#
#
# # ===================== TEMPORAL FUSION (VOTING SYSTEM) =====================
# class TemporalVoting:
#     def __init__(self, buffer_size=10):
#         self.classifications = deque(maxlen=buffer_size)
#
#     def add_classification(self, fire_class, confidence):
#         """Add new frame classification"""
#         self.classifications.append({
#             'class': fire_class,
#             'confidence': confidence
#         })
#
#     def get_stable_prediction(self, min_frames=5):
#         """Get majority vote from recent frames"""
#         if len(self.classifications) < min_frames:
#             return None  # Not enough data
#
#         # Count weighted votes (confidence as weight)
#         class_votes = defaultdict(float)
#         for entry in self.classifications:
#             class_votes[entry['class']] += entry['confidence']
#
#         # Winner = highest total confidence
#         winner_class = max(class_votes, key=class_votes.get)
#
#         # Calculate average confidence for winner
#         winner_confidences = [e['confidence'] for e in self.classifications
#                               if e['class'] == winner_class]
#         avg_confidence = sum(winner_confidences) / len(winner_confidences)
#
#         # Agreement ratio (how many frames agree with winner)
#         agreement_count = len(winner_confidences)
#         agreement_ratio = agreement_count / len(self.classifications)
#
#         # Stable if at least 60% agreement
#         is_stable = agreement_ratio >= 0.6
#
#         return {
#             'class': winner_class,
#             'confidence': avg_confidence,
#             'agreement': agreement_ratio,
#             'stable': is_stable,
#             'total_frames': len(self.classifications)
#         }
#
#     def reset(self):
#         """Clear buffer"""
#         self.classifications.clear()
#
#
# # ===================== FIRE DETECTION (YOLO) =====================
# def is_fire_present(frame):
#     """
#     Use YOLO to check: Is there fire in this frame? (YES/NO)
#     Returns: (fire_detected: bool, confidence: float, bboxes: list)
#     """
#     results = yolo_model(frame, verbose=False)
#
#     fire_bboxes = []
#     max_confidence = 0.0
#
#     for result in results:
#         boxes = result.boxes
#         for box in boxes:
#             cls = int(box.cls[0].item())
#             if cls == 0:  # Fire class
#                 conf = box.conf[0].item()
#                 if conf >= YOLO_CONFIDENCE_THRESHOLD:
#                     x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
#                     fire_bboxes.append((x1, y1, x2, y2))
#                     max_confidence = max(max_confidence, conf)
#
#     fire_detected = len(fire_bboxes) > 0
#     return fire_detected, max_confidence, fire_bboxes
#
#
# # ===================== FIRE CLASSIFICATION (CNN) =====================
# def classify_fire_type(frame):
#     """
#     Use CNN to classify: What type of fire?
#     Input: ENTIRE frame
#     """
#     try:
#         # Convert BGR to RGB
#         pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
#
#         # Transform and classify
#         img_tensor = test_transform(pil_img).unsqueeze(0)
#
#         with torch.no_grad():
#             outputs = cnn_model(img_tensor)
#             probabilities = torch.nn.functional.softmax(outputs, dim=1)
#             confidence, predicted = torch.max(probabilities, 1)
#
#         predicted_idx = predicted.item()
#
#         return {
#             'class': CLASS_NAMES[predicted_idx],  # Original class name
#             'display_name': CLASS_DISPLAY_NAMES[predicted_idx],  # Pretty name
#             'class_index': predicted_idx,
#             'confidence': confidence.item(),
#             'all_probabilities': {CLASS_DISPLAY_NAMES[i]: probabilities[0][i].item() * 100
#                                   for i in range(len(CLASS_NAMES))},
#             'success': True
#         }
#     except Exception as e:
#         return {'success': False, 'error': str(e)}
#
#
# # ===================== VIDEO PROCESSING =====================
# def process_video(video_path, output_path=None):
#     """
#     Enhanced 3-stage pipeline:
#     1. YOLO: Is there fire? (YES/NO)
#     2. If YES → CNN: What type? (Electrical/Gas/Wood/Cooking/Metal)
#     3. Temporal fusion: Stable prediction over time
#     """
#
#     cap = cv2.VideoCapture(video_path)
#
#     if not cap.isOpened():
#         print(f"Error: Cannot open video {video_path}")
#         return
#
#     # Get video properties
#     fps = int(cap.get(cv2.CAP_PROP_FPS))
#     width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#
#     print(f"\n{'=' * 70}")
#     print(f"Video: {fps} FPS | {width}x{height} | {total_frames} frames")
#     print(f"{'=' * 70}\n")
#
#     # Video writer
#     if output_path:
#         fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#         out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
#
#     # Temporal voting system
#     voter = TemporalVoting(buffer_size=TEMPORAL_BUFFER_SIZE)
#
#     frame_number = 0
#     frames_with_fire = 0
#     classification_history = []
#
#     print("Processing video...")
#     print("Pipeline: YOLO (fire?) → CNN (type?) → Temporal Fusion\n")
#
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break
#
#         frame_number += 1
#
#         # STAGE 1: YOLO checks - Is there fire?
#         fire_detected, yolo_confidence, fire_bboxes = is_fire_present(frame)
#
#         current_classification = None
#
#         if fire_detected:
#             frames_with_fire += 1
#
#             # STAGE 2: CNN classifies - What type of fire?
#             classification = classify_fire_type(frame)
#
#             if classification['success']:
#                 current_classification = classification
#                 # Add to temporal voting buffer
#                 voter.add_classification(
#                     classification['class'],
#                     classification['confidence']
#                 )
#
#                 # Store for final analysis
#                 classification_history.append({
#                     'frame': frame_number,
#                     'class': classification['class'],
#                     'confidence': classification['confidence']
#                 })
#
#         # STAGE 3: Get stable prediction from temporal fusion
#         stable_pred = voter.get_stable_prediction(min_frames=MIN_PERSISTENT_FRAMES)
#
#         # ===================== VISUALIZATION =====================
#
#         # Draw fire bounding boxes
#         for bbox in fire_bboxes:
#             x1, y1, x2, y2 = bbox
#             cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
#             # Add YOLO confidence on bbox
#             cv2.putText(frame, f"Fire {yolo_confidence:.2f}", (x1, y1 - 10),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
#
#         # Draw status banner
#         if stable_pred and stable_pred['stable']:
#             # STABLE CLASSIFICATION - Green banner
#             # Extract display name from stable prediction
#             stable_class_idx = CLASS_NAMES.index(stable_pred['class'])
#             fire_display_name = CLASS_DISPLAY_NAMES[stable_class_idx]
#             fire_class_short = fire_display_name.split('(')[0].strip()
#
#             # Top banner - classification
#             banner_height = 120
#             cv2.rectangle(frame, (0, 0), (width, banner_height), (0, 150, 0), -1)
#             cv2.rectangle(frame, (0, 0), (width, banner_height), (0, 255, 0), 5)
#
#             # Main text
#             main_text = f"FIRE TYPE: {fire_class_short}"
#             cv2.putText(frame, main_text, (20, 45),
#                         cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 255, 255), 3)
#
#             # Full classification
#             cv2.putText(frame, fire_display_name, (20, 80),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
#
#             # Confidence and agreement
#             detail_text = f"Confidence: {stable_pred['confidence'] * 100:.1f}% | Agreement: {stable_pred['agreement'] * 100:.0f}%"
#             cv2.putText(frame, detail_text, (20, 108),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
#
#         elif fire_detected:
#             # FIRE DETECTED BUT NOT STABLE YET - Yellow banner
#             banner_height = 90
#             cv2.rectangle(frame, (0, 0), (width, banner_height), (0, 180, 180), -1)
#             cv2.rectangle(frame, (0, 0), (width, banner_height), (0, 255, 255), 4)
#
#             cv2.putText(frame, "FIRE DETECTED - Analyzing...", (20, 38),
#                         cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 0), 3)
#
#             buffer_status = f"Collecting data: {len(voter.classifications)}/{MIN_PERSISTENT_FRAMES} frames"
#             cv2.putText(frame, buffer_status, (20, 70),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
#         else:
#             # NO FIRE - Gray banner
#             cv2.rectangle(frame, (0, 0), (width, 65), (80, 80, 80), -1)
#             cv2.putText(frame, "No Fire Detected", (20, 43),
#                         cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2)
#
#         # Bottom info bar with enhanced details
#         info_bg_y = height - 50
#         cv2.rectangle(frame, (0, info_bg_y), (width, height), (30, 30, 30), -1)
#
#         info_text = f"Frame: {frame_number}/{total_frames} | Fire frames: {frames_with_fire}"
#         cv2.putText(frame, info_text, (10, height - 28),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
#
#         # Show current classification probabilities if available
#         if current_classification:
#             top3_probs = sorted(current_classification['all_probabilities'].items(),
#                                 key=lambda x: x[1], reverse=True)[:3]
#             prob_text = " | ".join([f"{cls.split('(')[0].strip()}: {prob:.0f}%"
#                                     for cls, prob in top3_probs])
#             cv2.putText(frame, prob_text, (10, height - 8),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
#
#         # Write output
#         if output_path:
#             out.write(frame)
#
#         # Display
#         cv2.imshow('Fire Detection & Classification', frame)
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
#
#         # Progress updates
#         if frame_number % 30 == 0:
#             progress = frame_number / total_frames * 100
#             print(f"Progress: {frame_number}/{total_frames} ({progress:.1f}%)", end='')
#             if stable_pred and stable_pred['stable']:
#                 print(f" | Current: {stable_pred['class']} ({stable_pred['confidence'] * 100:.1f}%)")
#             else:
#                 print()
#
#     cap.release()
#     if output_path:
#         out.release()
#     cv2.destroyAllWindows()
#
#     # ===================== FINAL SUMMARY =====================
#     print("\n" + "=" * 70)
#     print("PROCESSING COMPLETE")
#     print("=" * 70)
#     print(f"Total Frames: {frame_number}")
#     print(f"Frames with Fire: {frames_with_fire} ({frames_with_fire / frame_number * 100:.1f}%)")
#
#     final_pred = voter.get_stable_prediction(min_frames=1)
#     if final_pred:
#         # Get display name for final prediction
#         final_class_idx = CLASS_NAMES.index(final_pred['class'])
#         final_display_name = CLASS_DISPLAY_NAMES[final_class_idx]
#
#         print(f"\n🔥 FINAL FIRE CLASSIFICATION:")
#         print(f"   Fire Type: {final_display_name}")
#         print(f"   Model Class: {final_pred['class']}")
#         print(f"   Confidence: {final_pred['confidence'] * 100:.2f}%")
#         print(f"   Agreement: {final_pred['agreement'] * 100:.2f}%")
#         print(f"   Analyzed Frames: {final_pred['total_frames']}")
#
#         # Show distribution of classifications
#         if classification_history:
#             print(f"\n   Classification Distribution:")
#             class_counts = defaultdict(int)
#             for entry in classification_history:
#                 class_counts[entry['class']] += 1
#
#             for cls in sorted(class_counts.keys()):
#                 count = class_counts[cls]
#                 percentage = count / len(classification_history) * 100
#                 cls_idx = CLASS_NAMES.index(cls)
#                 display_name = CLASS_DISPLAY_NAMES[cls_idx]
#                 print(f"      {display_name}: {count} frames ({percentage:.1f}%)")
#
#         # Show final frame probabilities
#         if len(voter.classifications) > 0:
#             # Get last valid frame
#             last_frame_copy = frame.copy()
#             last_classification = classify_fire_type(last_frame_copy)
#             if last_classification['success']:
#                 print(f"\n   Final Frame Probabilities:")
#                 sorted_probs = sorted(last_classification['all_probabilities'].items(),
#                                       key=lambda x: x[1], reverse=True)
#                 for cls, prob in sorted_probs:
#                     print(f"      {cls}: {prob:.2f}%")
#     else:
#         print("\n✓ No fire detected in video")
#
#     if output_path:
#         print(f"\n✓ Output saved to: {output_path}")
#     print("=" * 70)
#
#
# # ===================== MAIN =====================
# if __name__ == "__main__":
#     print("\n" + "=" * 70)
#     print("FIRE DETECTION & CLASSIFICATION PIPELINE")
#     print("=" * 70)
#     print("Stage 1: YOLO - Detect fire presence (YES/NO)")
#     print("Stage 2: CNN - Classify fire type:")
#     for i, (class_name, display_name) in enumerate(zip(CLASS_NAMES, CLASS_DISPLAY_NAMES)):
#         print(f"         {i}. {display_name}")
#     print("Stage 3: Temporal fusion for stable predictions")
#     print("=" * 70)
#
#     output_video = VIDEO_PATH.replace('.mp4', '_classified.mp4')
#     process_video(VIDEO_PATH, output_path=output_video)
