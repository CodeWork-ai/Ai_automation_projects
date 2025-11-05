import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from ultralytics import YOLO
from PIL import Image, ImageDraw
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet
from io import BytesIO
import tempfile
import os
import cv2
import mimetypes

# ---------------------- CONFIG ----------------------
YOLO_MODEL_PATH = r"D:\CODEWORK_PROJECTS\FireDetection\models\best.pt"
CNN_MODEL_PATH = r"D:\CODEWORK_PROJECTS\FireDetection\models\best_model.pth"
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

# Fire extinguisher recommendations based on fire class
EXTINGUISHER_RECOMMENDATIONS = {
    'Electrical_Fire_Cls_C': {
        'recommended': 'CO2 (Carbon Dioxide) Extinguisher',
        'alternatives': ['Dry Chemical (ABC) Extinguisher'],
        'never_use': 'Water-based extinguishers (risk of electrocution)',
        'description': 'Class C fires involve energized electrical equipment. CO2 is non-conductive and leaves no residue.'
    },
    'Gas_Fire_cls_B': {
        'recommended': 'Dry Chemical (ABC or BC) Extinguisher',
        'alternatives': ['CO2 Extinguisher', 'Foam Extinguisher'],
        'never_use': 'Water (can spread flammable liquids)',
        'description': 'Class B fires involve flammable liquids and gases. Dry chemical agents smother the fire effectively.'
    },
    'Wood_fire_cls_A': {
        'recommended': 'Water Extinguisher',
        'alternatives': ['Dry Chemical (ABC) Extinguisher', 'Foam Extinguisher'],
        'never_use': 'CO2 in enclosed spaces (oxygen displacement)',
        'description': 'Class A fires involve ordinary combustible materials like wood, paper, and fabric. Water cools and penetrates.'
    },
    'cooking_fire_Cls_K': {
        'recommended': 'Wet Chemical (Class K) Extinguisher',
        'alternatives': ['Fire Blanket for small fires'],
        'never_use': 'Water or dry chemical (can cause splatter and spread fire)',
        'description': 'Class K fires involve cooking oils and fats. Wet chemical creates a soapy foam that cools and seals the surface.'
    },
    'metal_fire_cls_D': {
        'recommended': 'Dry Powder (Class D) Extinguisher',
        'alternatives': ['Sand or other dry inert materials'],
        'never_use': 'Water, CO2, or standard dry chemical (can react violently with metals)',
        'description': 'Class D fires involve combustible metals like magnesium, titanium, or lithium. Special dry powder agents are required.'
    }
}

# ---------------------- EXACT CNN MODEL FROM YOUR TRAINING ----------------------
class FireCNN(nn.Module):
    """Your exact custom CNN architecture matching training script"""

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
            nn.Linear(128 * 8 * 8, 128),  # 64x64 -> 8x8 after 3 MaxPool2d layers
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# ---------------------- LOAD TRAINED CNN (FIXED FOR YOUR ARCHITECTURE) ----------------------
def load_trained_cnn(num_classes=5):
    """
    Load your exact custom CNN model architecture.
    This matches your training script exactly.
    """
    try:
        model = FireCNN(num_classes=num_classes)
        
        # Load the state dict
        state_dict = torch.load(CNN_MODEL_PATH, map_location='cpu')
        
        # Handle different checkpoint formats and module prefix
        if isinstance(state_dict, dict) and 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        
        # Remove 'module.' prefix if present (DataParallel training)
        if any(key.startswith('module.') for key in state_dict.keys()):
            new_state_dict = {}
            for key, value in state_dict.items():
                new_key = key.replace('module.', '') if key.startswith('module.') else key
                new_state_dict[new_key] = value
            state_dict = new_state_dict
        
        # Load the corrected state dict
        model.load_state_dict(state_dict)
        model.eval()
        
        st.success("✅ Custom FireCNN model loaded successfully!")
        return model
        
    except Exception as e:
        st.error(f"❌ Error loading CNN model: {str(e)}")
        # Try loading with strict=False as fallback
        try:
            model = FireCNN(num_classes=num_classes)
            state_dict = torch.load(CNN_MODEL_PATH, map_location='cpu')
            model.load_state_dict(state_dict, strict=False)
            model.eval()
            st.warning("⚠️ Model loaded with some missing keys (using strict=False)")
            return model
        except Exception as fallback_error:
            st.error(f"❌ Complete failure to load model: {str(fallback_error)}")
            return None

# ---------------------- FIRE DETECTION ----------------------
def detect_fire(file_path):
    """
    Detect fire regions in images or videos using YOLO.
    """
    try:
        model = YOLO(YOLO_MODEL_PATH)
        file_type, _ = mimetypes.guess_type(file_path)

        # ---- IMAGE HANDLING ----
        if file_type and file_type.startswith("image"):
            img = Image.open(file_path).convert("RGB")
            results = model(file_path)[0]
            draw = ImageDraw.Draw(img)

            fire_crop = None
            detection_count = 0
            
            for box in results.boxes.xyxy:
                x1, y1, x2, y2 = map(int, box[:4])
                draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
                fire_crop = img.crop((x1, y1, x2, y2))
                detection_count += 1
            
            return {
                "annotated_image": img, 
                "fire_region": fire_crop, 
                "detections": detection_count,
                "full_image": img  # For CNN classification
            }

        # ---- VIDEO HANDLING ----
        elif file_type and file_type.startswith("video"):
            cap = cv2.VideoCapture(file_path)
            fire_crop = None
            last_frame = None
            detection_count = 0

            while True:
                success, frame = cap.read()
                if not success:
                    break
                    
                results = model(frame)[0]
                for box in results.boxes.xyxy:
                    x1, y1, x2, y2 = map(int, box[:4])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    fire_crop = frame[y1:y2, x1:x2]
                    detection_count += 1
                last_frame = frame

            cap.release()

            if last_frame is not None:
                last_frame_rgb = cv2.cvtColor(last_frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(last_frame_rgb)
                if fire_crop is not None:
                    fire_crop = Image.fromarray(cv2.cvtColor(fire_crop, cv2.COLOR_BGR2RGB))
                return {
                    "annotated_image": img, 
                    "fire_region": fire_crop, 
                    "detections": detection_count,
                    "full_image": img  # For CNN classification
                }
            else:
                return {
                    "annotated_image": None, 
                    "fire_region": None, 
                    "detections": 0,
                    "full_image": None
                }

        else:
            raise ValueError("Unsupported file type.")
            
    except Exception as e:
        st.error(f"❌ Error in fire detection: {str(e)}")
        return {
            "annotated_image": None, 
            "fire_region": None, 
            "detections": 0,
            "full_image": None
        }

# ---------------------- FIRE CLASSIFICATION (UPDATED FOR YOUR MODEL) ----------------------
def classify_fire(detection_data):
    """
    Classify the type of fire using your exact custom CNN model.
    Uses the FULL IMAGE (like your training script) with 64x64 input size.
    """
    if detection_data["full_image"] is None or detection_data["detections"] == 0:
        return "No Fire Detected", 0.0, {}

    try:
        model = load_trained_cnn(num_classes=len(CLASS_NAMES))
        if model is None:
            return "Model Load Error", 0.0, {}

        # **EXACT SAME PREPROCESSING AS YOUR TRAINING SCRIPT**
        test_transform = transforms.Compose([
            transforms.Resize((64, 64)),  # Your model uses 64x64, not 224x224!
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Use full image for classification (same as your video script)
        img_tensor = test_transform(detection_data["full_image"]).unsqueeze(0)

        with torch.no_grad():
            outputs = model(img_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)

        predicted_idx = predicted.item()
        
        # Create probability dictionary
        all_probabilities = {
            CLASS_DISPLAY_NAMES[i]: probabilities[0][i].item() * 100
            for i in range(len(CLASS_NAMES))
        }

        return (
            CLASS_NAMES[predicted_idx], 
            confidence.item() * 100,
            all_probabilities
        )
        
    except Exception as e:
        st.error(f"❌ Error in fire classification: {str(e)}")
        return "Classification Error", 0.0, {}

# ---------------------- EXTINGUISHER RECOMMENDATION ----------------------
def get_extinguisher_recommendation(fire_class):
    """Get extinguisher recommendations based on fire class."""
    if fire_class in EXTINGUISHER_RECOMMENDATIONS:
        return EXTINGUISHER_RECOMMENDATIONS[fire_class]
    else:
        return {
            'recommended': 'Consult Fire Safety Professional',
            'alternatives': ['Call Emergency Services'],
            'never_use': 'Unknown fire type - avoid guessing',
            'description': 'Fire class not recognized. Contact emergency services immediately.'
        }

# ---------------------- PDF REPORT ----------------------
def generate_report(output_path, detection_data, fire_type, confidence, probabilities):
    """Generate a comprehensive PDF report with detection results and recommendations."""
    try:
        doc = SimpleDocTemplate(output_path, pagesize=A4)
        story = []
        styles = getSampleStyleSheet()

        # Title
        story.append(Paragraph("<b>🔥 Fire Detection & Classification Report</b>", styles['Title']))
        story.append(Spacer(1, 20))
        
        # Detection Summary
        story.append(Paragraph("<b>Detection Summary:</b>", styles['Heading2']))
        story.append(Paragraph(f"<b>Fire Detections:</b> {detection_data.get('detections', 0)}", styles['Normal']))
        story.append(Paragraph(f"<b>Fire Class:</b> {fire_type}", styles['Normal']))
        story.append(Paragraph(f"<b>Classification Confidence:</b> {confidence:.2f}%", styles['Normal']))
        story.append(Spacer(1, 20))

        # Classification Probabilities
        if probabilities:
            story.append(Paragraph("<b>🎯 Classification Probabilities:</b>", styles['Heading2']))
            sorted_probs = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
            for cls, prob in sorted_probs:
                story.append(Paragraph(f"• {cls}: {prob:.2f}%", styles['Normal']))
            story.append(Spacer(1, 20))

        # Extinguisher Recommendations
        if fire_type not in ["No Fire Detected", "Classification Error", "Model Load Error"]:
            recommendations = get_extinguisher_recommendation(fire_type)
            story.append(Paragraph("<b>🧯 Extinguisher Recommendations:</b>", styles['Heading2']))
            story.append(Paragraph(f"<b>Primary Recommendation:</b> {recommendations['recommended']}", styles['Normal']))
            story.append(Paragraph(f"<b>Alternative Options:</b> {', '.join(recommendations['alternatives'])}", styles['Normal']))
            story.append(Paragraph(f"<b>⚠️ Never Use:</b> {recommendations['never_use']}", styles['Normal']))
            story.append(Paragraph(f"<b>Description:</b> {recommendations['description']}", styles['Normal']))
            story.append(Spacer(1, 20))

        # Add annotated image if available
        if detection_data["annotated_image"]:
            img_buffer = BytesIO()
            detection_data["annotated_image"].save(img_buffer, format="PNG")
            img_buffer.seek(0)
            story.append(Paragraph("<b>Detection Results:</b>", styles['Heading2']))
            story.append(RLImage(img_buffer, width=400, height=300))
            story.append(Spacer(1, 20))

        # Safety Disclaimer
        story.append(Paragraph("<b>⚠️ Safety Disclaimer:</b>", styles['Heading2']))
        story.append(Paragraph("This report is generated by AI and should be used as guidance only. Always consult fire safety professionals and follow local fire safety regulations. In case of active fire, evacuate immediately and call emergency services.", styles['Italic']))
        
        story.append(Spacer(1, 20))
        story.append(Paragraph("Generated by Custom FireCNN Detection System", styles['Italic']))

        doc.build(story)
        return True
        
    except Exception as e:
        st.error(f"❌ Error generating PDF report: {str(e)}")
        return False

# ---------------------- STREAMLIT UI ----------------------
def main():
    st.set_page_config(
        page_title="🔥 Fire Detection & Classification", 
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Header
    st.title("🔥 Fire Detection & Classification System")
    st.markdown("**Custom FireCNN + YOLOv8 Pipeline**")
    st.divider()

    # Sidebar
    st.sidebar.header("🎛️ Controls")
    st.sidebar.markdown("Upload an image or video to detect and classify fire types.")
    
    option = st.sidebar.radio(
        "Select input type:", 
        ["📸 Image", "🎥 Video"], 
        horizontal=False
    )
    
    # File uploader
    file_types = ["jpg", "jpeg", "png"] if "Image" in option else ["mp4", "avi", "mov"]
    uploaded_file = st.sidebar.file_uploader(
        f"Upload your {option.lower().split()[1]} file", 
        type=file_types
    )
    
    # Model Status
    st.sidebar.divider()
    st.sidebar.subheader("🤖 Model Status")
    
    # Check YOLO model
    if os.path.exists(YOLO_MODEL_PATH):
        st.sidebar.success("✅ YOLO Model: Ready")
    else:
        st.sidebar.error("❌ YOLO Model: Not Found")
    
    # Check CNN model
    if os.path.exists(CNN_MODEL_PATH):
        st.sidebar.success("✅ Custom FireCNN: Ready")
    else:
        st.sidebar.error("❌ Custom FireCNN: Not Found")

    # Model Architecture Info
    st.sidebar.divider()
    st.sidebar.subheader("🏗️ Architecture Info")
    st.sidebar.info("""
    **Custom FireCNN:**
    • Input: 64×64×3 images
    • 3 Conv blocks (32→64→128)
    • Classifier: 8192→128→5
    • Output: 5 fire classes
    """)

    # Main content
    if uploaded_file:
        # Save uploaded file
        temp_dir = tempfile.mkdtemp()
        file_path = os.path.join(temp_dir, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.read())

        st.success(f"✅ {option.split()[1]} uploaded successfully: **{uploaded_file.name}**")

        # Create columns for layout
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # --- Step 1: Fire Detection ---
            st.subheader("🎯 Step 1: YOLOv8 Fire Detection")
            with st.spinner("Detecting fire regions..."):
                detections = detect_fire(file_path)
            
            if detections["annotated_image"]:
                st.image(
                    detections["annotated_image"], 
                    caption=f"🔥 Fire Detection Results ({detections['detections']} detections)",
                    use_column_width=True
                )
            else:
                st.warning("⚠️ No fire detected in the uploaded file.")

            # --- Step 2: Fire Classification ---
            st.subheader("🧠 Step 2: Custom FireCNN Classification")
            with st.spinner("Classifying fire type..."):
                fire_type, confidence, probabilities = classify_fire(detections)
            
            # Display results in metrics
            metric_col1, metric_col2 = st.columns(2)
            with metric_col1:
                # Map to display name for UI
                display_name = fire_type
                if fire_type in CLASS_NAMES:
                    idx = CLASS_NAMES.index(fire_type)
                    display_name = CLASS_DISPLAY_NAMES[idx]
                st.metric("🔥 Predicted Fire Type", display_name)
            with metric_col2:
                st.metric("📊 Confidence Score", f"{confidence:.1f}%")

            # Show all probabilities
            if probabilities:
                st.subheader("📊 Classification Probabilities")
                sorted_probs = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
                for cls, prob in sorted_probs:
                    st.progress(prob/100, text=f"{cls}: {prob:.1f}%")

        with col2:
            # --- Step 3: Extinguisher Recommendations ---
            st.subheader("🧯 Step 3: Extinguisher Guide")
            
            if fire_type not in ["No Fire Detected", "Classification Error", "Model Load Error"]:
                recommendations = get_extinguisher_recommendation(fire_type)
                
                # Primary recommendation
                st.success(f"**Recommended:** {recommendations['recommended']}")
                
                # Alternatives
                st.info(f"**Alternatives:** {', '.join(recommendations['alternatives'])}")
                
                # Warning
                st.error(f"**⚠️ Never Use:** {recommendations['never_use']}")
                
                # Description
                st.markdown(f"**ℹ️ Details:** {recommendations['description']}")
                
            else:
                st.info("🤖 No fire classification available for extinguisher recommendation.")

        # --- Step 4: Report Generation ---
        st.divider()
        st.subheader("📄 Step 4: Generate Detailed Report")
        
        report_col1, report_col2 = st.columns([1, 1])
        
        with report_col1:
            if st.button("📋 Generate PDF Report", type="primary", use_container_width=True):
                with st.spinner("Generating comprehensive report..."):
                    pdf_path = os.path.join(temp_dir, "Fire_Detection_Report.pdf")
                    success = generate_report(pdf_path, detections, fire_type, confidence, probabilities)
                    
                    if success:
                        with open(pdf_path, "rb") as pdf:
                            st.download_button(
                                "⬇️ Download Fire Safety Report", 
                                data=pdf, 
                                file_name=f"Fire_Report_{uploaded_file.name}.pdf", 
                                mime="application/pdf",
                                use_container_width=True
                            )
                        st.success("✅ Report generated successfully!")
                    else:
                        st.error("❌ Failed to generate report.")
        
        with report_col2:
            st.markdown("**📋 Report Contents:**")
            st.markdown("• Fire detection results")
            st.markdown("• Classification probabilities")
            st.markdown("• Extinguisher recommendations")
            st.markdown("• Safety guidelines")
            st.markdown("• Annotated images")

    else:
        # Welcome screen
        st.markdown("""
        ### 🚨 Welcome to the Custom FireCNN Detection System
        
        This AI system combines:
        
        1. **🎯 YOLOv8 Fire Detection** - Locates fire regions in images/videos
        2. **🧠 Custom FireCNN Classification** - Identifies fire types using your trained model
        3. **🧯 Expert Recommendations** - Suggests appropriate firefighting agents
        4. **📄 Detailed Reports** - Generates comprehensive safety documentation
        
        **🔥 Fire Classes Detected:**
        - **Class A**: Wood, paper, fabric (ordinary combustibles)
        - **Class B**: Flammable liquids and gases  
        - **Class C**: Electrical equipment fires
        - **Class D**: Combustible metals
        - **Class K**: Cooking oils and fats
        
        **⚙️ Model Architecture:**
        - **Input Size**: 64×64 RGB images
        - **Architecture**: Custom 3-block CNN (32→64→128 channels)
        - **Classifier**: Linear layers (8192→128→5 outputs)
        - **Training**: Your custom FireCNN model
        
        **🚀 Get Started:** Upload an image or video using the sidebar controls.
        """)
        
        st.info("💡 **Tip:** This system uses your exact training architecture with 64×64 input size.")

if __name__ == "__main__":
    main()
