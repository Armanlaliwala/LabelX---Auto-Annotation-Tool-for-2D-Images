import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import tempfile
import os
import zipfile
import io
from PIL import Image

# Custom class mapping for autonomous vehicle data
CUSTOM_CLASSES = {
    0: "Sedan",
    1: "SUV",
    2: "Animal",
    3: "Pedestrian",
    4: "Construction Vehicle",
    5: "Static On-road",
    6: "Road Cone"
}

# COCO classes fallback mapping to custom classes
COCO_TO_CUSTOM = {
    2: 0,   # car -> Sedan
    5: 1,   # bus -> SUV
    7: 1,   # truck -> SUV
    0: 3,   # person -> Pedestrian
    16: 2,  # bird -> Animal
    17: 2,  # cat -> Animal
    18: 2,  # dog -> Animal
    19: 2,  # horse -> Animal
}

class LabelXAnnotator:
    def __init__(self):
        self.model = None
        self.custom_model = False

    def load_model(self, model_path=None):
        """Load YOLOv8 model - custom or pretrained"""
        try:
            if model_path and os.path.exists(model_path):
                self.model = YOLO(model_path)
                self.custom_model = True
                st.success("✅ Custom model loaded successfully!")
            else:
                self.model = YOLO('yolov8n.pt')  # Pretrained COCO model
                self.custom_model = False
                st.warning("⚠ Using pretrained COCO model as fallback")
            return True
        except Exception as e:
            st.error(f"❌ Error loading model: {str(e)}")
            return False

    def predict_image(self, image, conf_threshold=0.5):
        """Run inference on single image"""
        if self.model is None:
            return None, []

        results = self.model(image, conf=conf_threshold)
        return results[0], self.extract_annotations(results[0])

    def extract_annotations(self, result):
        """Extract YOLO format annotations from results"""
        annotations = []

        if result.boxes is not None:
            boxes = result.boxes.xywhn.cpu().numpy()  # Normalized xywh
            classes = result.boxes.cls.cpu().numpy()

            for i, (box, cls) in enumerate(zip(boxes, classes)):
                # Map class if using COCO model
                if not self.custom_model:
                    cls_id = COCO_TO_CUSTOM.get(int(cls), None)
                    if cls_id is None:
                        continue  # Skip unmapped classes
                else:
                    cls_id = int(cls)

                # Ensure class is within our custom range
                if cls_id not in CUSTOM_CLASSES:
                    continue

                x_center, y_center, width, height = box
                annotations.append({
                    'class_id': cls_id,
                    'class_name': CUSTOM_CLASSES[cls_id],
                    'x_center': float(x_center),
                    'y_center': float(y_center),
                    'width': float(width),
                    'height': float(height)
                })

        return annotations

    def draw_annotations(self, image, annotations):
        """Draw bounding boxes and labels on image"""
        img_copy = image.copy()
        h, w = img_copy.shape[:2]

        colors = [
            (255, 0, 0),     # Sedan - Red
            (0, 255, 0),     # SUV - Green
            (0, 0, 255),     # Animal - Blue
            (255, 255, 0),   # Pedestrian - Yellow
            (255, 0, 255),   # Construction Vehicle - Magenta
            (0, 255, 255),   # Static On-road Object - Cyan
            (128, 0, 128)    # Road Cone - Purple
        ]

        for ann in annotations:
            x_center = int(ann['x_center'] * w)
            y_center = int(ann['y_center'] * h)
            width = int(ann['width'] * w)
            height = int(ann['height'] * h)

            x1 = x_center - width // 2
            y1 = y_center - height // 2
            x2 = x_center + width // 2
            y2 = y_center + height // 2

            color = colors[ann['class_id'] % len(colors)]

            # Draw bounding box
            cv2.rectangle(img_copy, (x1, y1), (x2, y2), color, 2)

            # Draw class name only (no confidence)
            label = ann['class_name']
            font_scale = 0.6
            font_thickness = 2
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness
            )

            cv2.rectangle(img_copy,
                         (x1, y1 - text_height - 10),
                         (x1 + text_width, y1),
                         color, -1)

            cv2.putText(img_copy, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness)

        return img_copy

    def create_yolo_annotation(self, annotations):
        """Create YOLO format annotation text"""
        lines = []
        for ann in annotations:
            line = f"{ann['class_id']} {ann['x_center']:.6f} {ann['y_center']:.6f} {ann['width']:.6f} {ann['height']:.6f}"
            lines.append(line)
        return '\n'.join(lines)


def main():
    st.set_page_config(page_title="LabelX - Auto-Annotation Tool", page_icon="🚗", layout="wide")
    st.title("🚗 LabelX - Auto-Annotation Tool")
    st.markdown("*AI-driven auto-annotation for autonomous vehicle data*")
    st.markdown("---")

    if 'annotator' not in st.session_state:
        st.session_state.annotator = LabelXAnnotator()

    st.sidebar.header("⚙ Model Configuration")

    model_option = st.sidebar.radio(
        "Select Model Type:",
        ["Use Pretrained COCO Model (Fallback)", "Upload Custom Model Weights"]
    )

    custom_model_path = None
    if model_option == "Upload Custom Model Weights":
        uploaded_weights = st.sidebar.file_uploader(
            "Upload YOLOv8 weights file (.pt)",
            type=['pt']
        )
        if uploaded_weights:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pt') as tmp_file:
                tmp_file.write(uploaded_weights.read())
                custom_model_path = tmp_file.name

    if st.sidebar.button("🔄 Load Model"):
        st.session_state.annotator.load_model(custom_model_path)

    if st.session_state.annotator.model is not None:
        model_type = "Custom" if st.session_state.annotator.custom_model else "COCO Fallback"
        st.sidebar.success(f"✅ Model Status: {model_type}")

    st.sidebar.markdown("### 🎯 Target Classes")
    for i, class_name in CUSTOM_CLASSES.items():
        st.sidebar.write(f"{i}: {class_name}")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.header("📤 Upload Data")
        uploaded_files = st.file_uploader(
            "Choose images",
            type=['jpg', 'jpeg', 'png'],
            accept_multiple_files=True
        )

        # Removed confidence UI
        conf_threshold = 0.5  # Default
        show_confidence = False  # Disabled

    with col2:
        st.header("📊 Processing Status")

        if uploaded_files and st.session_state.annotator.model is not None:
            annotations_data = []
            annotated_images = []

            progress_bar = st.progress(0)
            status_text = st.empty()

            for idx, uploaded_file in enumerate(uploaded_files):
                status_text.text(f"Processing {uploaded_file.name}...")

                image = Image.open(uploaded_file).convert('RGB')
                image_np = np.array(image)

                result, annotations = st.session_state.annotator.predict_image(image_np, conf_threshold)

                annotated_img = st.session_state.annotator.draw_annotations(image_np, annotations)

                annotations_data.append({
                    'filename': uploaded_file.name,
                    'annotations': annotations,
                    'yolo_format': st.session_state.annotator.create_yolo_annotation(annotations)
                })
                annotated_images.append((uploaded_file.name, annotated_img))

                progress_bar.progress((idx + 1) / len(uploaded_files))

            status_text.text("✅ Processing complete!")

            st.subheader("🖼 Annotated Images")
            for filename, annotated_img in annotated_images:
                st.write(f"{filename}")
                st.image(annotated_img, use_column_width=True)

            st.subheader("📥 Download Annotations")
            if st.button("📦 Generate Download Package"):
                zip_buffer = io.BytesIO()
                with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                    for data in annotations_data:
                        filename_base = os.path.splitext(data['filename'])[0]
                        zip_file.writestr(f"{filename_base}.txt", data['yolo_format'])

                zip_buffer.seek(0)
                st.download_button(
                    label="⬇ Download Annotations (YOLO format)",
                    data=zip_buffer.getvalue(),
                    file_name="labelx_annotations.zip",
                    mime="application/zip"
                )

        elif uploaded_files and st.session_state.annotator.model is None:
            st.warning("⚠ Please load a model first!")

        elif st.session_state.annotator.model is not None:
            st.info("📤 Upload images to start auto-annotation")

    st.markdown("---")
    st.markdown("*LabelX* - Built for autonomous vehicle perception systems | Powered by YOLOv8")


if __name__ == "__main__":
    main()
