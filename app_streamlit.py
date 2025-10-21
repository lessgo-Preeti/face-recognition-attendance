import os
from pathlib import Path
import json
import time
from datetime import datetime

import streamlit as st
import pandas as pd
import cv2
import numpy as np
import joblib
from PIL import Image


def get_paths():
    # Use repository-relative paths for GitHub deployment
    project_dir = Path(__file__).resolve().parent
    dataset_dir = project_dir / "dataset"
    models_dir = project_dir / "models"
    attendance_csv = project_dir / "attendance.csv"
    haar_path = models_dir / "haarcascade_frontalface_default.xml"
    best_model_path = models_dir / "best_model.joblib"
    pca_path = models_dir / "pca.joblib"
    labels_path = models_dir / "labels.json"
    return {
        "PROJECT_DIR": project_dir,
        "DATASET_DIR": dataset_dir,
        "MODELS_DIR": models_dir,
        "ATTENDANCE_CSV": attendance_csv,
        "HAAR_PATH": haar_path,
        "BEST_MODEL_PATH": best_model_path,
        "PCA_PATH": pca_path,
        "LABELS_PATH": labels_path,
    }


def page_health_check():
    st.title("🏥 Health Check")
    st.write("Verify environment, folders, and model artifacts.")
    
    # Tips section
    with st.expander("💡 Tips & Troubleshooting"):
        st.markdown("""
        **If files show "Missing":**
        - Run the Jupyter notebook cells 4→5→6→7 to create model files
        - Run cell 11 to create attendance.csv
        - Ensure you're in the correct directory
        
        **Expected files:**
        - `attendance.csv` - Attendance records
        - `models/haarcascade_frontalface_default.xml` - Face detection
        - `models/best_model.joblib` - Trained classifier
        - `models/pca.joblib` - Feature reduction
        - `models/labels.json` - Person name mapping
        """)

    paths = get_paths()
    cols = st.columns(2)

    with cols[0]:
        st.subheader("Folders")
        for key in ["PROJECT_DIR", "DATASET_DIR", "MODELS_DIR"]:
            p = paths[key]
            st.write(f"{key}: {p}")
            st.success("Exists") if p.exists() else st.error("Missing")

    with cols[1]:
        st.subheader("Files")
        for key in ["ATTENDANCE_CSV", "HAAR_PATH", "BEST_MODEL_PATH", "PCA_PATH", "LABELS_PATH"]:
            p = paths[key]
            st.write(f"{key}: {p}")
            st.success("Exists") if p.exists() else st.warning("Missing")

    # Quick attendance preview
    st.subheader("Attendance Preview")
    att_path = paths["ATTENDANCE_CSV"]
    if att_path.exists():
        try:
            df = pd.read_csv(att_path)
            st.dataframe(df.tail(10))
        except Exception as e:
            st.error(f"Failed to read attendance.csv: {e}")
    else:
        st.info("attendance.csv not found.")


def page_attendance_viewer():
    st.title("📊 Attendance Viewer")
    st.write("View, filter, and export attendance records.")
    
    paths = get_paths()
    
    # Load attendance data
    if not paths["ATTENDANCE_CSV"].exists():
        st.error("attendance.csv not found. Please mark some attendance first.")
        return
    
    try:
        df = pd.read_csv(paths["ATTENDANCE_CSV"])
        if df.empty:
            st.info("No attendance records found.")
            return
    except Exception as e:
        st.error(f"Failed to load attendance data: {e}")
        return
    
    # Convert date column to datetime for filtering
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    
    # Sidebar filters
    st.sidebar.subheader("🔍 Filters")
    
    # Date range filter
    if not df['date'].isna().all():
        min_date = df['date'].min().date()
        max_date = df['date'].max().date()
        date_range = st.sidebar.date_input(
            "Select date range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date
        )
        
        if len(date_range) == 2:
            start_date, end_date = date_range
            df_filtered = df[(df['date'].dt.date >= start_date) & (df['date'].dt.date <= end_date)]
        else:
            df_filtered = df
    else:
        df_filtered = df
    
    # Person filter
    unique_people = sorted(df_filtered['name'].unique())
    selected_people = st.sidebar.multiselect(
        "Select people",
        options=unique_people,
        default=unique_people
    )
    
    if selected_people:
        df_filtered = df_filtered[df_filtered['name'].isin(selected_people)]
    
    # Search by name
    search_term = st.sidebar.text_input("🔍 Search by name", "")
    if search_term:
        df_filtered = df_filtered[df_filtered['name'].str.contains(search_term, case=False, na=False)]
    
    # Display statistics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Records", len(df_filtered))
    with col2:
        st.metric("Unique People", df_filtered['name'].nunique())
    with col3:
        if not df_filtered.empty and not df_filtered['date'].isna().all():
            latest_date = df_filtered['date'].max().strftime('%Y-%m-%d')
            st.metric("Latest Date", latest_date)
    
    # Display filtered data
    st.subheader("📋 Attendance Records")
    
    if df_filtered.empty:
        st.info("No records match your filters.")
    else:
        # Sort by date and time
        df_display = df_filtered.sort_values(['date', 'time'], ascending=[False, False])
        
        # Display table
        st.dataframe(
            df_display,
            use_container_width=True,
            hide_index=True,
            column_config={
                "name": "Person",
                "date": "Date",
                "time": "Time"
            }
        )
        
        # Download button
        csv_data = df_display.to_csv(index=False)
        st.download_button(
            label="📥 Download CSV",
            data=csv_data,
            file_name=f"attendance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    
    # Attendance summary chart
    if not df_filtered.empty:
        st.subheader("📈 Attendance Summary")
        
        # Daily attendance count
        daily_counts = df_filtered.groupby(df_filtered['date'].dt.date).size()
        if not daily_counts.empty:
            st.bar_chart(daily_counts)
        
        # Person-wise attendance count
        person_counts = df_filtered['name'].value_counts()
        if not person_counts.empty:
            st.subheader("👥 Attendance by Person")
            st.bar_chart(person_counts)


def page_dataset_capture():
    st.title("📸 Dataset Capture")
    st.write("Capture and save new person images for training.")
    
    paths = get_paths()
    
    # Person name input
    person_name = st.text_input("Enter person name (e.g., 'alice', 'john')", "")
    
    if not person_name:
        st.info("Please enter a person name to start capturing.")
        return
    
    # Create person directory
    person_dir = paths["DATASET_DIR"] / person_name
    person_dir.mkdir(parents=True, exist_ok=True)
    
    # Check existing images
    existing_images = list(person_dir.glob("*.png")) + list(person_dir.glob("*.jpg"))
    st.info(f"📁 Folder: {person_dir}")
    st.info(f"📷 Existing images: {len(existing_images)}")
    
    # Capture controls
    col1, col2 = st.columns(2)
    with col1:
        start_capture = st.button("📸 Start Camera", type="primary")
    with col2:
        stop_capture = st.button("⏹️ Stop Camera")
    
    if start_capture:
        st.session_state.capture_active = True
        st.success(f"Camera started for {person_name}. Press 'c' to capture, 'q' to quit.")
    
    if stop_capture:
        st.session_state.capture_active = False
        st.info("Camera stopped.")
    
    # Capture session state
    if 'captured_count' not in st.session_state:
        st.session_state.captured_count = 0
    
    # Capture loop
    if st.session_state.get('capture_active', False):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("Cannot access webcam. Please check camera permissions.")
            st.session_state.capture_active = False
        else:
            frame_placeholder = st.empty()
            status_placeholder = st.empty()
            
            # Add capture button outside the while loop
            col1, col2 = st.columns([1, 1])
            with col1:
                capture_btn = st.button("📷 Capture Face", key="capture_face_btn")
            with col2:
                st.write("")  # Empty space for alignment
            
            try:
                while st.session_state.get('capture_active', False):
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    # Face detection
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    face_cascade = cv2.CascadeClassifier(str(paths["HAAR_PATH"]))
                    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)
                    
                    # Draw bounding boxes
                    for (x, y, w, h) in faces:
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                        cv2.putText(frame, f"Click 'Capture Face' button below", (x, y-10), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)
                    
                    # Display frame
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
                    
                    # Handle capture button press
                    if capture_btn:
                        if len(faces) > 0:
                            # Take first detected face
                            (x, y, w, h) = faces[0]
                            face_img = gray[y:y+h, x:x+w]
                            face_img = cv2.equalizeHist(face_img)
                            face_img = cv2.resize(face_img, (64, 64), interpolation=cv2.INTER_AREA)
                            
                            # Save image
                            img_count = len(existing_images) + st.session_state.captured_count + 1
                            out_path = person_dir / f"img_{img_count:03d}.png"
                            cv2.imwrite(str(out_path), face_img)
                            st.session_state.captured_count += 1
                            status_placeholder.success(f"✅ Captured image {st.session_state.captured_count}: {out_path.name}")
                        else:
                            status_placeholder.warning("⚠️ No face detected. Try again.")
                    
                    time.sleep(0.1)
                    
            finally:
                cap.release()
                st.session_state.capture_active = False
    
    # Reset capture count when starting new session
    if start_capture:
        st.session_state.captured_count = 0


def page_real_time_recognition():
    st.title("🎥 Real-time Face Recognition & Attendance")
    st.write("Start webcam recognition to mark attendance automatically.")
    
    # Tips section
    with st.expander("💡 Tips for Best Results"):
        st.markdown("""
        **For accurate recognition:**
        - Ensure good lighting on your face
        - Look directly at the camera
        - Keep face centered in the frame
        - Adjust confidence threshold if needed (lower = more lenient)
        
        **Color coding:**
        - 🟢 Green box = High confidence (will mark attendance)
        - 🟠 Orange box = Low confidence (won't mark attendance)
        
        **Auto-stop:** Camera closes automatically after successful attendance marking.
        """)
    
    paths = get_paths()
    
    # Check if model files exist
    if not all([paths["BEST_MODEL_PATH"].exists(), paths["PCA_PATH"].exists(), paths["LABELS_PATH"].exists()]):
        st.error("Model files missing. Please train the model first using the notebook.")
        return
    
    # Load model artifacts
    try:
        model = joblib.load(paths["BEST_MODEL_PATH"])
        pca = joblib.load(paths["PCA_PATH"])
        with open(paths["LABELS_PATH"], 'r') as f:
            id_to_name = json.load(f)
        id_to_name = {int(k): v for k, v in id_to_name.items()}
        st.success("Model loaded successfully!")
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return
    
    # Confidence threshold slider
    st.subheader("⚙️ Recognition Settings")
    confidence_threshold = st.slider(
        "Confidence Threshold", 
        min_value=0.1, 
        max_value=1.0, 
        value=0.5, 
        step=0.05,
        help="Higher values = more strict recognition (fewer false positives, but might miss some faces)"
    )
    st.info(f"Current threshold: {confidence_threshold:.2f} - Only faces with confidence ≥ {confidence_threshold:.2f} will be marked for attendance")
    
    # Face detection function
    def detect_faces_gray(image_bgr, scaleFactor=1.3, minNeighbors=5):
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(str(paths["HAAR_PATH"]))
        faces = face_cascade.detectMultiScale(gray, scaleFactor=scaleFactor, minNeighbors=minNeighbors)
        return [(int(x), int(y), int(w), int(h)) for (x, y, w, h) in faces]
    
    # Face preprocessing function
    def preprocess_face_crop(image_bgr, bbox):
        x, y, w, h = bbox
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        face = gray[y:y+h, x:x+w]
        face = cv2.equalizeHist(face)
        face = cv2.resize(face, (64, 64), interpolation=cv2.INTER_AREA)
        return face
    
    # Session state for attendance tracking
    if 'marked_attendance' not in st.session_state:
        st.session_state.marked_attendance = set()
    
    # Start/Stop buttons
    col1, col2 = st.columns(2)
    with col1:
        start_recognition = st.button("🎥 Start Recognition", type="primary")
    with col2:
        stop_recognition = st.button("⏹️ Stop Recognition")
    
    if start_recognition:
        # Start a fresh session: allow each person to mark once per new run
        st.session_state.marked_attendance = set()
        st.session_state.recognition_active = True
        st.success("Recognition started! Look at the camera.")
    
    if stop_recognition:
        st.session_state.recognition_active = False
        st.info("Recognition stopped.")
    
    # Recognition loop
    if st.session_state.get('recognition_active', False):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("Cannot access webcam. Please check camera permissions.")
            st.session_state.recognition_active = False
        else:
            frame_placeholder = st.empty()
            status_placeholder = st.empty()
            
            try:
                stop_now = False
                while st.session_state.get('recognition_active', False):
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    bboxes = detect_faces_gray(frame)
                    for (x, y, w, h) in bboxes:
                        try:
                            face_img = preprocess_face_crop(frame, (x, y, w, h))
                            x_input = (face_img.flatten().astype(np.float32) / 255.0)[None, :]
                            z = pca.transform(x_input)
                            
                            if hasattr(model, 'predict_proba'):
                                prob = model.predict_proba(z)[0]
                                conf = float(prob.max())
                                pred_id = int(prob.argmax())
                            else:
                                dec = model.decision_function(z)
                                if dec.ndim == 1:
                                    dec = np.vstack([-dec, dec]).T
                                e = np.exp(dec - dec.max(axis=1, keepdims=True))
                                prob = (e / e.sum(axis=1, keepdims=True))[0]
                                conf = float(prob.max())
                                pred_id = int(prob.argmax())
                            
                            name = id_to_name.get(pred_id, "unknown")
                            
                            # Color code bounding box based on confidence
                            if conf >= confidence_threshold:
                                color = (0, 255, 0)  # Green for high confidence
                                thickness = 3
                            else:
                                color = (0, 165, 255)  # Orange for low confidence
                                thickness = 2
                            
                            # Draw bounding box and label
                            cv2.rectangle(frame, (x, y), (x+w, y+h), color, thickness)
                            label = f"{name} ({conf:.2f})"
                            cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                            
                            # Mark attendance if confidence meets threshold
                            if conf >= confidence_threshold and name != "unknown" and name not in st.session_state.marked_attendance:
                                now = datetime.now()
                                row = {"name": name, "date": now.strftime("%Y-%m-%d"), "time": now.strftime("%H:%M:%S")}
                                try:
                                    df = pd.read_csv(paths["ATTENDANCE_CSV"])
                                    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
                                    df.to_csv(paths["ATTENDANCE_CSV"], index=False)
                                    st.session_state.marked_attendance.add(name)
                                    status_placeholder.success(f"✅ Attendance marked successfully: {name} at {now.strftime('%H:%M:%S')} (conf: {conf:.2f}, threshold: {confidence_threshold:.2f})")
                                    # Auto-stop recognition after successful attendance
                                    st.session_state.recognition_active = False
                                    stop_now = True
                                    st.success("🎉 Camera will close automatically after successful attendance marking!")
                                    break
                                except Exception as e:
                                    status_placeholder.error(f"❌ Failed to mark attendance: {e}")
                            
                        except Exception as e:
                            pass  # Skip failed face processing
                    
                    # If we flagged to stop, break outer loop now
                    if stop_now:
                        break

                    # Display frame
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
                    
                    time.sleep(0.1)  # Small delay to prevent overwhelming
                    
            finally:
                cap.release()
                st.session_state.recognition_active = False


def main():
    # Page config
    st.set_page_config(
        page_title="Face Recognition Attendance System",
        page_icon="📸",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for better styling
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .feature-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 0.25rem;
        padding: 0.75rem;
        margin: 0.5rem 0;
    }
    .info-box {
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        border-radius: 0.25rem;
        padding: 0.75rem;
        margin: 0.5rem 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Main header
    st.markdown('<h1 class="main-header">📸 Face Recognition Attendance System</h1>', unsafe_allow_html=True)
    
    # Sidebar navigation
    st.sidebar.title("🧭 Navigation")
    st.sidebar.markdown("---")
    
    page = st.sidebar.selectbox(
        "Go to",
        [
            "🏥 Health Check",
            "🎥 Real-time Recognition", 
            "📊 Attendance Viewer",
            "📸 Dataset Capture",
        ],
    )
    
    # Sidebar info
    st.sidebar.markdown("---")
    st.sidebar.markdown("### ℹ️ Quick Info")
    st.sidebar.markdown("""
    **How to use:**
    1. **Health Check** - Verify all files are ready
    2. **Dataset Capture** - Add new people to the system
    3. **Real-time Recognition** - Mark attendance automatically
    4. **Attendance Viewer** - View and export records
    """)
    
    # Route to appropriate page
    if "Health Check" in page:
        page_health_check()
    elif "Real-time Recognition" in page:
        page_real_time_recognition()
    elif "Attendance Viewer" in page:
        page_attendance_viewer()
    elif "Dataset Capture" in page:
        st.info("Dataset Capture temporarily disabled due to technical issues. Use the Jupyter notebook Cell 15 to capture images.")


if __name__ == "__main__":
    main()


