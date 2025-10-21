# 📸 Face Recognition Attendance System

A complete face recognition attendance system using traditional machine learning (OpenCV, scikit-learn, PCA) with an interactive Streamlit web interface.

## ✨ Features

- 🎥 **Real-time Recognition**: Webcam-based face detection and recognition
- 📊 **Attendance Tracking**: Automatic CSV logging with timestamps
- 📸 **Dataset Capture**: Add new people through the web interface
- 📈 **Analytics**: View attendance records with filtering and charts
- ⚙️ **Adjustable Threshold**: Control recognition sensitivity
- 🏥 **Health Check**: Verify system setup and file integrity

## 🚀 Quick Start

### Local Installation
```bash
git clone <your-repo-url>
cd attendance_system
pip install -r requirements.txt
streamlit run app_streamlit.py
```

### First Time Setup
1. **Train the model** using the Jupyter notebook:
   - Run Cells 4→5→6→7 to create model files
   - Run Cell 11 to create attendance.csv
2. **Add people** using Dataset Capture tab
3. **Start marking attendance** with Real-time Recognition

## 📁 Project Structure
```
attendance_system/
├── app_streamlit.py              # Main Streamlit application
├── face_recognition_attendance.ipynb  # Training notebook
├── requirements.txt              # Python dependencies
├── README.md                     # This file
├── attendance.csv                # Attendance records (auto-created)
├── dataset/                      # Person images
│   ├── person1/
│   ├── person2/
│   └── ...
└── models/                       # Trained models
    ├── haarcascade_frontalface_default.xml
    ├── best_model.joblib
    ├── pca.joblib
    └── labels.json
```

## 🌐 Streamlit Cloud Deployment

### Step 1: Push to GitHub
```bash
git init
git add .
git commit -m "Initial commit: Face Recognition Attendance System"
git branch -M main
git remote add origin https://github.com/yourusername/your-repo-name.git
git push -u origin main
```

### Step 2: Deploy on Streamlit Cloud
1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Click "New app"
3. Fill in:
   - **Repository**: `yourusername/your-repo-name`
   - **Branch**: `main`
   - **Main file path**: `attendance_system/app_streamlit.py`
4. Click "Deploy"

### Step 3: Share Your App
- Get your public URL: `https://your-app-name.streamlit.app`
- Share with others for remote attendance marking

## ⚠️ Important Notes

- **Webcam Access**: Streamlit Cloud cannot access local webcams. For cloud deployment, users need to run locally or use a camera-enabled device.
- **Model Files**: Ensure `models/` folder with trained files is included in your GitHub repo.
- **Dataset**: Include sample `dataset/` with a few people for demonstration.

## 🛠️ Technical Details

- **Face Detection**: OpenCV Haar Cascades
- **Feature Extraction**: PCA (100 components)
- **Classifiers**: SVM, Logistic Regression, Decision Tree, Random Forest
- **Preprocessing**: Grayscale, histogram equalization, 64x64 resize
- **Confidence Scoring**: Adjustable threshold (0.1-1.0)

## 📝 Usage Guide

1. **Health Check**: Verify all files are present
2. **Dataset Capture**: Add new people (5-10 images each)
3. **Real-time Recognition**: Mark attendance automatically
4. **Attendance Viewer**: View, filter, and export records

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📄 License

MIT License - feel free to use for educational and commercial purposes.

---

**Made with ❤️ using Streamlit, OpenCV, and scikit-learn**


