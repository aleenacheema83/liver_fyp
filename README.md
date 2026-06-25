# Liver Tumor Segmentation Tool

A beautiful, professional Flask-based web application for medical AI image segmentation.

## Features

- 🎨 Clean, medical-themed interface
- 📤 Drag & drop image upload
- 🔬 Real-time image preview
- 📥 Download segmented results
- 📱 Fully responsive design
- ❓ Interactive help section

## Setup Instructions

### 1. Create Virtual Environment

**Windows:**
```powershell

venv\Scripts\activate
```

**Linux/Mac:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Application

```bash
python app.py
```

The application will be available at: `http://localhost:5000`

## Project Structure

```
liver-fyp/
├── app.py                 # Flask application
├── requirements.txt       # Python dependencies
├── templates/
│   └── index.html        # Main HTML template
├── static/
│   ├── css/
│   │   └── style.css    # Stylesheet
│   └── js/
│       └── main.js      # JavaScript functionality
├── uploads/              # Uploaded images (auto-created)
└── results/              # Processed results (auto-created)
```

## Color Scheme

- **Primary:** Navy Blue (#1a365d)
- **Secondary:** Teal (#2d7a7a)
- **Accent:** Soft Green (#48bb78)
- **Background:** White / Light Grey

## Notes

- Currently, the application shows a placeholder result. Integrate your AI model in the `/process` route in `app.py`.
- Uploaded files are stored temporarily in the `uploads/` folder.
- Maximum file size: 16MB
- Supported formats: PNG, JPG, JPEG

## License

For educational and research use only.

