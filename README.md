# Smart Attendance System

A face recognition-based attendance system built with Streamlit and OpenCV.

## Features
- Face recognition-based login
- Automatic attendance marking
- View attendance history
- Dark mode UI

## Local Development
1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the app:
```bash
streamlit run app.py
```

## Deployment Options

### 1. Deploy to Streamlit Cloud (Recommended)
1. Push your code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repository
4. Deploy!

### 2. Deploy to Heroku
1. Install Heroku CLI
2. Login to Heroku:
```bash
heroku login
```

3. Create a new Heroku app:
```bash
heroku create your-app-name
```

4. Push to Heroku:
```bash
git push heroku main
```

### 3. Deploy to Railway
1. Go to [railway.app](https://railway.app)
2. Connect your GitHub repository
3. Add Python environment
4. Set start command: `streamlit run app.py`

## Environment Variables
No environment variables needed for basic setup.

## Directory Structure
```
attendance-app/
├── app.py              # Main Streamlit application
├── database.py         # Database operations
├── face_recognition.py # Face recognition logic
├── init_timetable.py   # Timetable initialization
├── utils.py           # Utility functions
├── requirements.txt   # Python dependencies
├── Procfile          # Deployment configuration
└── runtime.txt       # Python version specification
```
