import streamlit as st
import pandas as pd
import sqlite3
from datetime import datetime
from face_recognition import train_model, capture_images, get_user_by_face
from utils import clear_inputs
from database import (init_db, add_user, add_subject, mark_attendance, 
                     get_available_subjects, get_attendance_history,
                     get_user_details)
from init_timetable import init_timetable_data
import matplotlib.pyplot as plt
import logging
import os
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create necessary directories
os.makedirs("images", exist_ok=True)
os.makedirs("model", exist_ok=True)
os.makedirs("data", exist_ok=True)

# Page config
st.set_page_config(
    page_title="Smart Attendance System",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    /* Main app styling */
    .stApp {
        background-color: #1E1E1E;
        color: #E0E0E0;
    }
    
    /* Button styling */
    .stButton>button {
        width: auto !important;
        min-width: 200px;
        border-radius: 10px;
        background-color: #2196F3;
        color: white;
        border: none;
        padding: 0.5rem 1.5rem;
        font-size: 1rem;
        font-weight: 500;
        transition: all 0.3s ease;
        margin: 0 auto;
        display: block;
    }
    .stButton>button:hover {
        background-color: #1976D2;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    .stButton>button:active {
        transform: translateY(0);
    }
    
    /* Success box styling */
    .success-box {
        padding: 1.5rem;
        border-radius: 10px;
        background-color: rgba(76, 175, 80, 0.1);
        border: 1px solid #4CAF50;
        color: #4CAF50;
        margin-bottom: 1.5rem;
        width: auto !important;
    }
    .success-box h3 {
        color: #4CAF50;
        margin-bottom: 0.5rem;
    }
    
    /* Navbar styling */
    [data-testid="stSidebarNav"] {
        background-color: #2D2D2D;
        padding: 1rem 0;
    }
    [data-testid="stSidebarNav"] button {
        cursor: pointer !important;
        color: #E0E0E0 !important;
        width: auto !important;
        padding: 0.75rem 1.5rem;
        text-align: left;
        background: none;
        border: none;
        font-size: 1rem;
        transition: all 0.3s ease;
    }
    [data-testid="stSidebarNav"] button:hover {
        background-color: #3D3D3D;
        padding-left: 2rem;
    }
    
    /* Form styling */
    .stTextInput>div>div {
        background-color: #2D2D2D;
        color: #E0E0E0;
        border-radius: 8px;
        width: auto !important;
    }
    .stTextInput>label {
        color: #E0E0E0;
    }
    
    /* DataFrame styling */
    .dataframe {
        width: auto !important;
        margin: 1rem 0;
        border-collapse: collapse;
    }
    .dataframe th {
        background-color: #2196F3;
        color: white;
        padding: 0.75rem;
        text-align: left;
        font-weight: 500;
    }
    .dataframe td {
        background-color: #2D2D2D;
        color: #E0E0E0;
        padding: 0.75rem;
        border-bottom: 1px solid #404040;
    }
    .dataframe tr:hover td {
        background-color: #3D3D3D;
    }
    
    /* Info/Warning/Error message styling */
    .stAlert {
        background-color: #2D2D2D;
        color: #E0E0E0;
        border-radius: 10px;
        padding: 1rem;
        width: auto !important;
    }
    .stAlert > div {
        border: none !important;
    }
    
    /* Header styling */
    h1, h2, h3 {
        color: #2196F3;
        margin-bottom: 1.5rem;
    }
    
    /* Container styling */
    .block-container {
        max-width: none !important;
        width: auto !important;
        padding: 2rem;
    }
    
    /* Element container */
    .element-container {
        width: auto !important;
    }
    
    /* Stacked containers */
    .stVerticalBlock {
        width: auto !important;
    }
    
    /* Selectbox */
    .stSelectbox {
        width: auto !important;
    }
</style>
""", unsafe_allow_html=True)

# Navbar styles
st.markdown("""
<style>
    /* Navbar styles */
    [data-testid="stSidebarNav"] {
        background-color: #333;
        padding-top: 1rem;
    }
    [data-testid="stSidebarNav"] button {
        cursor: pointer !important;
        color: white !important;
        width: auto !important;
        padding: 0.5rem 1rem;
        text-align: left;
        background: none;
        border: none;
    }
    [data-testid="stSidebarNav"] button:hover {
        background-color: #444;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state variables
if 'initialized' not in st.session_state:
    st.session_state.initialized = True
    st.session_state.user_id = None
    st.session_state.current_page = None
    st.session_state.face_scanned = False
    st.session_state.db_initialized = False

# Initialize session state
if 'current_page' not in st.session_state:
    st.session_state.current_page = "Login"

# Initialize database only once when app starts
if not st.session_state.db_initialized:
    try:
        # Check if database exists and has tables
        if not os.path.exists('data/attendance.db'):
            init_db()
            init_timetable_data()
        st.session_state.db_initialized = True
    except Exception as e:
        st.error(f"Database initialization error: {str(e)}")
        st.stop()

# Title with icon
st.markdown("<h1>🎯 Smart Attendance System</h1>", unsafe_allow_html=True)

# Navigation
if st.session_state.user_id:
    choice = st.selectbox("", ["Mark Attendance", "View Attendance", "Logout"])
else:
    choice = st.selectbox("", ["Login", "Register"])

if choice == "Register":
    st.markdown("<h2>📝 New User Registration</h2>", unsafe_allow_html=True)
    
    # Initialize registration state if not exists
    if 'registration_step' not in st.session_state:
        st.session_state.registration_step = 'form'
        st.session_state.temp_name = ''
        st.session_state.temp_roll = ''
    
    if st.session_state.registration_step == 'form':
        with st.form("registration_form"):
            name = st.text_input("Full Name")
            roll_number = st.text_input("Roll Number")
            submitted = st.form_submit_button("Register")
            
            if submitted and name and roll_number:
                if add_user(name, roll_number):
                    st.session_state.temp_name = name
                    st.session_state.temp_roll = roll_number
                    st.session_state.registration_step = 'capture'
                    st.rerun()
                else:
                    st.error("❌ Roll number already exists!")
    
    elif st.session_state.registration_step == 'capture':
        st.success("✅ Registration successful! Please proceed to capture your face.")
        if st.button("📸 Capture Face"):
            with st.spinner("Capturing..."):
                if capture_images(st.session_state.temp_name, st.session_state.temp_roll):
                    with st.spinner("Training model..."):
                        if train_model("images", "model"):
                            st.success("✨ Face captured successfully! You can now login.")
                            # Reset registration state
                            st.session_state.registration_step = 'form'
                            st.session_state.temp_name = ''
                            st.session_state.temp_roll = ''
                            # Redirect to login
                            st.session_state.current_page = "Login"
                            time.sleep(1)
                            st.rerun()
                        else:
                            st.error("❌ Error training model. Please try again.")
                else:
                    st.error("❌ Error capturing face. Please try again.")

elif choice == "Login":
    st.markdown("<h2>🔐 User Login</h2>", unsafe_allow_html=True)
    
    if st.session_state.user_id:
        st.session_state.current_page = "Mark Attendance"
        st.rerun()
    else:
        if 'temp_user_id' not in st.session_state:
            st.session_state.temp_user_id = None

        if st.session_state.temp_user_id is None:
            st.info("🎥 Click the button below to start face recognition")
            scan_button = st.button("🔍 Scan Your Face")
        else:
            st.info("✅ Face scanned successfully! Click proceed to continue")
            scan_button = st.button("🔄 Scan Again")

        if scan_button:
            with st.spinner("Scanning..."):
                user_id = get_user_by_face("model")
                if user_id:
                    st.session_state.temp_user_id = user_id
                    user_details = get_user_details(user_id)
                    if user_details:
                        name, roll_number = user_details
                        st.success(f"✨ Welcome back, {name}!")
                        st.rerun()
                else:
                    st.error("❌ Face not recognized. Please register if you're a new user.")
                    st.session_state.temp_user_id = None

        if st.session_state.temp_user_id:
            if st.button("➡️ Proceed to Mark Attendance"):
                st.session_state.user_id = st.session_state.temp_user_id
                st.session_state.temp_user_id = None
                st.session_state.current_page = "Mark Attendance"
                st.rerun()

elif choice == "Logout":
    # Immediately clear everything and redirect
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()

elif choice == "Mark Attendance":
    if 'user_id' not in st.session_state or not st.session_state.user_id:
        st.warning("Please login first!")
        st.stop()
    
    st.markdown("<h2>📝 Mark Attendance</h2>", unsafe_allow_html=True)
    
    # Get user details for display
    user_details = get_user_details(st.session_state.user_id)
    if user_details:
        name, roll_number = user_details
        st.markdown(f"""
            <div class='success-box'>
                <h3>👋 Welcome, {name}!</h3>
                <p>Roll Number: {roll_number}</p>
            </div>
        """, unsafe_allow_html=True)
    
    # Display current day and time
    current_day = datetime.now().strftime('%A')
    current_time = datetime.now().strftime('%I:%M %p')
    
    st.markdown(f"""
        <div style='background-color: #2D2D2D; padding: 1rem; border-radius: 8px; margin: 1rem 0;'>
            <h4 style='color: #2196F3; margin: 0;'>📅 {current_day}</h4>
            <p style='color: #B0BEC5; margin: 0.5rem 0 0 0;'>⏰ {current_time}</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Get available classes
    st.markdown("<h3>📚 Available Classes</h3>", unsafe_allow_html=True)
    current_time = datetime.now().time()
    day_of_week = datetime.now().weekday() + 1
    available_subjects = get_available_subjects(day_of_week)
    
    if not available_subjects:
        st.info("No classes scheduled at this time.")
    else:
        current_subjects = []
        for subject in available_subjects:
            subject_id, name, code, start_time, end_time = subject
            try:
                start = datetime.strptime(start_time, '%H:%M').time()
                end = datetime.strptime(end_time, '%H:%M').time()
                
                if start <= current_time <= end:
                    current_subjects.append(subject)
                    st.markdown(f"""
                        <div class='subject-card'>
                            <h4 style='color: #2196F3; margin: 0;'>{name} ({code})</h4>
                            <p style='color: #B0BEC5; margin: 0.5rem 0;'>⏰ {start_time} - {end_time}</p>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    if st.button(f"✓ Mark Present for {name}", key=f"mark_{subject_id}"):
                        try:
                            if mark_attendance(st.session_state.user_id, subject_id):
                                st.markdown("""
                                    <div class='success-box'>
                                        <p>✅ Attendance marked successfully!</p>
                                    </div>
                                """, unsafe_allow_html=True)
                            else:
                                st.warning("Attendance already marked for this class today")
                        except Exception as e:
                            st.error(f"Error marking attendance: {str(e)}")
            except ValueError as e:
                st.error(f"Error processing time: {str(e)}")
        
        if not current_subjects:
            st.info("No classes available at this time.")

elif choice == "View Attendance":
    if 'user_id' not in st.session_state or not st.session_state.user_id:
        st.warning("Please login first!")
        st.stop()
    
    st.markdown("<h2>📈 Attendance History</h2>", unsafe_allow_html=True)
    
    # Get user details for display
    user_details = get_user_details(st.session_state.user_id)
    if user_details:
        name, roll_number = user_details
        st.markdown(f"""
            <div class='success-box'>
                <h3>👋 Welcome, {name}!</h3>
                <p>Roll Number: {roll_number}</p>
            </div>
        """, unsafe_allow_html=True)
    
    # Get attendance history
    attendance_history = get_attendance_history(st.session_state.user_id)
    
    if not attendance_history:
        st.info("No attendance records found.")
    else:
        # Create a DataFrame for better display
        records = []
        for subject_id, subject_name, subject_code, timestamp in attendance_history:
            # Convert timestamp to datetime
            dt = datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S')
            records.append({
                'Subject': f"{subject_name} ({subject_code})",
                'Date': dt.strftime('%Y-%m-%d'),
                'Time': dt.strftime('%I:%M %p')
            })
        
        df = pd.DataFrame(records)
        
        # Display the DataFrame with custom styling
        st.markdown("""
            <style>
            .dataframe {
                width: auto !important;
                margin: 1rem 0;
                border-collapse: collapse;
            }
            .dataframe th {
                background-color: #2196F3;
                color: white;
                padding: 0.5rem;
                text-align: left;
            }
            .dataframe td {
                background-color: #2D2D2D;
                color: #B0BEC5;
                padding: 0.5rem;
                border-bottom: 1px solid #404040;
            }
            </style>
        """, unsafe_allow_html=True)
        
        st.dataframe(df, use_container_width=True)
