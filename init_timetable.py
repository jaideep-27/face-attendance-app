from database import add_subject, add_timetable_entry
import os

def init_timetable_data():
    # Create necessary directories
    os.makedirs("images", exist_ok=True)
    os.makedirs("model", exist_ok=True)
    os.makedirs("data", exist_ok=True)
    
    # Add subjects
    subjects = [
        ("Data Structures", "CS201"),
        ("Database Management", "CS202"),
        ("Operating Systems", "CS203"),
        ("Computer Networks", "CS204"),
        ("Software Engineering", "CS205"),
        ("Machine Learning", "CS206"),
        ("Web Development", "CS207"),
        ("Mobile Computing", "CS208")
    ]
    
    subject_ids = {}
    for name, code in subjects:
        subject_id = add_subject(name, code)
        if subject_id:
            subject_ids[code] = subject_id
    
    # Only proceed with timetable if we have subjects
    if not subject_ids:
        print("No subjects were added. Skipping timetable initialization.")
        return
    
    # Time slots from 9 AM to 10 PM
    time_slots = [
        ("09:00", "10:00"),
        ("10:00", "11:00"),
        ("11:00", "12:00"),
        ("12:00", "13:00"),
        ("13:00", "14:00"),
        ("14:00", "15:00"),
        ("15:00", "16:00"),
        ("16:00", "17:00"),
        ("17:00", "18:00"),
        ("18:00", "19:00"),
        ("19:00", "20:00"),
        ("20:00", "21:00"),
        ("21:00", "22:00")
    ]
    
    # Add timetable entries for each day (1-5 represents Monday to Friday)
    days = range(1, 6)  # Monday to Friday
    subject_list = list(subject_ids.keys())
    
    for day in days:
        for i, (start_time, end_time) in enumerate(time_slots):
            # Rotate through subjects for each slot
            subject_code = subject_list[i % len(subject_list)]
            subject_id = subject_ids[subject_code]
            add_timetable_entry(subject_id, day, start_time, end_time)

if __name__ == "__main__":
    init_timetable_data()
