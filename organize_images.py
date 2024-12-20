import os
import shutil

def organize_images():
    source_dir = 'images'
    if not os.path.exists(source_dir):
        print("Images directory not found")
        return
    
    # Create a temporary directory
    temp_dir = 'images_temp'
    os.makedirs(temp_dir, exist_ok=True)
    
    # Move files to user ID folders
    for filename in os.listdir(source_dir):
        if not filename.endswith(('.jpg', '.jpeg', '.png')):
            continue
            
        # Extract user ID from filename (format: name_id_number.jpg)
        try:
            user_id = filename.split('_')[1]
            user_dir = os.path.join(temp_dir, user_id)
            os.makedirs(user_dir, exist_ok=True)
            
            # Move the file
            shutil.copy2(
                os.path.join(source_dir, filename),
                os.path.join(user_dir, filename)
            )
        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")
    
    # Replace original directory with organized one
    shutil.rmtree(source_dir)
    os.rename(temp_dir, source_dir)
    print("Images organized successfully")

if __name__ == "__main__":
    organize_images()
