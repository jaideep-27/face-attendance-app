import os
import shutil

def restore_images():
    source_dir = 'images'
    temp_dir = 'images_temp'
    
    # Create temp directory
    os.makedirs(temp_dir, exist_ok=True)
    
    # Move all images from subdirectories to temp
    for root, dirs, files in os.walk(source_dir):
        for file in files:
            if file.endswith(('.jpg', '.jpeg', '.png')):
                src_path = os.path.join(root, file)
                dst_path = os.path.join(temp_dir, file)
                shutil.copy2(src_path, dst_path)
    
    # Replace images directory
    shutil.rmtree(source_dir)
    os.rename(temp_dir, source_dir)
    print("Images restored to original structure")

if __name__ == "__main__":
    restore_images()
