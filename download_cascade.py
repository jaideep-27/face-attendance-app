import os
import urllib.request

def download_cascade():
    cascade_dir = 'model'
    cascade_file = 'haarcascade_frontalface_default.xml'
    cascade_path = os.path.join(cascade_dir, cascade_file)
    
    # Create model directory if it doesn't exist
    os.makedirs(cascade_dir, exist_ok=True)
    
    # Download cascade file if it doesn't exist
    if not os.path.exists(cascade_path):
        url = f'https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/{cascade_file}'
        print(f'Downloading {cascade_file}...')
        urllib.request.urlretrieve(url, cascade_path)
        print('Download complete!')

if __name__ == '__main__':
    download_cascade()
