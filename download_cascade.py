import os
import urllib.request

cascade_url = 'https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml'
cascade_path = os.path.join('model', 'haarcascade_frontalface_default.xml')

print(f"Downloading cascade classifier to {cascade_path}...")
urllib.request.urlretrieve(cascade_url, cascade_path)
print("Download complete!")
