import os
import urllib.request

# Create models directory if it doesn't exist
if not os.path.exists('models'):
    os.makedirs('models')

# Download cascade classifier
url = 'https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml'
classifier_path = os.path.join('models', 'haarcascade_frontalface_default.xml')

if not os.path.exists(classifier_path):
    print("Downloading cascade classifier...")
    urllib.request.urlretrieve(url, classifier_path)
    print("Download complete!")
