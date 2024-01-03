import os
import requests
from PIL import Image
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing import image as keras_image
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalMaxPooling2D
from numpy.linalg import norm
from sklearn.neighbors import NearestNeighbors
import pickle

# Directory to save images
image_directory = 'Dress_sample'
if not os.path.exists(image_directory):
    os.makedirs(image_directory)

# Initialize the ResNet50 model
model = ResNet50(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
model.trainable = False
model = Sequential([model, GlobalMaxPooling2D()])

def download_and_extract_features(url, model, image_directory):
    filename = url.split('/')[-1]
    if not filename.endswith(('.jpg', '.png', '.jpeg')):
        filename += '.jpg'
    file_path = os.path.join(image_directory, filename)

    if not os.path.exists(file_path):
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                with open(file_path, 'wb') as f:
                    f.write(response.content)
                print(f"Downloaded: {url}")
            else:
                print(f"Failed to download (status code {response.status_code}): {url}")
                return None
        except requests.RequestException as e:
            print(f"Error downloading {url}: {e}")
            return None

    try:
        img = Image.open(file_path)
        img = img.resize((224, 224))
        img_array = keras_image.img_to_array(img)
        expand_img = np.expand_dims(img_array, axis=0)
        preprocessed_img = preprocess_input(expand_img)
        result_to_resnet = model.predict(preprocessed_img)
        flatten_result = result_to_resnet.flatten()
        result_normalized = flatten_result / norm(flatten_result)
        return result_normalized
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def process_excel_and_extract_features(excel_file_path, model, image_directory):
    df = pd.read_excel(excel_file_path)
    image_urls = df.iloc[:, 0].tolist()
    image_features = []
    img_files = []

    for url in image_urls:
        features = download_and_extract_features(url, model, image_directory)
        if features is not None:
            image_features.append(features)
            img_files.append(url)

    pickle.dump(image_features, open("image_dress_feature.pkl", "wb"))
    pickle.dump(img_files, open("img_dress_files.pkl", "wb"))
    print("Image processing and feature extraction completed.")


def extract_img_features(image_path, model):
    try:
        img = Image.open(image_path)

        # Convert image to RGB if it's not
        if img.mode != 'RGB':
            img = img.convert('RGB')

        img = img.resize((224, 224))
        img_array = keras_image.img_to_array(img)
        expand_img = np.expand_dims(img_array, axis=0)
        preprocessed_img = preprocess_input(expand_img)
        result_to_resnet = model.predict(preprocessed_img)
        flatten_result = result_to_resnet.flatten()
        result_normalized = flatten_result / norm(flatten_result)

        # Check if the resulting feature vector contains NaN values
        if np.isnan(result_normalized).any():
            raise ValueError("NaN values detected in feature extraction for " + image_path)

        return result_normalized

    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

# Load precomputed features and image files lists
features_list = pickle.load(open("image_dress_feature.pkl", "rb"))
img_files_list = pickle.load(open("img_dress_files.pkl", "rb"))

def recommendd(features, features_list):
    neighbors = NearestNeighbors(n_neighbors=5, algorithm='ball_tree')

    # Convert features_list to a 2D array if it's not already
    features_array = np.array(features_list)
    neighbors.fit(features_array)

    # Ensure 'features' is also a 2D array
    features_reshaped = np.array(features).reshape(1, -1)
    distances, indices = neighbors.kneighbors(features_reshaped)

    return indices

# Path to your Excel file
excel_file_path = '/Users/josevinueza/Documents/linkstopictures.xlsx'

# Uncomment the line below to process the Excel file and extract features
#process_excel_and_extract_features(excel_file_path, model, image_directory)
