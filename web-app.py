import os
import subprocess
import shutil
import cv2
import streamlit as st
from PIL import Image
from streamlit_image_zoom import image_zoom
import random
import pandas as pd
import glob
import numpy as np

st.set_page_config(page_title="PharmaScan", page_icon="💊", layout="wide")

def load_class_names(class_names_file):
    class_names = {}
    with open(class_names_file, 'r') as f:
        for line in f:
            parts = line.strip().split(': ')
            if len(parts) != 2:
                continue  # Skip lines that don't match the expected format
            class_id, class_name = parts
            class_names[int(class_id)] = class_name.strip("'")
    return class_names

# Dynamic color map generator for consistent colors per class ID
def generate_color_map(num_classes):
    return {i: tuple(random.randint(0, 255) for _ in range(3)) for i in range(num_classes)}

# Load classes and generate color map
COLOR_MAP = generate_color_map(1300)
CLASS_NAMES = load_class_names('class_names.txt')  # Load class names from the file
num_classes = len(CLASS_NAMES)  # Count the classes

# Sidebar CSS for custom theme
st.markdown(
    """
    <style>
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #f0f2f6 !important;
        padding: 20px !important;
        border-radius: 15px;
    }
    .stSidebar > div:first-child {
        padding-top: 0px;
    }
    /* Custom sidebar widget colors */
    .css-1v0mbdj p {
        font-weight: bold;
        color: #333;
    }
    .css-1v0mbdj h1, .css-1v0mbdj h2, .css-1v0mbdj h3 {
        color: #333;
    }
    /* Rounded edges for sidebar elements */
    .stFileUploader, .stTextInput, .stButton, .stSelectbox {
        border-radius: 10px;
    }
    /* Custom color for dataframe table header */
    .stDataFrame thead tr th {
        background-color: #f7f7f7;
        font-weight: bold;
        color: #333;
    }
    </style>
    """, 
    unsafe_allow_html=True
)

# Function to resize an image while maintaining its aspect ratio
def resize_image(image, target_width, target_height):
    height, width = image.shape[:2]
    aspect_ratio = width / height
    
    if aspect_ratio > 1:  # Wider than tall
        new_width = target_width
        new_height = int(target_width / aspect_ratio)
    else:  # Taller than wide
        new_height = target_height
        new_width = int(target_height * aspect_ratio)
    
    return cv2.resize(image, (new_width, new_height))

# Function to run YOLOv5 on an image and get the output image path and detected objects
def run_yolov5(image_path, best_weights):
    results_dir = r'yolo_results'
    
    # Remove previous results if they exist
    if os.path.exists(results_dir):
        shutil.rmtree(results_dir)

    # Run YOLOv5 detection using subprocess
    command = [
        'python', r'yolov5\detect.py',#update the path to your yolov5 
        '--weights', best_weights,
        '--img', '640',
        '--conf', '0.4',
        '--source', image_path,
        '--save-txt',
        '--save-conf',
        '--save-crop',
        '--project', r'',
        '--name', 'yolo_results',
        '--exist-ok'
    ]
    
    subprocess.run(command)

    # Check if the YOLO result image exists
    yolo_result_folder = os.path.join(r'yolo_results')#update the path here also
    yolo_image_path = os.path.join(yolo_result_folder, os.path.basename(image_path))
    if not os.path.exists(yolo_image_path):
        st.warning(f"Attention : L'image du résultat de YOLO n'a pas été trouvée à {yolo_image_path}")
        return None, []

    # Read the detection data to retrieve class names and colors
    detection_txt = os.path.join(yolo_result_folder, "labels", f"{os.path.splitext(os.path.basename(image_path))[0]}.txt")
    objects_detected = []
    if os.path.exists(detection_txt):
        with open(detection_txt, "r") as file:
            for line in file:
                class_id, x, y, w, h, conf = line.strip().split()
                class_id = int(class_id)
                class_name = CLASS_NAMES.get(class_id, f"Classe inconnue {class_id}")
                objects_detected.append((class_name, float(conf)))

    # Load the YOLO result image with updated labels
    return cv2.imread(yolo_image_path), objects_detected

# Streamlit app interface

# Title and introduction text
st.title("Bienvenue sur **PharmaScan** 💊")
st.markdown("""**PharmaScan** est une application intelligente de détection des noms de médicaments manuscrits dans les ordonnances médicales. Utilisant un modèle YOLOv5 pour identifier et extraire les noms des médicaments, elle offre une interface facile à utiliser et conviviale pour les professionnels de santé.""")

st.markdown("---")

# Add the scan and gallery buttons below the introduction text
col1, col2 = st.columns(2)

with col1:
    scan_button = st.button("📷 Scanner l'image", key="scan_button")

with col2:
    gallery_button = st.button("🖼️ Choisir dans la galerie", key="gallery_button")

# Button styling to match the logo colors
st.markdown("""
    <style>
    div.stButton > button:first-child {
        background-color: #268BD2; /* Color similar to the logo */
        color: white;
        border-radius: 10px;
        padding: 10px 20px;
        font-size: 16px;
    }
    div.stButton > button:hover {
        background-color: #2AA198; /* Lighter shade for hover */
        color: white;
    }
    </style>
""", unsafe_allow_html=True)
company_name = '''
    <p style="font-size: 30px; text-align: center;">
        <span style="color: #0052AD; font-weight: bold;">Pharma</span>
        <span style="color: #49BBC6; font-weight: bold;">Scan</span>
    </p>
'''
st.sidebar.markdown(company_name, unsafe_allow_html=True)  # Add colored company name
st.sidebar.image("logo.png", use_column_width=True)  # Display logo in sidebar

model_path = r'last2\weights\last.pt'#st.sidebar.text_input("Chemin des pondérations du modèle", r'C:\Users\moham\Downloads\moh\last2\weights\last.pt')

# Sidebar - Image upload
uploaded_image = st.sidebar.file_uploader("Téléchargez une image de l'ordonnance", type=["jpg", "jpeg", "png"])
run_detection = st.sidebar.button("Exécuter la détection")

# Sidebar - Sample images selection
sample_images = glob.glob("dada/*.jpg")
sample_images = [os.path.basename(img) for img in sample_images]

# Sidebar dropdown for sample image selection
st.sidebar.subheader("Ou choisissez un exemple d'image")
selected_sample = st.sidebar.selectbox("Sélectionnez une image d'exemple", [""] + sample_images)

# Placeholder for the uploaded and inference result images
col1, col2 = st.columns(2)

with col1:
    with st.container():
        st.markdown('<div class="image-frame">', unsafe_allow_html=True)
        image_display = st.empty()
        st.markdown('</div>', unsafe_allow_html=True)

with col2:
    with st.container():
        st.markdown('<div class="image-frame">', unsafe_allow_html=True)
        result_display = st.empty()
        st.markdown('</div>', unsafe_allow_html=True)

# Display uploaded or selected sample image immediately
if uploaded_image:
    temp_image_path = "temp_uploaded_image.jpg"
    with open(temp_image_path, "wb") as f:
        f.write(uploaded_image.getbuffer())
    pil_image = Image.open(temp_image_path)
    image_display.image(pil_image, caption="Image téléchargée", use_column_width=True)

elif selected_sample:
    temp_image_path = os.path.join("dada", selected_sample)
    pil_image = Image.open(temp_image_path)
    image_display.image(pil_image, caption="Image d'exemple", use_column_width=True)

# Run detection if the button is pressed
if run_detection and (uploaded_image or selected_sample):
    result_yolo, objects_detected = run_yolov5(temp_image_path, model_path)

    if result_yolo is None:
        st.warning("Il y a eu un problème avec la détection YOLO.")
    else:
        # Resize the result image to match the size of the uploaded image
        result_yolo_resized = result_yolo
        with col2:
            result_display.image(result_yolo_resized, caption="Résultat de la détection", use_column_width=True)

        st.subheader("Médicaments Détectés")
        df = pd.DataFrame(objects_detected, columns=["Médicament", "Confiance"])
        st.dataframe(df)
