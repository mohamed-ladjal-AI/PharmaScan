import os
import subprocess
import shutil
import cv2
import numpy as np
from tkinter import Tk
from tkinter.filedialog import askopenfilename
import matplotlib.pyplot as plt

# Function to run YOLOv5 on an image and get the output image path
def run_yolov5(image_path, best_weights):
    results_dir = r'C:\Users\moham\Desktop\yolo_results'
    
    # Remove previous results if they exist
    if os.path.exists(results_dir):
        shutil.rmtree(results_dir)

    # Run YOLOv5 detection using subprocess
    command = [
        'python', r'C:\Users\moham\Downloads\yolov5\detect.py',
        '--weights', best_weights,
        '--img', '640',
        '--conf', '0.4',
        '--source', image_path,
        '--save-txt',
        '--save-conf',  # Ensure confidences are saved with the output
        '--save-crop',  # Save the cropped images with detections
        '--project', r'C:\Users\moham\Desktop',  # Specify the project path
        '--name', 'yolo_results',  # Ensure results are saved in yolo_results
        '--exist-ok'  # Allow overwriting of previous results
    ]
    
    subprocess.run(command)

    # The YOLOv5 results should be saved under 'runs/detect/yolo_results'
    yolo_result_folder = os.path.join(r'C:\Users\moham\Desktop\yolo_results')

    # Check if the image exists in the expected path (in runs/detect/yolo_results)
    yolo_image_path = os.path.join(yolo_result_folder, os.path.basename(image_path))

    # Check if the result image exists
    if not os.path.exists(yolo_image_path):
        print(f"Warning: YOLO result image not found at {yolo_image_path}")
        return None

    # Load the YOLO result image
    return cv2.imread(yolo_image_path)

# Main function to combine everything
def main():
    # Set up paths
    yolo_weights = r'C:\Users\moham\Downloads\moh\last2\weights\last.pt'

    # Use Tkinter to create a file dialog for image selection
    Tk().withdraw()
    image_path = askopenfilename(title="Select an Image", filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])

    if not image_path:
        print("No image selected. Exiting.")
        return

    # Run YOLOv5 (Model 2)
    result_yolo = run_yolov5(image_path, yolo_weights)

    # Check if YOLO result was loaded
    if result_yolo is None:
        print("Skipping YOLO visualization as no result was found.")
        return

    # Display the YOLO result
    plt.imshow(cv2.cvtColor(result_yolo, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title('YOLOv5 Result')
    plt.show()

# Run the main function
if __name__ == '__main__':
    main()
