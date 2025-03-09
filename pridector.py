import random
import matplotlib.pyplot as plt
import cv2
import os
import subprocess
import shutil
from tkinter import Tk
from tkinter.filedialog import askopenfilename

# Function to load ground truth or prediction boxes from .txt files
def load_boxes(label_path):
    boxes = []
    with open(label_path, 'r') as f:
        for line in f:
            values = line.strip().split()
            class_id = int(values[0])
            x_center, y_center, width, height = map(float, values[1:5])
            boxes.append([class_id, x_center, y_center, width, height])
    return boxes

# Function to draw bounding boxes on an image with class names and larger font
def draw_boxes(image, boxes, class_names, color):
    h, w, _ = image.shape
    for box in boxes:
        class_id, x_center, y_center, width, height = box
        # Convert YOLO format to (x1, y1, x2, y2)
        x1 = int((x_center - width / 2) * w)
        y1 = int((y_center - height / 2) * h)
        x2 = int((x_center + width / 2) * w)
        y2 = int((y_center + height / 2) * h)
        # Draw the rectangle
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 3)
        # Put class name text with larger font
        class_name = class_names[class_id]
        font_scale = 1.5
        cv2.putText(image, class_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 3)
    return image

# Function to load class names from the classes.txt file
def load_class_names(class_file_path):
    with open(class_file_path, 'r') as f:
        class_names = [line.strip() for line in f.readlines()]
    return class_names

# Set paths
image_folder = r'C:\Users\moham\Downloads\moh\dada'
label_folder = r'C:\Users\moham\Downloads\moh\Labels'
class_file_path = r'C:\Users\moham\Downloads\moh\Labels\classes.txt'
best_weights = r'C:\Users\moham\Downloads\moh\last2\weights\last.pt'

# Load class names
class_names = load_class_names(class_file_path)

# Use Tkinter to create a file dialog for image selection
Tk().withdraw()  # Hide the main tkinter window
image_path = askopenfilename(title="Select an Image", filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])

# Check if an image was selected
if not image_path:
    print("No image selected. Exiting.")
    exit()

# Remove the previous results directory
results_dir = r'C:\Users\moham\Desktop\yolo_results'
if os.path.exists(results_dir):
    shutil.rmtree(results_dir)

# Run the YOLOv5 detection using subprocess
command = [
    'python', r'C:\Users\moham\Downloads\yolov5\detect.py',
    '--weights', best_weights,
    '--img', '640',
    '--conf', '0.4',
    '--source', image_path,
    '--save-txt',
    '--save-conf',
    '--nosave',
    '--project', r'C:\Users\moham\Desktop',
    '--name', 'yolo_results'
]

subprocess.run(command)

# Load predicted boxes from YOLOv5 output
pred_label_folder = os.path.join(results_dir, 'labels')
pred_label_file = os.path.join(pred_label_folder, os.path.basename(image_path).replace('.jpg', '.txt').replace('.jpeg', '.txt').replace('.png', '.txt'))
predicted_boxes = load_boxes(pred_label_file) if os.path.exists(pred_label_file) else []

# Load the original image
image = cv2.imread(image_path)

# Draw predicted boxes (random colors for each box)
for box in predicted_boxes:
    color = tuple(random.randint(0, 255) for _ in range(3))
    image = draw_boxes(image, [box], class_names, color)

# Save the image with annotations
output_path = os.path.join(results_dir, os.path.basename(image_path).replace('.jpg', '_annotated.jpg').replace('.jpeg', '_annotated.jpeg').replace('.png', '_annotated.png'))
cv2.imwrite(output_path, image)

# Plot predictions
plt.figure(figsize=(8, 8))
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Predictions')
plt.show()

print(f"Annotated image saved at: {output_path}")
