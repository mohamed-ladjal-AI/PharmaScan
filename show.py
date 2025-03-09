import cv2
import os

# Paths to the folders
image_folder = 'reconstracted all'
labels_folder = 'Labels'
classes_file = os.path.join(labels_folder, 'classes.txt')

# Load class names
with open(classes_file, 'r') as f:
    class_names = [line.strip() for line in f.readlines()]

# Get screen size for proper scaling
screen_width = 1280  # Set your screen width (in pixels)
screen_height = 720  # Set your screen height (in pixels)

# Function to resize the image to fit the screen
def resize_image_to_fit_screen(image):
    height, width = image.shape[:2]
    scaling_factor = min(screen_width / width, screen_height / height)
    return cv2.resize(image, (int(width * scaling_factor), int(height * scaling_factor)), interpolation=cv2.INTER_AREA)

# Function to display a YOLO formatted dataset
def display_yolo_dataset(image_folder, labels_folder):
    image_files = [f for f in os.listdir(image_folder) if f.endswith(('.jpg', '.jpeg', '.png'))]

    for image_file in image_files:
        image_path = os.path.join(image_folder, image_file)
        label_path = os.path.join(labels_folder, image_file.replace('.jpg', '.txt').replace('.jpeg', '.txt').replace('.png', '.txt'))

        # Load image
        image = cv2.imread(image_path)
        height, width = image.shape[:2]

        # Resize image to fit the screen
        image_resized = resize_image_to_fit_screen(image)

        # Load annotations
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                annotations = f.readlines()

            # Draw bounding boxes on the resized image
            for annotation in annotations:
                class_id, x_center, y_center, box_width, box_height = map(float, annotation.strip().split())

                # Convert YOLO format (normalized) back to pixel coordinates
                x_center, y_center = int(x_center * width), int(y_center * height)
                box_width, box_height = int(box_width * width), int(box_height * height)
                x_min = int(x_center - box_width / 2)
                y_min = int(y_center - box_height / 2)
                x_max = int(x_center + box_width / 2)
                y_max = int(y_center + box_height / 2)

                # Adjust bounding box for resized image
                scaling_factor = min(screen_width / width, screen_height / height)
                x_min = int(x_min * scaling_factor)
                y_min = int(y_min * scaling_factor)
                x_max = int(x_max * scaling_factor)
                y_max = int(y_max * scaling_factor)

                # Draw the rectangle and label
                cv2.rectangle(image_resized, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                label = class_names[int(class_id)]
                cv2.putText(image_resized, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Display the image
        cv2.imshow('YOLO Dataset', image_resized)
        
        # Wait for user input to proceed to next image
        if cv2.waitKey(0) & 0xFF == ord('q'):  # Press 'q' to exit
            break

    cv2.destroyAllWindows()

# Call the function
display_yolo_dataset(image_folder, labels_folder)
