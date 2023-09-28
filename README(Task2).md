# Mega-Project

# YOLOv8 Object Detection with Roboflow and Ultralytics

This repository contains code for performing object detection using YOLOv8, a state-of-the-art object detection algorithm, in combination with Roboflow for dataset management and Ultralytics for training and inference.

## Prerequisites

Before you begin, make sure you have the following prerequisites installed:

- NVIDIA GPU (for GPU acceleration)
- [NVIDIA System Management Interface (nvidia-smi)](https://developer.nvidia.com/nvidia-system-management-interface)
- Python 3
- [Ultralytics](https://github.com/ultralytics/ultralytics)
- [Roboflow](https://pypi.org/project/roboflow/)

You can install Ultralytics and Roboflow using pip:

```python
!pip install ultralytics==8.0.20
!pip install roboflow
```



# YOLOv8 Object Detection Training and Inference

This section provides code for training and inference using YOLOv8 for object detection tasks. The code includes training, validation, and prediction steps.

## Training

To train a YOLOv8 model, use the following command:

```python
%cd {HOME}
!yolo task=detect mode=train model=yolov8s.pt data={dataset.location}/data.yaml epochs=25 imgsz=800 plots=True
```
After training, the model weights and results will be saved in the {HOME}/runs/detect/train3/ directory.
You can view the training results using the following command:
```python
%cd {HOME}
!ls {HOME}/runs/detect/train3/
```


## Validation
To perform validation using the trained model, use the following command:
```python
%cd {HOME}
!yolo task=detect mode=val model={HOME}/runs/detect/train3/weights/best.pt data={dataset.location}/data.yaml
```
![image](https://github.com/NorhanM-A/Mega-Project/assets/72838396/17f1ceae-369a-43e3-a4c9-5e6b7bbe5cfb)



# Object Detection Result Analysis and Cropping

This section of the repository contains code for analyzing object detection results and cropping objects from images. The code utilizes OpenCV for image processing and visualization.

## Prerequisites

Before running the code, make sure you have the following prerequisites installed:
- Python 3
- OpenCV (`cv2` library)
- [IPython.display](https://ipython.org/ipython-doc/stable/api/generated/IPython.display.html) for displaying images

You can install OpenCV and IPython.display using pip:

```python
pip install opencv-python-headless
pip install ipython
```

### Usage
1. Define the directories for label files, image files, and the output directory where cropped images will be saved.

2. Customize the class mapping according to your dataset. The provided code includes an example class mapping.

3. Set the maximum number of images you want to display and adjust the margin value for cropping as needed.

4. Run the code to analyze object detection results and crop objects from images.
```python
import os
import glob
import cv2
from IPython.display import Image, display  # Import the Image and display functions from IPython.display

# Directory where label files are stored
label_dir = '/content/datasets/suas-final-1/test/labels/'
image_dir = '/content/datasets/suas-final-1/test/images/'  # Updated image directory path
output_dir = '/content/test_cropped/'  # Directory to save cropped images

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Define your class mapping here
class_mapping = {
    0: '0',
    1: '1',
    2: '2',
    3: '3',
    4: '4',
    5: '5',
    6: '6',
    7: '7',
    8: '8',
    9: '9',
    10: 'A',
    11: 'B',
    12: 'C',
    13: 'D',
    14: 'E',
    15: 'F',
    16: 'G',
    17: 'H',
    18: 'I',
    19: 'J',
    20: 'K',
    21: 'L',
    22: 'M',
    23: 'N',
    24: 'O',
    25: 'P',
    26: 'Q',
    27: 'R',
    28: 'S',
    29: 'T',
    30: 'U',
    31: 'V',
    32: 'W',
    33: 'X',
    34: 'Y',
    35: 'Z',
    36: 'circle',
    37: 'cross',
    38: 'g',
    39: 'hexagon',
    40: 'pentagon',
    41: 'plus',
    42: 'rectangle',
    43: 'star',
    44: 'trapezium',
    45: 'triangle'
}

# Function to get class name based on class index
def get_class_name(class_index):
    return class_mapping.get(class_index, str(class_index))

# Counter to keep track of displayed images
displayed_images = 0

# Maximum number of images to display
max_images_to_display = 15

# Define a margin value (as a fraction of the bounding box width and height)
margin = 0.5  # Adjust this value as needed (e.g., 10% margin)

# Iterate through label files and extract results
for label_path in glob.glob(os.path.join(label_dir, '*.txt')):
    if displayed_images >= max_images_to_display:
        break  # Exit the loop after displaying the desired number of images

    # Get the image file corresponding to this label
    image_filename = os.path.basename(label_path).replace('.txt', '.jpg')  # Extract image filename from label path
    image_path = os.path.join(image_dir, image_filename)  # Full path to the image

    # Read the label file
    with open(label_path, 'r') as file:
        lines = file.readlines()

    # Initialize lists to store class and bounding box information
    class_indices = []
    bounding_boxes = []

    for line in lines:
        data = line.strip().split()
        if len(data) == 5:
            # Extract class index, x, y, width, and height from label
            class_index = int(data[0])
            x_center = float(data[1])
            y_center = float(data[2])
            width = float(data[3])
            height = float(data[4])

            # Expand the bounding box by applying margin
            x_center -= width * margin / 2
            y_center -= height * margin / 2
            width += width * margin
            height += height * margin

            # Append class index and modified bounding box information to the lists
            class_indices.append(class_index)
            bounding_boxes.append((x_center, y_center, width, height))



    # Now you have lists containing class indices and bounding box information for each class in the image
    

    # Load the image using OpenCV
    img = cv2.imread(image_path)

    # Print class names
    for i in range(len(class_indices)):
        class_index = class_indices[i]
        class_name = get_class_name(class_index)
        print(f"Class Name {i+1}: {class_name}")

    # Print bounding box coordinates for bounding box 1
    if len(bounding_boxes) >= 2:
        print(f"Bounding Box 2 (x_center, y_center, width, height): {bounding_boxes[1]}")

        # Get the coordinates of bounding box 1
        x_center, y_center, width, height = bounding_boxes[1]

        # Calculate the top-left and bottom-right coordinates of the bounding box
        x1 = int((x_center - width / 2) * img.shape[1])
        y1 = int((y_center - height / 2) * img.shape[0])
        x2 = int((x_center + width / 2) * img.shape[1])
        y2 = int((y_center + height / 2) * img.shape[0])

        # Crop the image to the bounding box coordinates
        cropped_img = img[y1:y2, x1:x2]

        # Save the cropped image
        output_filename = os.path.join(output_dir, f'croped_{os.path.basename(image_path)}')
        cv2.imwrite(output_filename, cropped_img)
        print(f"Cropped image saved as: {output_filename}")

    # Display the original image
    display(Image(filename=image_path))

    print("\n")

    # Increment the displayed images counter
    displayed_images += 1
```

![6](https://github.com/NorhanM-A/Mega-Project/assets/72838396/073e1098-531c-408c-8c39-81489abbe245)
![5](https://github.com/NorhanM-A/Mega-Project/assets/72838396/8cce2766-5093-422a-b3ea-80ab96e0472a)
![4](https://github.com/NorhanM-A/Mega-Project/assets/72838396/0b6f8e4b-cff9-4447-90ca-a1689d8b7b7d)
![3](https://github.com/NorhanM-A/Mega-Project/assets/72838396/c4174a96-4c65-4a77-b8cb-8473fc48ec57)
![2](https://github.com/NorhanM-A/Mega-Project/assets/72838396/5980d3b3-e65e-4173-822f-b37abbe538a0)
![1](https://github.com/NorhanM-A/Mega-Project/assets/72838396/34d26af4-6493-4f8e-882a-3fbd32b00d1e)
![7](https://github.com/NorhanM-A/Mega-Project/assets/72838396/7a83caf1-504f-4873-90c2-43b929e22e85)





# Shape and Alphanumeric Detection and Classification

Before analyzing the unique colors in the dataset, we performed a series of steps to detect and classify shapes and alphanumeric characters within the dataset. These steps are crucial for understanding the composition of the images and preparing the data for color analysis.

## Shape and Alphanumeric Detection

We first employed object detection techniques to identify shapes and alphanumeric characters within the images. This involved using advanced algorithms,  including YOLO (You Only Look Once), to locate and delineate these objects accurately.

The resulting data includes information such as the coordinates of bounding boxes, class labels, and confidence scores for each detected object.

## Classification of Shapes and Alphanumeric Characters

Following the detection step, we performed classification to categorize the detected shapes and alphanumeric characters. 

The output of the classification step allows us to understand what specific shapes (e.g., rectangles, circles) and alphanumeric characters (e.g., letters and numbers) are present in the dataset.

## Unique Color Analysis

After successfully detecting and classifying shapes and alphanumeric characters, we proceeded to analyze the unique colors present in the dataset. The previous steps helped us gain insights into the composition of the images, allowing us to better understand the color distribution within each object category.






# Unique Colors Analysis

In this section, we perform an analysis of unique colors present in the dataset using the `pandas` library in Python.

## Prerequisites

Ensure you have already read the data from the CSV file as demonstrated in this code:
```python
import pandas as pd

filepath="/content/final_data_colors.csv"

data_set=pd.read_csv(filepath)
```

## Analysis

To identify the unique colors in the dataset and count their occurrences, we can use the following code:

```python
from ctypes import sizeof
print("colors:", data_set['label'].unique(),"\n\ntotal number of colors:", len(data_set['label'].unique()))
```
output: colors: ['Blue' 'Brown' 'Green' 'Pink' 'Yellow' 'Orange' 'Purple' 'Red' 'Grey' 'White' 'Black'] 

total number of colors: 11





# Data Preprocessing: Normalizing RGB Values

In this section, we perform data preprocessing by normalizing the RGB values of colors in the dataset using the `sklearn` library in Python. The dataset was initially loaded from a CSV file, and now we are preparing it for further analysis.



## Data Preprocessing

First, we extract the RGB values and color names from the dataset:

```python
RGB_values = data_set[['red', 'green', 'blue']].values
Label_Names = data_set[['label']].values
```
Next, we normalize the RGB values using MinMaxScaler from sklearn:
from sklearn.preprocessing import MinMaxScaler

```python
# Normalize the RGB values
scaler = MinMaxScaler()
normalized_RGB_values = scaler.fit_transform(RGB_values)
```

The normalized RGB values are now in the normalized_RGB_values array.

To create a new dataset with normalized RGB values and color labels, we create a DataFrame:

```python
import pandas as pd

normalized_data_set = pd.DataFrame(normalized_RGB_values, columns=['normalized_red', 'normalized_green', 'normalized_blue'])
normalized_data_set['label'] = data_set[['label']].values
```

Finally, we save this normalized dataset to a new CSV file named 'normalized_colors.csv':
```python
normalized_data_set.to_csv('normalized_colors.csv', index=False)
```


# Dataset Splitting: Training, Validation, and Testing Sets

In this section, we split the dataset into three distinct sets: a training set, a validation set, and a testing set. This separation is essential for training and evaluating machine learning models effectively.

## Dataset Splitting

We use the `train_test_split` function from the `sklearn` library to perform the dataset split. Here's how the dataset is divided:

- **Training Set:** This set is used to train machine learning models. It comprises a significant portion of the dataset and is used to teach the model patterns and relationships between features and labels.

- **Validation Set:** The validation set is used to tune hyperparameters and monitor the model's performance during training. It helps prevent overfitting by providing an independent evaluation of the model's performance.

- **Testing Set:** The testing set is reserved for the final evaluation of the trained model. It should not be used during training or hyperparameter tuning. It allows us to assess how well the model generalizes to unseen data.

## Splitting Details

The dataset splitting is performed as follows:

```python
from sklearn.model_selection import train_test_split

X = normalized_data_set[['normalized_red', 'normalized_green', 'normalized_blue']]
y = normalized_data_set['label']

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
```
The output of this splitting process is as follows:

Total Dataset Size: 5052
Training Set Size: 4041
Validation Set Size: 505
Testing Set Size: 506



# Color Prediction using K-Nearest Neighbors (KNN)

In this section, we demonstrate the use of a K-Nearest Neighbors (KNN) classifier to predict the dominant colors present in images. The KNN model has been trained on the normalized RGB values of the dataset and is used to classify the colors of objects in images.


## Color Prediction

We use the trained KNN classifier to predict the colors of objects in images from a specified directory. Here's how the process works:

```python
#k-nearest
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

# Create a KNN classifier with k=5 (you can adjust this parameter)
knn_classifier = KNeighborsClassifier(n_neighbors=5)

# Fit the classifier on the training data
knn_classifier.fit(X_train, y_train)

# Predict the labels on the validation set
y_val_pred = knn_classifier.predict(X_val)

# Calculate the accuracy score
accuracy = accuracy_score(y_val, y_val_pred)

# Print the accuracy score
print("Accuracy on the validation set:", accuracy)

# Print the classification report
print("Classification Report:")
print(classification_report(y_val, y_val_pred))

#accuracy is 0.857423
```

# Hyperparameter Tuning and Learning Curve for K-Nearest Neighbors (KNN)

In this section, we perform hyperparameter tuning for the KNN classifier and visualize its learning curve. Hyperparameter tuning helps determine the optimal number of nearest neighbors (`n_neighbors`) for the KNN model, while the learning curve provides insights into the model's performance with varying amounts of training data.

## Hyperparameter Tuning

We experiment with different values of `n_neighbors` to find the best configuration for the KNN classifier. Here's the process:

1. Initialize a KNN classifier with a range of `n_neighbors` values (from 1 to 20).

2. Train the classifier on the training data and evaluate its accuracy on the validation set for each `n_neighbors` value.

3. Print the accuracy results and identify the best `n_neighbors` value that yields the highest accuracy.

The code provided performs this hyperparameter tuning and prints the best `n_neighbors` value along with the corresponding accuracy.
```python
#k-nearest
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

best_accuracy = 0.0
best_n_neighbors = None

# Try different values of n_neighbors
for n_neighbors in range(1, 21):
    knn_classifier = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn_classifier.fit(X_train, y_train)
    y_val_pred = knn_classifier.predict(X_val)
    accuracy = accuracy_score(y_val, y_val_pred)

    print(f"n_neighbors: {n_neighbors}, Accuracy: {accuracy:.4f}")

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_n_neighbors = n_neighbors

print("\nBest n_neighbors:", best_n_neighbors)
print("Best accuracy:", best_accuracy)


import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve

train_sizes, train_scores, test_scores = learning_curve(
    KNeighborsClassifier(n_neighbors=best_n_neighbors), X_train, y_train, cv=5,
    train_sizes=np.linspace(0.1, 1.0, 10), scoring='accuracy')

# Calculate mean and standard deviation for training and test scores
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

# Plot learning curve
plt.figure(figsize=(10, 6))
plt.title("Learning Curve for KNN Classifier")
plt.xlabel("Training Examples")
plt.ylabel("Accuracy")
plt.grid()

plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                 train_scores_mean + train_scores_std, alpha=0.1, color="r")
plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                 test_scores_mean + test_scores_std, alpha=0.1, color="g")
plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
         label="Training Accuracy")
plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
         label="Cross-validation Accuracy")

plt.legend(loc="best")
plt.show()
```
![image](https://github.com/NorhanM-A/Mega-Project/assets/72838396/76f49ae8-0c58-43f5-aeba-7c86959b9e25)



## Learning Curve

After selecting the best `n_neighbors` value, we visualize the learning curve to assess the KNN classifier's performance. The learning curve shows how the accuracy of the model changes as the size of the training dataset increases. This analysis helps us understand if the model is underfitting or overfitting.

We use the `learning_curve` function from `sklearn` to plot the learning curve. The curve includes both training and cross-validation accuracy scores for different training dataset sizes.
![image](https://github.com/NorhanM-A/Mega-Project/assets/72838396/ec5a2d53-bdbb-4949-b9f8-4c10f4080df0)

slected n neighbors=5 with an accuracy of 0.8574257425742574


# Color Prediction and Visualization using K-Nearest Neighbors (KNN)

In this section, we demonstrate how to predict the dominant colors in images using a trained K-Nearest Neighbors (KNN) classifier. We also visualize the results to assess the accuracy of color prediction.


## Color Prediction Process


```python
from colorthief import ColorThief
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import os

# Define a dictionary to map color labels to numerical values
color_label_mapping = {
    'Blue': 0,
    'Brown': 1,
    'Green': 2,
    'Pink': 3,
    'Yellow': 4,
    'Orange': 5,
    'Purple': 6,
    'Red': 7,
    'Grey': 8,
    'White': 9,
    'Black': 10
}

# Load your trained KNN model (X_train and y_train_numeric should be defined)
knn_classifier = KNeighborsClassifier(n_neighbors=5)
knn_classifier.fit(X_train, y_train_numeric)

# Directory containing image files
image_directory = '/content/test_cropped' 

# List all files in the directory
image_files = [os.path.join(image_directory, filename) for filename in os.listdir(image_directory) if filename.endswith(('.jpg', '.png', '.jpeg'))]

for image_path in image_files:
    color_thief = ColorThief(image_path)

    # Get the dominant color palette (top 3 dominant colors)
    palette = color_thief.get_palette(color_count=3)

    # Assuming palette is a list of (R, G, B) tuples, you can normalize it similarly
    normalized_palette = [scaler.transform([color])[0] for color in palette]  # Reshape to remove the outer list

    # Predict the color labels using the trained KNN classifier
    predicted_color_numerics = knn_classifier.predict(normalized_palette)

    # Convert the predicted numeric color labels back to the original color labels
    predicted_color_labels = [color_label_mapping_reverse[i] for i in predicted_color_numerics]

    # Display the image
    img = plt.imread(image_path)
    plt.figure(figsize=(8, 4))

    # Plot the original image
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.title("Original Image")
    plt.axis('off')

    # Plot the dominant colors as color swatches
    plt.subplot(1, 2, 2)
    plt.imshow([palette], aspect='auto')
    plt.title("Dominant Colors")
    plt.axis('off')

    # Display the predicted color labels
    plt.suptitle("Predicted Colors: " + ', '.join(predicted_color_labels))
    plt.show()
```

![image](https://github.com/NorhanM-A/Mega-Project/assets/72838396/cb206d72-0325-46b6-8ce5-cafa62334f20)

