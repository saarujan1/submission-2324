import numpy as np
import cv2
import os
import datetime
from natsort import natsorted

from PIL import Image
from sklearn.neighbors import KNeighborsClassifier


# Create tiny image representation
def create_tiny_image(image):
    # TODO: refactor
    rows, cols = image.shape
    center_x = cols // 2
    center_y = rows // 2
    side = min(rows, cols)
    start_x = center_x - side // 2
    start_y = center_y - side // 2
    cropped = image[start_y:start_y + side, start_x:start_x + side]
    tiny_image = cv2.resize(cropped, (16, 16), cv2.INTER_AREA)
    tiny_image = tiny_image.flatten()
    # Normalise
    tiny_image = tiny_image - np.mean(tiny_image)
    tiny_image /= np.linalg.norm(tiny_image)
    return tiny_image

# Return all images in a folder
def load_images_from_folder(folder_path):
    images = []
    file_list = natsorted(os.listdir(folder_path))
    for filename in file_list:
        if filename.endswith(('.jpg', '.png', '.jpeg')):
            image = Image.open(os.path.join(folder_path, filename))
            images.append(np.asarray(image))
            print("debug: filename: " + filename );
    return images

# Return the name of all the folders to get the labels for each category
def get_label_names(folder_path):
    label_name = []
    for label in os.listdir(folder_path):
        item_path = os.path.join(folder_path, label)
        if os.path.isdir(item_path):
            label_name.append(label)
    return label_name

# Load and prepare training images
train_images = []
train_labels = []
test_images = []
test_labels_predicted = []
k_value = 15
split_val = 80

label_names = get_label_names('training/training')

# Import training data
for label in label_names:
    result = load_images_from_folder('training/training/{}'.format(label))
    train_images.extend(result)
    train_labels.extend([label] * len(result))

# Create tiny images from training date
for i in range(len(train_images)):
    train_images[i] = create_tiny_image(train_images[i])

train_images = np.array(train_images)

# Import test images
test_images = load_images_from_folder('testing/testing')

# Create tiny images from test data
for i in range(len(test_images)):
    test_images[i] = create_tiny_image(test_images[i])

train_images = np.array(train_images)

test_labels_predicted = []

# k-NN classifier fit.
knn = KNeighborsClassifier(n_neighbors=k_value)
knn.fit(train_images, train_labels)

# Predict the labels in test data
for image in test_images:
    test_labels_predicted.extend(knn.predict([image]))

# Write the result of predictions to file
timestamp = datetime.datetime.now().strftime("%d%H%M%S")
filename = f"{timestamp}... k = {k_value}- _results.txt"

with open(filename, 'w') as file:
    #file.write("k = {}".format(k_value) + '\n')
    counter = 0
    for item in test_labels_predicted:
        file.write("{}.jpg {}".format(counter, str(item)) + '\n')
        counter += 1

print("Done")