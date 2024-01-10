import cv2
import numpy as np
from skimage.transform import resize
from PIL import Image
from sklearn.cluster import MiniBatchKMeans
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

# Function to calculate keypoints and descriptors for an image using dense SIFT and a Gaussian pyramid
def dsift_features_gaussian_pyramid(img, steps, pyramid_levels=3):
    keypoints_all = []
    descriptors_all = []
    sift = cv2.SIFT_create()
    current_img = img

    # Iteratively downscale the image to create a Gaussian pyramid
    for _ in range(pyramid_levels):
        rows, columns = current_img.shape
        keypoints = []

        # Generate keypoints on a dense grid
        for y in range(0, rows, steps*2):
            for x in range(0, columns, steps*2):
                keypoints.append(cv2.KeyPoint(x, y, steps*2))

        # Compute SIFT descriptors for each keypoint
        _, descriptors = sift.compute(current_img, keypoints)
        keypoints_all.extend(keypoints)
        descriptors_all.extend(descriptors if descriptors is not None else [])

        # Downscale the image for the next level of the pyramid
        current_img = cv2.pyrDown(current_img)

    return keypoints_all, np.array(descriptors_all)

# Function to resize an image to a set dimension, ensuring uniformity in feature extraction
def resizeImage(image, height, width):
    img_resized = resize(image, (height, width))
    img_resized = Image.fromarray((img_resized * 255).astype(np.uint8))
    npimage_resized = np.asarray(img_resized)

    return npimage_resized

# Function to load and preprocess training images, ensuring uniform dimensions
def import_training():
    images = []
    targets = []

    # Load each image, resize it, and associate it with a target label
    for class_i in range(total_classes):
        for img in range(total_image_per_class):
            if img % 20 == 0:
                print(((class_i * total_image_per_class + img) / (total_classes * total_image_per_class)) * 100)
            image = cv2.imread(f'training/{classes[class_i]}/{img}.jpg', cv2.IMREAD_GRAYSCALE)
            img_resized = resizeImage(image, 200, 200)
            npimage_resized = np.asarray(img_resized)

            images.append(npimage_resized)
            targets.append(class_i)

    return images, targets

# Function to load and preprocess testing images
def import_testing():
    images = []

    # Load each testing image and resize it
    for img in range(2987):
        if img not in [1314, 2938, 2962]:
            if img % 20 == 0:
                print((img / 2987) * 100)
            image = cv2.imread(f'testing/{img}.jpg', cv2.IMREAD_GRAYSCALE)
            img_resized = resizeImage(image, 200, 200)
            img_resized = Image.fromarray((img_resized * 255).astype(np.uint8))
            npimage_resized = np.asarray(img_resized)

            images.append(npimage_resized)
    return images

# Function to generate histograms of visual words for each image
def generate_feature_histograms(images):
    histograms = []

    # Process each image to create its histogram
    for i, image in enumerate(images):
        if i % 10 == 0:
            print((i / len(images)) * 100)

        kp, des = dsift_features_gaussian_pyramid(image, 3, 3)

        histogram = np.zeros(k)
        total_kp = np.size(kp)

        # Assign each descriptor to a cluster and update the histogram
        for d in des:
            idx = kmeans.predict([d])
            histogram[idx] += 1 / total_kp

        histograms.append(histogram)

    return np.array(histograms)

# Class names and parameters for the classification task
classes = np.array(['bedroom', 'Coast', 'Forest', 'Highway', 'industrial', 'Insidecity', 'kitchen', 'livingroom', 'Mountain', 'Office', 'OpenCountry', 'store', 'Street', 'Suburb', 'TallBuilding'])
step = 4
total_classes = len(classes)
total_image_per_class = 100

# Load images and split them into training and validation sets
print("Importing images...")
images, targets = import_training()
images_final_test = import_testing()

images_train, images_validate, targets_train, targets_validate = train_test_split(images, targets, train_size=0.9, random_state = 0)

# Extract SIFT features from all training images
print("Calculating SIFT features...")
features = []
for i, image in enumerate(images_train):
    if i % 20 == 0:
        print((i / len(images_train)) * 100)
    kp, des = dsift_features_gaussian_pyramid(image, 3, 3)
    for d in des:
        features.append(d)

# Apply KMeans clustering to the aggregated SIFT features
print("Calculating KMeans of all SIFT features for each image..")
k = total_classes * 10
batch_size = 128
kmeans = MiniBatchKMeans(n_clusters=k, batch_size=batch_size, verbose=1).fit(features)
kmeans.verbose = False

# Generate histograms for the training set
print("Generating histograms for training set...")
X_train = generate_feature_histograms(images_train)
y_train = targets_train

# normalise features before feeding them into the classifier
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

# Train the MLP Classifier on the normalised histograms
print("Training MLP Classifier...")
mlp = MLPClassifier(max_iter=500000)
mlp.fit(X_train, y_train)

# Prepare validation data and evaluate the classifier
print("Generating histograms for validation set...")
X_validate = generate_feature_histograms(images_validate)
X_validate = scaler.transform(X_validate)
y_validate = targets_validate

print("Testing validation set with MLP...")
mlp_correct = sum(mlp.predict(X_validate) == y_validate)
print(f"MLP Accuracy: {mlp_correct / len(X_validate)}\n")

# Generate histograms for the test set and predict labels
print("Generating histograms for test set...")
X_final_test = generate_feature_histograms(images_final_test)
X_final_test = scaler.transform(X_final_test)

mlp_predictions = mlp.predict_proba(X_final_test)
f = open("run3.txt", "a")
for i, pred in enumerate(mlp_predictions):
    i_adjusted = i + (1 if i >= 1314 else 0) + (1 if i >= 2938 else 0) + (1 if i >= 2962 else 0)
    np_pred = np.array(pred)
    f.write(f"{i_adjusted}.jpg {classes[np.argmax(np_pred)]}\n")
f.close()