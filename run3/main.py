import cv2
import numpy as np
from skimage.transform import resize
from PIL import Image
from sklearn.cluster import MiniBatchKMeans
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from scipy.stats import uniform, randint

# Calculate keypoints and descriptors for a given image
# Using dense SIFT of size step
def dsift_features_gaussian_pyramid(img, steps, pyramid_levels=3):
    keypoints_all = []
    descriptors_all = []
    sift = cv2.SIFT_create()
    current_img = img

    for _ in range(pyramid_levels):
        rows, columns = current_img.shape
        keypoints = []

        for y in range(0, rows, steps*2):
            for x in range(0, columns, steps*2):
                keypoints.append(cv2.KeyPoint(x, y, steps*2))

        _, descriptors = sift.compute(current_img, keypoints)
        keypoints_all.extend(keypoints)
        descriptors_all.extend(descriptors if descriptors is not None else [])

        current_img = cv2.pyrDown(current_img)

    return keypoints_all, np.array(descriptors_all)
# Resize image to set dimensions so that they all produce the same amount of sift features
# Return as numpy array
def resizeImage(image, height, width):
    img_resized = resize(image, (height, width))
    img_resized = Image.fromarray((img_resized * 255).astype(np.uint8))
    npimage_resized = np.asarray(img_resized)

    return npimage_resized

# Import training images 
def import_training():
    images = []
    targets = []

    for class_i in range(total_classes):
        for img in range(total_image_per_class):
            if img%20 == 0:
                print(((class_i*total_image_per_class+img)/(total_classes*total_image_per_class))*100)
            image = cv2.imread(f'training/{classes[class_i]}/{img}.jpg', cv2.IMREAD_GRAYSCALE)
            img_resized = resize(image, (200, 200))
            img_resized = Image.fromarray((img_resized * 255).astype(np.uint8))
            npimage_resized = np.asarray(img_resized)

            images.append(npimage_resized)
            targets.append(class_i)

    return images, targets

# Import testing images 
def import_testing():
    images = []

    for img in range(2987):
        if img not in [1314, 2938, 2962]:
            if img%20 == 0:
                print((img/2987)*100)
            image = cv2.imread(f'testing/{img}.jpg', cv2.IMREAD_GRAYSCALE)
            img_resized = resize(image, (200, 200))
            img_resized = Image.fromarray((img_resized * 255).astype(np.uint8))
            npimage_resized = np.asarray(img_resized)

            images.append(npimage_resized)
    return images

def generate_histograms(images):
    histograms = []

    for i, image in enumerate(images):
        if i%10 == 0:
            print((i/len(images))*100)

        kp, des = dsift_features_gaussian_pyramid(image, 3, 3)

        histogram = np.zeros(k)
        total_kp = np.size(kp)

        for d in des:
            idx = kmeans.predict([d])
            histogram[idx] += 1/total_kp

        histograms.append(histogram)

    return np.array(histograms)

classes = np.array(['bedroom', 'Coast', 'Forest', 'Highway', 'industrial', 'Insidecity', 'kitchen', 'livingroom', 'Mountain', 'Office', 'OpenCountry', 'store', 'Street', 'Suburb', 'TallBuilding'])
step = 4
total_classes = len(classes)
total_image_per_class = 100

# Save all images, and their associated class, in an array then divide into training and validation sets
print("Importing images...")
images, targets = import_training()
images_final_test = import_testing()

images_train, images_validate, targets_train, targets_validate = train_test_split(images, targets, train_size=0.9, random_state = 0)

# Get all SIFT features from all images and store in an array
print("Calculating SIFT features...")
features = []
for i, image in enumerate(images_train):
    if i%20 == 0:
        print((i/len(images_train))*100)
    kp, des = dsift_features_gaussian_pyramid(image, 3, 3)
    for d in des:
        features.append(d)

# Calculate KMeans over the features
print("Calculating KMeans of all SIFT features for each image..")
k = total_classes * 10
batch_size = 128
kmeans = MiniBatchKMeans(n_clusters=k, batch_size=batch_size, verbose=1).fit(features)
kmeans.verbose = False

# Generating histograms for each image using the SIFT clusters
print("Generating histograms for training set...")
X_train = generate_histograms(images_train)
y_train = targets_train

# Feature Normalization
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

# Training MLP Classifier
print("Training MLP Classifier...")
mlp = MLPClassifier(max_iter=500000)
mlp.fit(X_train, y_train)

# Generate histograms for Validation data
print("Generating histograms for validation set...")
X_validate = generate_histograms(images_validate)
X_validate = scaler.transform(X_validate)
y_validate = targets_validate

# Testing validation set with MLP
print("Testing validation set with MLP...")
mlp_correct = sum(mlp.predict(X_validate) == y_validate)
print(f"MLP Accuracy: {mlp_correct / len(X_validate)}\n")

# Generate histograms for test data
print("Generating histograms for test set...")
X_final_test = generate_histograms(images_final_test)
X_final_test = scaler.transform(X_final_test)

# Generating labels for test set
mlp_predictions = mlp.predict_proba(X_final_test)
f = open("run3.txt", "a")
for i, pred in enumerate(mlp_predictions):
    i_adjusted = i + (1 if i >= 1314 else 0) + (1 if i >= 2938 else 0) + (1 if i >= 2962 else 0)
    np_pred = np.array(pred)
    f.write(f"{i_adjusted}.jpg {classes[np.argmax(np_pred)]}\n")
f.close()