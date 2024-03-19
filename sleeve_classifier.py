# Import necessary libraries
import numpy as np
import tensorflow as tf
import requests
from PIL import Image
import json
from io import BytesIO
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

# Define constants
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10

# Define your model class
class YourModelClass(models.Sequential):
    def __init__(self):
        super().__init__()
        # Define your model layers here, similar to how you defined in your main script
        self.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3)))
        self.add(layers.MaxPooling2D((2, 2)))
        self.add(layers.Conv2D(64, (3, 3), activation='relu'))
        self.add(layers.MaxPooling2D((2, 2)))
        self.add(layers.Conv2D(128, (3, 3), activation='relu'))
        self.add(layers.MaxPooling2D((2, 2)))
        self.add(layers.Conv2D(128, (3, 3), activation='relu'))
        self.add(layers.MaxPooling2D((2, 2)))
        self.add(layers.Flatten())
        self.add(layers.Dropout(0.5))
        self.add(layers.Dense(512, activation='relu'))
        self.add(layers.Dense(2, activation='softmax', name='gender_output'))  # 2 classes: male/female
        self.add(layers.Dense(2, activation='softmax', name='sleeve_output'))  # Sleeve type output

# Load JSON data
with open('scraped_data.json') as f:
    data = json.load(f)

# Initialize lists for image data and labels
image_data = []
gender_labels = []
sleeve_labels = []

# Function to preprocess image
def preprocess_image(url):
    response = requests.get(url, timeout=30)
    img = Image.open(BytesIO(response.content))
    img = img.resize((224, 224))  # Resize images to a uniform size
    img = np.array(img) / 255.0  # Normalize pixel values
    return img

# Download images and assign labels
for key, value in data.items():
    # Download and preprocess image
    img_data = preprocess_image(value['img'])
    
    # Append image data to list
    image_data.append(img_data)
    
    # Assign gender label
    if value['gender'] == 'Men':
        gender_labels.append(0)  # Male
    else:
        gender_labels.append(1)  # Female
    # Assign sleeve-type label
    if value['sleeve_type'] == 'full sleeve':
        sleeve_labels.append(0)  # full sleeve
    else:
        sleeve_labels.append(1)  # half sleeve

# Convert lists to numpy arrays
image_data = np.array(image_data)
gender_labels = np.array(gender_labels)
sleeve_labels = np.array(sleeve_labels)

# Split the dataset into training and testing sets
X_train, X_test, gender_train, gender_test, sleeve_train, sleeve_test = train_test_split(image_data, gender_labels, sleeve_labels, test_size=0.2, random_state=42)
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
# Define CNN model
model = YourModelClass()

# Compile the model
model.compile(optimizer='adam',
              loss={'gender_output': 'sparse_categorical_crossentropy', 'sleeve_output': 'sparse_categorical_crossentropy'},
              metrics={'gender_output': 'accuracy', 'sleeve_output': 'accuracy'})


# Data augmentation for training set
train_datagen = ImageDataGenerator(rescale=1./255)
# Combine gender and sleeve labels into a single dictionary
train_labels = np.concatenate((gender_train.reshape(-1, 1), sleeve_train.reshape(-1, 1)), axis=1)

# Create a generator for training data
train_generator = train_datagen.flow(X_train, train_labels, batch_size=BATCH_SIZE)

# Data augmentation for testing set (only rescale, no augmentation)
test_datagen = ImageDataGenerator(rescale=1./255)
test_labels = np.concatenate((gender_test.reshape(-1, 1), sleeve_test.reshape(-1, 1)), axis=1)
test_generator = test_datagen.flow(X_test, test_labels, batch_size=BATCH_SIZE)

# Train the model
model.fit(train_generator, epochs=EPOCHS, validation_data=test_generator)

# Evaluate the model
test_loss, gender_test_loss, sleeve_test_loss, gender_test_acc, sleeve_test_acc = model.evaluate(test_generator)
print('Gender Test accuracy:', gender_test_acc*100)
print('Sleeve Test accuracy:', sleeve_test_acc*100)
