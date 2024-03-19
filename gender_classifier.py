# Import necessary libraries
import numpy as np
import tensorflow as tf
import requests
from PIL import Image
import json
from io import BytesIO
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define constants
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10

# Define model class
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
        self.add(layers.Dense(2, activation='softmax'))  # 2 classes: male/female

# Load JSON data
with open('scraped_data.json') as f:
    data = json.load(f)

# Initialize lists for image data and labels
image_data = []
gender_labels = []

# preprocess image
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
    
    image_data.append(img_data)
    
    if value['gender'] == 'Men':
        gender_labels.append(0)  # Male
    else:
        gender_labels.append(1)  # Female

# Convert lists to numpy arrays
image_data = np.array(image_data)
gender_labels = np.array(gender_labels)

# Check shapes
print("Image Data Shape:", image_data.shape)
print("Gender Labels Shape:", gender_labels.shape)

# Define CNN model
model = YourModelClass()

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Data augmentation
train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
train_generator = train_datagen.flow(image_data, gender_labels, batch_size=BATCH_SIZE)

# Train the model
model.fit(train_generator, epochs=EPOCHS)

# Testing using dummy data
test_data = np.random.rand(100, 224, 224, 3)  
test_labels = np.random.randint(2, size=100)  

# Evaluate the model
test_loss, test_acc = model.evaluate(test_data, test_labels)
print('Test accuracy:', test_acc*100)