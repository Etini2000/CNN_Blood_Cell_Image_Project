Blood Cell Classification using Convolutional Neural Networks (CNN)A deep learning project that builds, trains, and evaluates a Convolutional Neural Network (CNN) from scratch to classify white blood cell images (Monocytes vs. Lymphocytes) using TensorFlow and Keras.🔬 Project OverviewComputer vision plays a critical role in modern medical diagnostics. This project serves as a hands-on introduction to image classification by training a neural network to distinguish between two key types of white blood cells:MonocytesLymphocytesBy processing 1,000 blood cell images, the CNN extracts hierarchical visual patterns (such as nuclear shape, cytoplasmic texture, and cellular boundaries) to make accurate predictions.🛠️ Tech Stack & ToolsLanguage: Python 3Environment: Google Colab (GPU-accelerated runtime)Libraries: TensorFlow, Keras, NumPy, Matplotlib, scikit-learn📊 Dataset StructureThe dataset consists of image directories categorized by cell types:dataset/
├── Lymphocyte images/
│   ├── cell_001.jpeg
│   └── ...
└── Monocyte images/
    ├── cell_001.jpeg
    └── ...
🧠 CNN Model ArchitectureThe neural network is built using a sequential API consisting of feature-extraction blocks followed by classification layers:Rescaling Layer: Normalizes raw pixel values from [0, 255] to [0.0, 1.0].Convolutional Blocks:Conv2D (32 filters, $3 \times 3$ kernel, ReLU activation) + MaxPooling2D ($2 \times 2$)Conv2D (64 filters, $3 \times 3$ kernel, ReLU activation) + MaxPooling2D ($2 \times 2$)Conv2D (128 filters, $3 \times 3$ kernel, ReLU activation) + MaxPooling2D ($2 \times 2$)Classifier Layers:Flatten()Dense (64 units, ReLU activation)Dense (1 unit, Sigmoid activation for binary classification)Total Parameters: $\approx 3.3$ million trainable parameters.🚀 Step-by-Step Code Implementation1. Mount Google Drive & Load Dataimport tensorflow as tf
from tensorflow.keras import layers, models
import os

# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Define base directory
base_dir = '/content/drive/MyDrive/Your_Blood_Cell_Dataset_Path'

# Load dataset using TensorFlow utility
train_dataset = tf.keras.utils.image_dataset_from_directory(
    base_dir,
    image_size=(128, 128),
    batch_size=32,
    label_mode='binary',
    shuffle=True,
    seed=123
)

class_names = train_dataset.class_names
print("Class names found:", class_names)
2. Build and Compile the Modelmodel = models.Sequential([
    layers.Rescaling(1./255, input_shape=(128, 128, 3)),
    
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()
3. Train the ModelEPOCHS = 10
history = model.fit(train_dataset, epochs=EPOCHS)
4. Visualize Training Performanceimport matplotlib.pyplot as plt

acc = history.history['accuracy']
loss = history.history['loss']
epochs_range = range(len(acc))

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy', color='blue', marker='o')
plt.title('Training Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss', color='red', marker='o')
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.grid(True)
plt.show()
5. Make Predictions on Unseen Imagesimport numpy as np
from tensorflow.keras.preprocessing import image

def predict_blood_cell(img_path):
    img = image.load_img(img_path, target_size=(128, 128))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    
    prediction = model.predict(img_array)
    score = prediction[0][0]
    
    print(f"Raw prediction score: {score:.4f}")
    if score < 0.5:
        print(f"Prediction: This is likely a **{class_names[0]}**")
    else:
        print(f"Prediction: This is likely a **{class_names[1]}**")
        
    display(img)

# Example usage:
# predict_blood_cell('/content/drive/MyDrive/test_cell.jpeg')
📈 Results & PerformanceAfter training for 10 epochs, the model achieved a stable convergence with:Training Accuracy: $\approx 86.4\%$Loss: $\approx 0.3178$💾 Saving the Modelmodel.save('/content/drive/MyDrive/blood_cell_cnn_model.h5')
