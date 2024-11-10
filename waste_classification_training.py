import tensorflow as tf
from tensorflow.keras import layers, applications

def create_school_waste_model(input_shape=(224, 224, 3)):
    """
    Create a model specifically trained for school waste classification
    """
    # Use EfficientNetV2B0 as base model (good balance of size/performance)
    base_model = applications.EfficientNetV2B0(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )
    
    # Freeze base model layers
    base_model.trainable = False
    
    # Create school-specific classification model
    model = tf.keras.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.2),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.2),
        # 4 output classes: Paper, Containers, Compost, Waste
        layers.Dense(4, activation='softmax')
    ])
    
    return model

def create_data_augmentation():
    """
    Create data augmentation pipeline specific to waste images
    """
    return tf.keras.Sequential([
        layers.RandomRotation(0.2),
        layers.RandomZoom(0.2),
        layers.RandomFlip("horizontal"),
        layers.RandomBrightness(0.2),
        layers.RandomContrast(0.2),
    ])

# Custom training configuration
TRAINING_CONFIG = {
    'categories': ['paper', 'containers', 'compost', 'waste'],
    'input_size': (224, 224),
    'batch_size': 32,
    'epochs': 50,
    'learning_rate': 0.001,
    'validation_split': 0.2,
    # Class weights to handle imbalanced datasets
    'class_weights': {
        0: 1.0,  # paper
        1: 1.0,  # containers
        2: 1.2,  # compost (slightly higher weight due to variety)
        3: 1.1   # waste
    }
}

# Data preprocessing function
def preprocess_image(image_path):
    """
    Preprocess images according to school waste classification needs
    """
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, TRAINING_CONFIG['input_size'])
    image = tf.cast(image, tf.float32) / 127.5 - 1
    return image
