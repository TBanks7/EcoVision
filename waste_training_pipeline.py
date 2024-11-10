import tensorflow as tf
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from datetime import datetime
from waste_classification_training import TRAINING_CONFIG

def train_waste_classification():
    """
    Main training pipeline that coordinates all the training functions
    """
    print("Starting waste classification training pipeline...")

    # 1. Setup paths and configurations
    BASE_PATH = "dataset/waste_images"
    MODEL_SAVE_PATH = "models/waste_classifier"
    LOG_DIR = "logs/waste_classifier/" + datetime.now().strftime("%Y%m%d-%H%M%S")

    # 2. Create data generators
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        BASE_PATH,
        validation_split=TRAINING_CONFIG['validation_split'],
        subset="training",
        seed=123,
        image_size=TRAINING_CONFIG['input_size'],
        batch_size=TRAINING_CONFIG['batch_size']
    )

    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        BASE_PATH,
        validation_split=TRAINING_CONFIG['validation_split'],
        subset="validation",
        seed=123,
        image_size=TRAINING_CONFIG['input_size'],
        batch_size=TRAINING_CONFIG['batch_size']
    )

    # 3. Configure dataset performance
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    # 4. Create data augmentation layer
    data_augmentation = create_data_augmentation()

    # 5. Create and compile the model
    model = create_school_waste_model()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(TRAINING_CONFIG['learning_rate']),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # 6. Setup callbacks
    callbacks = [
        # Save best model
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(MODEL_SAVE_PATH, 'best_model'),
            save_best_only=True,
            monitor='val_accuracy'
        ),
        # Early stopping
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        ),
        # TensorBoard logging
        tf.keras.callbacks.TensorBoard(
            log_dir=LOG_DIR,
            histogram_freq=1
        ),
        # Learning rate reduction on plateau
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=3,
            min_lr=1e-6
        )
    ]

    # 7. Train the model
    print("Starting model training...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=TRAINING_CONFIG['epochs'],
        callbacks=callbacks,
        class_weight=TRAINING_CONFIG['class_weights']
    )

    # 8. Save the final model
    model.save(os.path.join(MODEL_SAVE_PATH, 'final_model'))

    # 9. Convert to TensorFlow.js format
    print("Converting model to TensorFlow.js format...")
    import tensorflowjs as tfjs
    tfjs.converters.save_keras_model(model, os.path.join(MODEL_SAVE_PATH, 'tfjs_model'))

    # 10. Evaluate the model
    print("Evaluating model performance...")
    evaluation = model.evaluate(val_ds)
    
    # 11. Save training history and metrics
    save_training_results(history, evaluation, MODEL_SAVE_PATH)

    return model, history

def save_training_results(history, evaluation, save_path):
    """
    Save training history and evaluation metrics
    """
    # Convert history to dataframe
    history_df = pd.DataFrame(history.history)
    history_df.to_csv(os.path.join(save_path, 'training_history.csv'))

    # Save evaluation metrics
    metrics = {
        'test_loss': evaluation[0],
        'test_accuracy': evaluation[1],
        'final_training_accuracy': history.history['accuracy'][-1],
        'final_validation_accuracy': history.history['val_accuracy'][-1],
        'training_epochs': len(history.history['accuracy'])
    }
    
    pd.Series(metrics).to_json(os.path.join(save_path, 'evaluation_metrics.json'))

def plot_training_results(history):
    """
    Plot training and validation metrics
    """
    import matplotlib.pyplot as plt

    # Plot training history
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Accuracy plot
    ax1.plot(history.history['accuracy'], label='Training Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()

    # Loss plot
    ax2.plot(history.history['loss'], label='Training Loss')
    ax2.plot(history.history['val_loss'], label='Validation Loss')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Verify GPU availability
    print("GPU Available: ", tf.config.list_physical_devices('GPU'))
    
    # Create necessary directories
    os.makedirs("models/waste_classifier", exist_ok=True)
    os.makedirs("logs/waste_classifier", exist_ok=True)

    try:
        # Run the training pipeline
        model, history = train_waste_classification()
        
        # Plot results
        plot_training_results(history)
        
        print("Training completed successfully!")
        
    except Exception as e:
        print(f"Error during training: {str(e)}")

# Expected directory structure:
# dataset/
#   waste_images/
#     paper/
#       image1.jpg
#       image2.jpg
#       ...
#     containers/
#       image1.jpg
#       image2.jpg
#       ...
#     compost/
#       image1.jpg
#       image2.jpg
#       ...
#     waste/
#       image1.jpg
#       image2.jpg
#       ...
