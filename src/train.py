import os
import sys
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import json
from datetime import datetime

def configure_gpu():
    """Configure GPU settings for optimal performance"""
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Enable memory growth to avoid allocating all GPU memory at once
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            
            print(f"✅ Found {len(gpus)} GPU(s)")
            for i, gpu in enumerate(gpus):
                print(f"   GPU {i}: {gpu.name}")
        except RuntimeError as e:
            print(f"❌ GPU configuration error: {e}")
    else:
        print("⚠️  No GPUs found, using CPU")

# Fix Unicode encoding for Windows
if sys.platform.startswith('win'):
    os.environ['PYTHONIOENCODING'] = 'utf-8'

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model import MultiTaskCNN

class ModelTrainer:
    def __init__(self):
        # Get absolute paths
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.project_root = os.path.dirname(self.script_dir)
        self.data_dir = os.path.join(self.project_root, 'data', 'processed_data')
        self.models_dir = os.path.join(self.project_root, 'models')
        
        self.model = None
        self.mt_cnn = None
        self.history = None
        
        # Create models directory if it doesn't exist
        os.makedirs(self.models_dir, exist_ok=True)
        
        print(f"Project root: {self.project_root}")
        print(f"Data directory: {self.data_dir}")
        print(f"Models directory: {self.models_dir}")
        
    def load_data(self):
        """Load preprocessed data"""
        print("Loading preprocessed data...")
        
        try:
            # Check if files exist
            images_path = os.path.join(self.data_dir, 'images.npy')
            ages_path = os.path.join(self.data_dir, 'ages.npy')
            genders_path = os.path.join(self.data_dir, 'genders.npy')
            metadata_path = os.path.join(self.data_dir, 'metadata.pkl')
            
            print(f"Looking for:")
            print(f"  Images: {images_path} (exists: {os.path.exists(images_path)})")
            print(f"  Ages: {ages_path} (exists: {os.path.exists(ages_path)})")
            print(f"  Genders: {genders_path} (exists: {os.path.exists(genders_path)})")
            print(f"  Metadata: {metadata_path} (exists: {os.path.exists(metadata_path)})")
            
            # Load arrays
            images = np.load(images_path)
            ages = np.load(ages_path)
            genders = np.load(genders_path)
            
            # Load metadata
            with open(metadata_path, 'rb') as f:
                metadata = pickle.load(f)
            
            print(f"Data loaded successfully!")
            print(f"   - Images shape: {images.shape}")
            print(f"   - Ages shape: {ages.shape}")
            print(f"   - Genders shape: {genders.shape}")
            print(f"   - Age classes: {metadata['num_age_classes']}")
            print(f"   - Gender classes: {metadata['num_gender_classes']}")
            
            return images, ages, genders, metadata
            
        except Exception as e:
            print(f"Error loading data: {e}")
            print("Make sure to run preprocessing.py first!")
            return None, None, None, None
    
    def split_data(self, images, ages, genders, test_size=0.2, val_size=0.2):
        """Split data into train, validation, and test sets"""
        print(f"Splitting data...")
        print(f"   - Test size: {test_size}")
        print(f"   - Validation size: {val_size}")
        
        # First split: separate test set
        X_temp, X_test, y_age_temp, y_age_test, y_gender_temp, y_gender_test = train_test_split(
            images, ages, genders, 
            test_size=test_size, 
            random_state=42, 
            stratify=genders
        )
        
        # Second split: separate train and validation
        val_size_adjusted = val_size / (1 - test_size)  # Adjust for remaining data
        X_train, X_val, y_age_train, y_age_val, y_gender_train, y_gender_val = train_test_split(
            X_temp, y_age_temp, y_gender_temp,
            test_size=val_size_adjusted,
            random_state=42,
            stratify=y_gender_temp
        )
        
        print(f"Data split complete:")
        print(f"   - Training: {len(X_train)} samples ({len(X_train)/len(images)*100:.1f}%)")
        print(f"   - Validation: {len(X_val)} samples ({len(X_val)/len(images)*100:.1f}%)")
        print(f"   - Test: {len(X_test)} samples ({len(X_test)/len(images)*100:.1f}%)")
        
        return (X_train, X_val, X_test, 
                y_age_train, y_age_val, y_age_test,
                y_gender_train, y_gender_val, y_gender_test)
    
    def create_model(self, input_shape, use_pretrained=True):
        """Create and compile model"""
        print("Creating model...")
        
        self.mt_cnn = MultiTaskCNN(input_shape=input_shape)
        self.model = self.mt_cnn.create_model(use_pretrained=use_pretrained)
        self.model = self.mt_cnn.compile_model(self.model, learning_rate=0.0001)
        
        return self.model
    
    def train_model(self, X_train, X_val, y_age_train, y_age_val, 
               y_gender_train, y_gender_val, epochs=50, batch_size=128):
        """Train the model - let Keras calculate steps automatically"""
        print(f"Starting training...")
        print(f"   - Epochs: {epochs}")
        print(f"   - Batch size: {batch_size}")
        print(f"   - Training samples: {len(X_train)}")
        print(f"   - Validation samples: {len(X_val)}")
    
        # Enable mixed precision for faster training (optional)
        tf.keras.mixed_precision.set_global_policy('mixed_float16')

        # Prepare training data
        y_train = {
            'age_output': y_age_train,
            'gender_output': y_gender_train
        }

        y_val = {
          'age_output': y_age_val,
          'gender_output': y_gender_val
        }

        # Get callbacks
        model_name = os.path.join(self.models_dir, f'best_model_{datetime.now().strftime("%Y%m%d_%H%M%S")}.keras')
        callbacks = self.mt_cnn.create_callbacks(model_name=model_name, patience=20)

        # Train model - NO steps_per_epoch or validation_steps
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1,
            shuffle=True
       )

        return self.history

    def evaluate_model(self, X_test, y_age_test, y_gender_test):
        """Evaluate model on test set"""
        print("Evaluating model...")
        
        y_test = {
            'age_output': y_age_test,
            'gender_output': y_gender_test
        }
        
        # Evaluate
        test_results = self.model.evaluate(X_test, y_test, verbose=0)
        
        # Get predictions
        predictions = self.model.predict(X_test, verbose=0)
        age_pred, gender_pred = predictions
        
        # Calculate accuracies
        age_accuracy = np.mean(np.argmax(age_pred, axis=1) == y_age_test)
        gender_accuracy = np.mean(np.argmax(gender_pred, axis=1) == y_gender_test)
        
        print(f"Test Results:")
        print(f"   - Age Accuracy: {age_accuracy:.4f} ({age_accuracy*100:.2f}%)")
        print(f"   - Gender Accuracy: {gender_accuracy:.4f} ({gender_accuracy*100:.2f}%)")
        print(f"   - Overall Loss: {test_results[0]:.4f}")
        
        return {
            'age_accuracy': age_accuracy,
            'gender_accuracy': gender_accuracy,
            'overall_loss': test_results[0],
            'detailed_results': test_results
        }
    
    def save_training_results(self, test_results):
        """Save training results and metadata"""
        print(f"Saving training results...")
        
        # Save history
        if self.history:
            history_dict = {key: [float(val) for val in values] 
                           for key, values in self.history.history.items()}
            
            with open(os.path.join(self.models_dir, 'training_history.json'), 'w') as f:
                json.dump(history_dict, f, indent=2)
        
        # Save test results
        with open(os.path.join(self.models_dir, 'test_results.json'), 'w') as f:
            json.dump(test_results, f, indent=2)
        
        print(f"Training results saved to {self.models_dir}/")

def main():
    """Main training function"""
    print("Starting Age & Gender Detection Model Training")
    print("=" * 70)
    
    # Initialize trainer
    trainer = ModelTrainer()

    #configure GPU
    configure_gpu()
    
    # Load data
    images, ages, genders, metadata = trainer.load_data()
    if images is None:
        return
    
    # Split data
    data_splits = trainer.split_data(images, ages, genders)
    (X_train, X_val, X_test, 
     y_age_train, y_age_val, y_age_test,
     y_gender_train, y_gender_val, y_gender_test) = data_splits
    
    # Create model
    model = trainer.create_model(
        input_shape=images.shape[1:],
        use_pretrained=True
    )
    
    # Train model
    history = trainer.train_model(
        X_train, X_val,
        y_age_train, y_age_val,
        y_gender_train, y_gender_val,
        epochs=50,
        batch_size=128
    )
    
    # Evaluate model
    test_results = trainer.evaluate_model(X_test, y_age_test, y_gender_test)
    
    # Save results
    trainer.save_training_results(test_results)
    
    print("\nTraining completed successfully!")
    print("Next step: Run streamlit_app.py to test the model")

if __name__ == "__main__":
    main()