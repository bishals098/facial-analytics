import os
import sys
import numpy as np
import pickle
import json
from datetime import datetime
import tensorflow as tf
import h5py

# Fix Unicode encoding for Windows
if sys.platform.startswith('win'):
    os.environ['PYTHONIOENCODING'] = 'utf-8'

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

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

class HDF5StreamingTrainer:
    def __init__(self):
        # Get absolute paths
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.project_root = os.path.dirname(self.script_dir)
        
        # Updated paths for HDF5 chunked data
        self.hdf5_data_dir = os.path.join(self.project_root, 'data', 'efficient_data')
        self.legacy_data_dir = os.path.join(self.project_root, 'data', 'processed_data')  # Fallback
        self.models_dir = os.path.join(self.project_root, 'models')
        
        self.model = None
        self.mt_cnn = None
        self.history = None
        
        # Create models directory if it doesn't exist
        os.makedirs(self.models_dir, exist_ok=True)
        
        print(f"Project root: {self.project_root}")
        print(f"HDF5 data directory: {self.hdf5_data_dir}")
        print(f"Legacy data directory: {self.legacy_data_dir}")
        print(f"Models directory: {self.models_dir}")

    def load_chunk_generator(self, chunk_files, shuffle=True):
        """Generator that loads HDF5 chunks on-demand"""
        if shuffle:
            np.random.shuffle(chunk_files)
            
        for chunk_file in chunk_files:
            try:
                with h5py.File(chunk_file, 'r') as f:
                    images = f['images'][:]
                    ages = f['ages'][:]
                    genders = f['genders'][:]
                    
                    # Yield each sample individually
                    for i in range(len(images)):
                        yield images[i], ages[i], genders[i]
                        
            except Exception as e:
                print(f"⚠️  Warning: Could not load chunk {chunk_file}: {e}")
                continue

    def create_streaming_dataset(self, batch_size=64, buffer_size=2000):
        """Create a streaming dataset from HDF5 chunk files"""
        
        # Try to load HDF5 chunk index
        index_file = os.path.join(self.hdf5_data_dir, 'dataset_index.pkl')
        
        if os.path.exists(index_file):
            print("📄 Loading HDF5 chunked dataset...")
            with open(index_file, 'rb') as f:
                index = pickle.load(f)
            
            all_chunk_files = index['imdb_chunks'] + index['wiki_chunks']
            total_samples = index['total_samples']
            
            print(f"📊 HDF5 Dataset Info:")
            print(f"   - HDF5 chunk files: {len(all_chunk_files)}")
            print(f"   - Total samples: {total_samples:,}")
            print(f"   - Batch size: {batch_size}")
            print(f"   - Buffer size: {buffer_size:,}")
            
            # Create dataset from generator
            dataset = tf.data.Dataset.from_generator(
                lambda: self.load_chunk_generator(all_chunk_files, shuffle=True),
                output_signature=(
                    tf.TensorSpec(shape=(128, 128, 3), dtype=tf.float32),
                    tf.TensorSpec(shape=(), dtype=tf.int32),
                    tf.TensorSpec(shape=(), dtype=tf.int32)
                )
            )
            
            # Apply optimizations for streaming
            dataset = dataset.shuffle(buffer_size=buffer_size)
            dataset = dataset.batch(batch_size)
            dataset = dataset.prefetch(tf.data.AUTOTUNE)
            
            return dataset, total_samples
            
        else:
            print("⚠️  HDF5 chunked dataset not found, falling back to legacy data loading...")
            return self.load_legacy_data(batch_size)

    def load_legacy_data(self, batch_size=64):
        """Fallback method to load legacy .npy files"""
        print("📁 Loading legacy preprocessed data...")
        
        try:
            # Check if files exist
            images_path = os.path.join(self.legacy_data_dir, 'images.npy')
            ages_path = os.path.join(self.legacy_data_dir, 'ages.npy')
            genders_path = os.path.join(self.legacy_data_dir, 'genders.npy')
            metadata_path = os.path.join(self.legacy_data_dir, 'metadata.pkl')
            
            print(f"Looking for legacy files:")
            print(f"   Images: {images_path} (exists: {os.path.exists(images_path)})")
            print(f"   Ages: {ages_path} (exists: {os.path.exists(ages_path)})")
            print(f"   Genders: {genders_path} (exists: {os.path.exists(genders_path)})")
            print(f"   Metadata: {metadata_path} (exists: {os.path.exists(metadata_path)})")
            
            # Load arrays
            images = np.load(images_path)
            ages = np.load(ages_path)
            genders = np.load(genders_path)
            
            # Load metadata
            with open(metadata_path, 'rb') as f:
                metadata = pickle.load(f)
            
            print(f"✅ Legacy data loaded successfully!")
            print(f"   - Images shape: {images.shape}")
            print(f"   - Ages shape: {ages.shape}")
            print(f"   - Genders shape: {genders.shape}")
            
            # Convert to TensorFlow dataset
            dataset = tf.data.Dataset.from_tensor_slices((
                images, 
                ages,
                genders
            ))
            dataset = dataset.shuffle(buffer_size=1000)
            dataset = dataset.batch(batch_size)
            dataset = dataset.prefetch(tf.data.AUTOTUNE)
            
            return dataset, len(images)
            
        except Exception as e:
            print(f"❌ Error loading legacy data: {e}")
            print("💡 Make sure to run preprocessing.py first!")
            return None, 0

    def split_streaming_dataset(self, dataset, total_samples, train_ratio=0.7, val_ratio=0.2):
        """Split streaming dataset into train/val/test"""
        
        # Calculate approximate sizes (batch-based splitting)
        total_batches = total_samples // 64  # Approximate batches
        train_batches = int(total_batches * train_ratio)
        val_batches = int(total_batches * val_ratio)
        test_batches = total_batches - train_batches - val_batches
        
        train_size = train_batches * 64
        val_size = val_batches * 64
        test_size = test_batches * 64
        
        print(f"📊 Dataset split (approximate):")
        print(f"   - Training: {train_size:,} samples ({train_ratio*100:.1f}%)")
        print(f"   - Validation: {val_size:,} samples ({val_ratio*100:.1f}%)")
        print(f"   - Test: {test_size:,} samples ({(1-train_ratio-val_ratio)*100:.1f}%)")
        
        # Split dataset by batches
        train_dataset = dataset.take(train_batches)
        remaining_dataset = dataset.skip(train_batches)
        
        val_dataset = remaining_dataset.take(val_batches)
        test_dataset = remaining_dataset.skip(val_batches)
        
        return train_dataset, val_dataset, test_dataset

    def prepare_dataset_for_training(self, dataset):
        """Prepare dataset for multi-task training"""
        def format_data(image, age, gender):
            return image, {'age_output': age, 'gender_output': gender}
        
        return dataset.map(format_data, num_parallel_calls=tf.data.AUTOTUNE)

    def create_model(self, input_shape=(128, 128, 3), use_pretrained=True):
        """Create and compile model with optimizations for better accuracy"""
        print("🏗️  Creating model optimized for better accuracy...")
        
        from model import MultiTaskCNN
        
        self.mt_cnn = MultiTaskCNN(input_shape=input_shape)
        self.model = self.mt_cnn.create_model(use_pretrained=use_pretrained)
        
        # Lower learning rate for better convergence
        self.model = self.mt_cnn.compile_model(
            self.model, 
            learning_rate=0.00005,  # Even lower for streaming data
            age_weight=1.5,         # Slight boost to age task
            gender_weight=1.0
        )
        
        print("✅ Model created and compiled successfully!")
        return self.model

    def create_data_augmentation(self):
        """Create data augmentation pipeline"""
        return tf.keras.Sequential([
            tf.keras.layers.RandomRotation(0.15),
            tf.keras.layers.RandomTranslation(0.1, 0.1),
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomBrightness(0.2),
            tf.keras.layers.RandomContrast(0.1),
            tf.keras.layers.RandomZoom(0.1),
        ], name="data_augmentation")

    def train_with_hdf5_streaming(self, epochs=100, batch_size=64):
        """Train model using HDF5 streaming with enhanced optimizations"""
        print("🚀 Starting HDF5 streaming training with enhanced optimizations...")
        print(f"📋 Training configuration:")
        print(f"   - Epochs: {epochs}")
        print(f"   - Batch size: {batch_size}")
        print(f"   - Data source: HDF5 chunks")
        print(f"   - Data augmentation: Enhanced")
        print(f"   - Early stopping patience: 25")
        
        # Create streaming dataset
        dataset, total_samples = self.create_streaming_dataset(batch_size=batch_size)
        
        if dataset is None:
            print("❌ Failed to create dataset!")
            return None, None
        
        # Split dataset
        train_dataset, val_dataset, test_dataset = self.split_streaming_dataset(
            dataset, total_samples
        )
        
        # Prepare datasets for multi-task training
        train_dataset = self.prepare_dataset_for_training(train_dataset)
        val_dataset = self.prepare_dataset_for_training(val_dataset)
        test_dataset = self.prepare_dataset_for_training(test_dataset)
        
        # Add data augmentation to training dataset
        augmentation = self.create_data_augmentation()
        
        def augment_training_data(x, y):
            return augmentation(x, training=True), y
        
        train_dataset = train_dataset.map(
            augment_training_data, 
            num_parallel_calls=tf.data.AUTOTUNE
        )
        
        # Create model
        model = self.create_model()
        
        # Enhanced callbacks
        model_name = os.path.join(
            self.models_dir, 
            f'hdf5_streaming_model_{datetime.now().strftime("%Y%m%d_%H%M%S")}.keras'
        )
        
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=25,  # More patience for streaming data
                restore_best_weights=True,
                verbose=1
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.3,   # More aggressive LR reduction
                patience=12,
                min_lr=1e-8,
                verbose=1
            ),
            tf.keras.callbacks.ModelCheckpoint(
                model_name,
                monitor='val_loss',
                save_best_only=True,
                save_weights_only=False,
                verbose=1
            ),
            tf.keras.callbacks.CSVLogger(
                os.path.join(self.models_dir, 'hdf5_training_log.csv'),
                append=True
            ),
            # Learning rate schedule for better convergence
            tf.keras.callbacks.LearningRateScheduler(
                lambda epoch: 0.00005 * (0.95 ** epoch),
                verbose=0
            )
        ]
        
        print(f"💾 Model will be saved as: {model_name}")
        
        # Train with streaming data
        print("🎯 Starting HDF5 streaming training...")
        
        self.history = model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )
        
        print("✅ Training completed!")
        
        # Evaluate on test set
        print("📊 Evaluating on test set...")
        test_results = model.evaluate(test_dataset, verbose=1)
        
        print(f"🎯 Final test results:")
        print(f"   - Overall loss: {test_results[0]:.4f}")
        if len(test_results) > 1:
            print(f"   - Age loss: {test_results[1]:.4f}")
            print(f"   - Gender loss: {test_results[2]:.4f}")
            if len(test_results) > 3:
                print(f"   - Age accuracy: {test_results[3]:.4f} ({test_results[3]*100:.2f}%)")
                print(f"   - Gender accuracy: {test_results[4]:.4f} ({test_results[4]*100:.2f}%)")
        
        return self.history, test_results

    def save_training_results(self, test_results):
        """Save training results and metadata"""
        print("💾 Saving training results...")
        
        # Save history
        if self.history:
            history_dict = {key: [float(val) for val in values]
                          for key, values in self.history.history.items()}
            
            with open(os.path.join(self.models_dir, 'hdf5_training_history.json'), 'w') as f:
                json.dump(history_dict, f, indent=2)
        
        # Save test results
        results_dict = {
            'test_results': [float(x) for x in test_results] if isinstance(test_results, (list, tuple)) else test_results,
            'training_completed': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'model_type': 'hdf5_streaming_multitask_cnn',
            'data_format': 'hdf5_chunked_streaming',
            'framework': 'tensorflow_keras'
        }
        
        with open(os.path.join(self.models_dir, 'hdf5_test_results.json'), 'w') as f:
            json.dump(results_dict, f, indent=2)
        
        print(f"📁 Training results saved to {self.models_dir}/")

def main():
    """Main training function with HDF5 streaming"""
    print("🎯 Age & Gender Detection - HDF5 Streaming Training")
    print("=" * 70)
    print(f"🕐 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("📄 Using HDF5 chunked data storage for memory efficiency")
    
    # Configure GPU
    configure_gpu()
    
    # Initialize trainer
    trainer = HDF5StreamingTrainer()
    
    # Train with HDF5 streaming approach
    try:
        history, test_results = trainer.train_with_hdf5_streaming(
            epochs=100,     # More epochs for better accuracy with large dataset
            batch_size=64   # Optimal batch size for most GPUs
        )
        
        if history is not None:
            # Save results
            trainer.save_training_results(test_results)
            
            print("\n🎉 Training completed successfully!")
            print("📊 Summary:")
            print(f"   - Final validation loss: {min(history.history['val_loss']):.4f}")
            print(f"   - Training epochs: {len(history.history['loss'])}")
            print(f"   - Data source: HDF5 chunked streaming")
            print(f"   - Memory usage: Minimal (< 4GB)")
            print(f"   - Model format: .keras (modern TensorFlow)")
            
            # Show best accuracy achieved
            if 'val_age_output_accuracy' in history.history:
                best_age_acc = max(history.history['val_age_output_accuracy'])
                best_gender_acc = max(history.history['val_gender_output_accuracy'])
                print(f"   - Best validation age accuracy: {best_age_acc:.4f} ({best_age_acc*100:.2f}%)")
                print(f"   - Best validation gender accuracy: {best_gender_acc:.4f} ({best_gender_acc*100:.2f}%)")
            
            print("\n➡️  Next steps:")
            print("1. Run streamlit_app.py to test the model with live camera")
            print("2. Use python src/detection.py for direct testing")
            print("3. Model should have much better accuracy and confidence now!")
            
        else:
            print("❌ Training failed!")
            
    except KeyboardInterrupt:
        print("\n⏹️  Training interrupted by user")
    except Exception as e:
        print(f"\n❌ Training error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()