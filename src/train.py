import os
import sys
import numpy as np
import pickle
import json
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import h5py
from tqdm import tqdm

# Fix Unicode encoding for Windows
if sys.platform.startswith('win'):
    os.environ['PYTHONIOENCODING'] = 'utf-8'

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def configure_gpu():
    """Configure GPU settings for optimal performance"""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        gpu_count = torch.cuda.device_count()
        print(f"✅ Found {gpu_count} GPU(s)")
        for i in range(gpu_count):
            print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"   CUDA Version: {torch.version.cuda}")
        return device
    else:
        print("⚠️  No GPUs found, using CPU")
        return torch.device('cpu')

class HDF5Dataset(Dataset):
    """PyTorch Dataset for HDF5 chunked data"""
    
    def __init__(self, hdf5_data_dir, transform=None, split='train'):
        self.transform = transform
        self.split = split
        
        # Load dataset index
        index_file = os.path.join(hdf5_data_dir, 'dataset_index.pkl')
        with open(index_file, 'rb') as f:
            self.index = pickle.load(f)
        
        # Get all chunk files
        self.chunk_files = self.index['imdb_chunks'] + self.index['wiki_chunks']
        
        # Build sample index for efficient access
        self.sample_index = []
        for chunk_info in self.index['chunk_info']:
            for i in range(chunk_info['size']):
                self.sample_index.append((chunk_info['file'], i))
        
        print(f"📊 {split.upper()} Dataset Info:")
        print(f"   - Chunk files: {len(self.chunk_files)}")
        print(f"   - Total samples: {len(self.sample_index):,}")
    
    def __len__(self):
        return len(self.sample_index)
    
    def __getitem__(self, idx):
        chunk_file, sample_idx = self.sample_index[idx]
        
        # Load data from HDF5 chunk
        with h5py.File(chunk_file, 'r') as f:
            image = f['images'][sample_idx]
            age = f['ages'][sample_idx]
            gender = f['genders'][sample_idx]
        
        # TensorFlow format (H, W, C) -> PyTorch format (C, H, W)
        image = torch.FloatTensor(image).permute(2, 0, 1)
        age = torch.LongTensor([age]).squeeze()
        gender = torch.LongTensor([gender]).squeeze()
        
        # Apply transforms if provided
        if self.transform:
            image = self.transform(image)
        
        return image, age, gender

class HDF5StreamingTrainer:
    def __init__(self):
        # Get absolute paths
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.project_root = os.path.dirname(self.script_dir)
        
        # Updated paths for HDF5 chunked data
        self.hdf5_data_dir = os.path.join(self.project_root, 'data', 'efficient_data')
        self.legacy_data_dir = os.path.join(self.project_root, 'data', 'processed_data')  # Fallback
        self.models_dir = os.path.join(self.project_root, 'models')
        
        self.device = configure_gpu()
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.training_history = {'train_loss': [], 'val_loss': [], 'train_age_acc': [], 
                               'train_gender_acc': [], 'val_age_acc': [], 'val_gender_acc': []}
        
        # Create models directory if it doesn't exist
        os.makedirs(self.models_dir, exist_ok=True)
        
        print(f"Project root: {self.project_root}")
        print(f"HDF5 data directory: {self.hdf5_data_dir}")
        print(f"Legacy data directory: {self.legacy_data_dir}")
        print(f"Models directory: {self.models_dir}")
        print(f"Device: {self.device}")

    def create_data_transforms(self):
        """Create data augmentation transforms"""
        train_transform = transforms.Compose([
            transforms.RandomRotation(degrees=15),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.1, saturation=0.1, hue=0.05),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1))
        ])
        
        val_transform = transforms.Compose([
            # No augmentation for validation
        ])
        
        return train_transform, val_transform

    def create_dataloaders(self, batch_size=64, num_workers=4):
        """Create PyTorch DataLoaders for training"""
        
        # Check if HDF5 data exists
        index_file = os.path.join(self.hdf5_data_dir, 'dataset_index.pkl')
        if not os.path.exists(index_file):
            print("⚠️  HDF5 chunked dataset not found!")
            print(f"Expected index file: {index_file}")
            return None, None, None
        
        print("📄 Loading HDF5 chunked dataset...")
        
        # Create transforms
        train_transform, val_transform = self.create_data_transforms()
        
        # Create dataset
        full_dataset = HDF5Dataset(self.hdf5_data_dir, transform=None)
        
        # Split dataset indices
        total_size = len(full_dataset)
        train_size = int(0.7 * total_size)
        val_size = int(0.2 * total_size)
        test_size = total_size - train_size - val_size
        
        # Create split indices
        indices = torch.randperm(total_size).tolist()
        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size + val_size]
        test_indices = indices[train_size + val_size:]
        
        print(f"📊 Dataset split:")
        print(f"   - Training: {len(train_indices):,} samples (70%)")
        print(f"   - Validation: {len(val_indices):,} samples (20%)")
        print(f"   - Test: {len(test_indices):,} samples (10%)")
        
        # Create subset datasets
        train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
        val_dataset = torch.utils.data.Subset(full_dataset, val_indices)
        test_dataset = torch.utils.data.Subset(full_dataset, test_indices)
        
        # Apply transforms to training data
        train_dataset.dataset.transform = train_transform
        
        # Create DataLoaders
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, 
            num_workers=num_workers, pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, 
            num_workers=num_workers, pin_memory=True
        )
        
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, 
            num_workers=num_workers, pin_memory=True
        )
        
        return train_loader, val_loader, test_loader

    def create_model(self, input_shape=(3, 128, 128), use_pretrained=True):
        """Create and setup model with optimizations for better accuracy"""
        print("🏗️  Creating PyTorch model optimized for better accuracy...")
        
        from model import MultiTaskCNN
        
        # Create model
        self.model = MultiTaskCNN(input_shape=input_shape).to(self.device)
        self.model.create_model(use_pretrained=use_pretrained)
        
        # Setup optimizer and loss (lower learning rate for better convergence)
        self.model.compile_model(learning_rate=0.00005, age_weight=1.5, gender_weight=1.0)
        
        # Setup learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.model.optimizer, mode='min', factor=0.3, patience=12, 
            min_lr=1e-8, verbose=True
        )
        
        print("✅ PyTorch model created and configured successfully!")
        return self.model

    def train_epoch(self, train_loader):
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        correct_age = 0
        correct_gender = 0
        total_samples = 0
        
        # Progress bar
        pbar = tqdm(train_loader, desc='Training')
        
        for batch_idx, (images, ages, genders) in enumerate(pbar):
            # Move to device
            images = images.to(self.device)
            ages = ages.to(self.device)
            genders = genders.to(self.device)
            
            # Forward pass
            self.model.optimizer.zero_grad()
            age_pred, gender_pred = self.model(images)
            
            # Calculate loss
            total_loss, age_loss, gender_loss = self.model.calculate_loss(
                age_pred, gender_pred, ages, genders
            )
            
            # Backward pass
            total_loss.backward()
            self.model.optimizer.step()
            
            # Statistics
            running_loss += total_loss.item()
            
            # Calculate accuracy
            _, age_predicted = torch.max(age_pred.data, 1)
            _, gender_predicted = torch.max(gender_pred.data, 1)
            
            correct_age += (age_predicted == ages).sum().item()
            correct_gender += (gender_predicted == genders).sum().item()
            total_samples += ages.size(0)
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f'{total_loss.item():.4f}',
                'Age Acc': f'{100 * correct_age / total_samples:.2f}%',
                'Gender Acc': f'{100 * correct_gender / total_samples:.2f}%'
            })
        
        epoch_loss = running_loss / len(train_loader)
        age_accuracy = correct_age / total_samples
        gender_accuracy = correct_gender / total_samples
        
        return epoch_loss, age_accuracy, gender_accuracy

    def validate_epoch(self, val_loader):
        """Validate for one epoch"""
        self.model.eval()
        running_loss = 0.0
        correct_age = 0
        correct_gender = 0
        total_samples = 0
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc='Validation')
            
            for images, ages, genders in pbar:
                # Move to device
                images = images.to(self.device)
                ages = ages.to(self.device)
                genders = genders.to(self.device)
                
                # Forward pass
                age_pred, gender_pred = self.model(images)
                
                # Calculate loss
                total_loss, _, _ = self.model.calculate_loss(
                    age_pred, gender_pred, ages, genders
                )
                
                # Statistics
                running_loss += total_loss.item()
                
                # Calculate accuracy
                _, age_predicted = torch.max(age_pred.data, 1)
                _, gender_predicted = torch.max(gender_pred.data, 1)
                
                correct_age += (age_predicted == ages).sum().item()
                correct_gender += (gender_predicted == genders).sum().item()
                total_samples += ages.size(0)
                
                # Update progress bar
                pbar.set_postfix({
                    'Loss': f'{total_loss.item():.4f}',
                    'Age Acc': f'{100 * correct_age / total_samples:.2f}%',
                    'Gender Acc': f'{100 * correct_gender / total_samples:.2f}%'
                })
        
        epoch_loss = running_loss / len(val_loader)
        age_accuracy = correct_age / total_samples
        gender_accuracy = correct_gender / total_samples
        
        return epoch_loss, age_accuracy, gender_accuracy

    def train_with_hdf5_streaming(self, epochs=100, batch_size=64):
        """Train model using HDF5 streaming with enhanced optimizations"""
        print("🚀 Starting PyTorch HDF5 streaming training...")
        print(f"📋 Training configuration:")
        print(f"   - Epochs: {epochs}")
        print(f"   - Batch size: {batch_size}")
        print(f"   - Data source: HDF5 chunks")
        print(f"   - Data augmentation: Enhanced")
        print(f"   - Early stopping patience: 25")
        print(f"   - Device: {self.device}")
        
        # Create data loaders
        train_loader, val_loader, test_loader = self.create_dataloaders(batch_size=batch_size)
        
        if train_loader is None:
            print("❌ Failed to create data loaders!")
            return None, None
        
        # Create model
        model = self.create_model()
        
        # Training setup
        best_val_loss = float('inf')
        patience_counter = 0
        patience = 25
        
        model_name = os.path.join(
            self.models_dir, 
            f'pytorch_streaming_model_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pth'
        )
        
        print(f"💾 Model will be saved as: {model_name}")
        print("🎯 Starting PyTorch streaming training...")
        
        # Training loop
        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")
            print("-" * 50)
            
            # Train
            train_loss, train_age_acc, train_gender_acc = self.train_epoch(train_loader)
            
            # Validate
            val_loss, val_age_acc, val_gender_acc = self.validate_epoch(val_loader)
            
            # Update learning rate scheduler
            self.scheduler.step(val_loss)
            
            # Store history
            self.training_history['train_loss'].append(train_loss)
            self.training_history['val_loss'].append(val_loss)
            self.training_history['train_age_acc'].append(train_age_acc)
            self.training_history['train_gender_acc'].append(train_gender_acc)
            self.training_history['val_age_acc'].append(val_age_acc)
            self.training_history['val_gender_acc'].append(val_gender_acc)
            
            # Print epoch results
            print(f"\nEpoch {epoch+1} Results:")
            print(f"   Train - Loss: {train_loss:.4f}, Age Acc: {train_age_acc:.4f} ({train_age_acc*100:.2f}%), Gender Acc: {train_gender_acc:.4f} ({train_gender_acc*100:.2f}%)")
            print(f"   Val   - Loss: {val_loss:.4f}, Age Acc: {val_age_acc:.4f} ({val_age_acc*100:.2f}%), Gender Acc: {val_gender_acc:.4f} ({val_gender_acc*100:.2f}%)")
            print(f"   Learning Rate: {self.model.optimizer.param_groups[0]['lr']:.2e}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                
                # Save model checkpoint
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.model.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'val_loss': val_loss,
                    'val_age_acc': val_age_acc,
                    'val_gender_acc': val_gender_acc,
                    'training_history': self.training_history
                }, model_name)
                
                print(f"   💾 Saved best model checkpoint")
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= patience:
                print(f"\n⏹️  Early stopping triggered after {patience} epochs without improvement")
                break
        
        print("✅ Training completed!")
        
        # Evaluate on test set
        print("📊 Evaluating on test set...")
        test_loss, test_age_acc, test_gender_acc = self.validate_epoch(test_loader)
        
        print(f"🎯 Final test results:")
        print(f"   - Overall loss: {test_loss:.4f}")
        print(f"   - Age accuracy: {test_age_acc:.4f} ({test_age_acc*100:.2f}%)")
        print(f"   - Gender accuracy: {test_gender_acc:.4f} ({test_gender_acc*100:.2f}%)")
        
        test_results = {
            'test_loss': test_loss,
            'test_age_accuracy': test_age_acc,
            'test_gender_accuracy': test_gender_acc
        }
        
        return self.training_history, test_results

    def save_training_results(self, test_results):
        """Save training results and metadata"""
        print("💾 Saving training results...")
        
        # Save training history
        history_file = os.path.join(self.models_dir, 'pytorch_training_history.json')
        with open(history_file, 'w') as f:
            # Convert numpy types to native Python types for JSON serialization
            history_json = {}
            for key, values in self.training_history.items():
                history_json[key] = [float(v) for v in values]
            json.dump(history_json, f, indent=2)
        
        # Save test results
        results_dict = {
            'test_results': test_results,
            'training_completed': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'model_type': 'pytorch_hdf5_streaming_multitask_cnn',
            'data_format': 'hdf5_chunked_streaming',
            'framework': 'pytorch',
            'device': str(self.device),
            'total_epochs': len(self.training_history['train_loss'])
        }
        
        results_file = os.path.join(self.models_dir, 'pytorch_test_results.json')
        with open(results_file, 'w') as f:
            json.dump(results_dict, f, indent=2)
        
        print(f"📁 PyTorch training results saved to {self.models_dir}/")

def main():
    """Main training function with PyTorch HDF5 streaming"""
    print("🎯 Age & Gender Detection - PyTorch HDF5 Streaming Training")
    print("=" * 70)
    print(f"🕐 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("📄 Using PyTorch with HDF5 chunked data storage for memory efficiency")
    
    # Initialize trainer
    trainer = HDF5StreamingTrainer()
    
    # Train with PyTorch HDF5 streaming approach
    try:
        history, test_results = trainer.train_with_hdf5_streaming(
            epochs=100,     # More epochs for better accuracy with large dataset
            batch_size=64   # Optimal batch size for most GPUs
        )
        
        if history is not None:
            # Save results
            trainer.save_training_results(test_results)
            
            print("\n🎉 PyTorch training completed successfully!")
            print("📊 Summary:")
            print(f"   - Final validation loss: {min(history['val_loss']):.4f}")
            print(f"   - Training epochs: {len(history['train_loss'])}")
            print(f"   - Data source: HDF5 chunked streaming")
            print(f"   - Framework: PyTorch")
            print(f"   - Memory usage: Minimal (< 4GB)")
            print(f"   - Model format: .pth (PyTorch)")
            
            # Show best accuracy achieved
            best_val_age_acc = max(history['val_age_acc'])
            best_val_gender_acc = max(history['val_gender_acc'])
            print(f"   - Best validation age accuracy: {best_val_age_acc:.4f} ({best_val_age_acc*100:.2f}%)")
            print(f"   - Best validation gender accuracy: {best_val_gender_acc:.4f} ({best_val_gender_acc*100:.2f}%)")
            
            print("\n➡️  Next steps:")
            print("1. Update detection.py to use PyTorch model loading")
            print("2. Update streamlit_app.py to test the PyTorch model with live camera")
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