import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os
import json
from datetime import datetime

class MultiTaskCNN(nn.Module):
    def __init__(self, input_shape=(3, 128, 128), num_age_classes=5, num_gender_classes=2):
        super(MultiTaskCNN, self).__init__()
        
        # Note: PyTorch uses (C, H, W) format vs TensorFlow's (H, W, C)
        self.input_shape = input_shape
        self.num_age_classes = num_age_classes
        self.num_gender_classes = num_gender_classes
        
        # These will be set in create_model
        self.backbone = None
        self.shared_layers = None
        self.age_branch = None
        self.gender_branch = None
        self.use_pretrained = None
        
    def create_model(self, use_pretrained=True, architecture='vgg16'):
        """Create multi-task CNN for age and gender prediction"""
        print(f"Creating multi-task model...")
        print(f" - Input shape: {self.input_shape}")
        print(f" - Age classes: {self.num_age_classes}")
        print(f" - Gender classes: {self.num_gender_classes}")
        print(f" - Using pretrained: {use_pretrained}")
        
        self.use_pretrained = use_pretrained
        
        if use_pretrained and architecture == 'vgg16':
            # Use VGG16 as backbone (pre-trained on ImageNet)
            print(" - Using VGG16 backbone")
            vgg16 = models.vgg16(pretrained=True)
            
            # Remove the classifier (last layer)
            self.backbone = nn.Sequential(*list(vgg16.features))
            
            # Freeze early layers, fine-tune later layers
            trainable_layers = 0
            total_layers = len(list(self.backbone.parameters()))
            
            for i, param in enumerate(self.backbone.parameters()):
                if i < total_layers - 12:  # Freeze early layers
                    param.requires_grad = False
                else:
                    param.requires_grad = True
                    trainable_layers += 1
            
            print(f" - Using VGG16 backbone with {trainable_layers} trainable parameters")
            
            # VGG16 feature dimension after global average pooling
            feature_dim = 512  # VGG16 final conv layer outputs 512 channels
            
            # Add global average pooling
            self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
            self.flatten = nn.Flatten()
            
        else:
            # Custom CNN architecture
            print(" - Using custom CNN architecture")
            self.backbone = nn.Sequential(
                # First block
                nn.Conv2d(3, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Dropout(0.25),
                
                # Second block  
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Dropout(0.25),
                
                # Third block
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Dropout(0.25),
                
                # Fourth block
                nn.Conv2d(128, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Dropout(0.25),
                
                # Global Average Pooling
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Dropout(0.5)
            )
            
            feature_dim = 256
            self.global_avg_pool = None  # Already included in backbone
            self.flatten = None
        
        # Shared dense layers
        self.shared_layers = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Age prediction branch
        self.age_branch = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, self.num_age_classes)
        )
        
        # Gender prediction branch
        self.gender_branch = nn.Sequential(
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, self.num_gender_classes)
        )
        
        print(f"Model created successfully!")
        return self
    
    def forward(self, x):
        """Forward pass through the network"""
        # Extract features using backbone
        features = self.backbone(x)
        
        # For VGG16, apply additional pooling and flattening
        if self.use_pretrained and self.global_avg_pool is not None:
            features = self.global_avg_pool(features)
            features = self.flatten(features)
        
        # Pass through shared layers
        shared_features = self.shared_layers(features)
        
        # Multi-task outputs
        age_output = self.age_branch(shared_features)
        gender_output = self.gender_branch(shared_features)
        
        return age_output, gender_output
    
    def compile_model(self, learning_rate=0.001, age_weight=1.0, gender_weight=1.0):
        """Setup optimizer and loss functions (PyTorch equivalent of Keras compile)"""
        print(f"Setting up model training configuration...")
        print(f" - Learning rate: {learning_rate}")
        print(f" - Age weight: {age_weight}")
        print(f" - Gender weight: {gender_weight}")
        
        # Define optimizer
        self.optimizer = optim.Adam(
            self.parameters(),
            lr=learning_rate,
            betas=(0.9, 0.999),
            eps=1e-07
        )
        
        # Loss functions (CrossEntropyLoss includes softmax)
        self.age_criterion = nn.CrossEntropyLoss()
        self.gender_criterion = nn.CrossEntropyLoss()
        
        # Store loss weights for training
        self.age_weight = age_weight
        self.gender_weight = gender_weight
        
        print("Model compiled with explicit multi-output configuration!")
        return self
    
    def calculate_loss(self, age_pred, gender_pred, age_true, gender_true):
        """Calculate weighted multi-task loss"""
        age_loss = self.age_criterion(age_pred, age_true)
        gender_loss = self.gender_criterion(gender_pred, gender_true)
        
        # Apply weights
        total_loss = self.age_weight * age_loss + self.gender_weight * gender_loss
        
        return total_loss, age_loss, gender_loss
    
    def calculate_accuracy(self, predictions, targets):
        """Calculate accuracy for a single task"""
        _, predicted = torch.max(predictions.data, 1)
        total = targets.size(0)
        correct = (predicted == targets).sum().item()
        return correct / total

class PyTorchCallbacks:
    """PyTorch equivalent of Keras callbacks"""
    
    def __init__(self, model_name='best_multitask_model.pth', patience=15, monitor='val_loss'):
        self.model_name = model_name
        self.patience = patience
        self.monitor = monitor
        
        # Early stopping
        self.best_loss = float('inf')
        self.patience_counter = 0
        self.best_model_state = None
        
        # Learning rate reduction
        self.lr_patience = patience // 2
        self.lr_patience_counter = 0
        self.factor = 0.5
        self.min_lr = 1e-7
        
        # CSV logging
        self.training_log = []
        
        print(f"Initialized callbacks:")
        print(f" - Model checkpoint: {model_name}")
        print(f" - Early stopping patience: {patience}")
        print(f" - LR reduction patience: {self.lr_patience}")
    
    def on_epoch_end(self, model, epoch, train_loss, val_loss, train_age_acc, train_gender_acc, 
                     val_age_acc, val_gender_acc):
        """Called at the end of each epoch"""
        
        # Log metrics
        log_entry = {
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'train_age_acc': train_age_acc,
            'train_gender_acc': train_gender_acc,
            'val_age_acc': val_age_acc,
            'val_gender_acc': val_gender_acc,
            'lr': model.optimizer.param_groups[0]['lr']
        }
        self.training_log.append(log_entry)
        
        # Early stopping logic
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.patience_counter = 0
            self.lr_patience_counter = 0
            
            # Save best model
            self.best_model_state = model.state_dict().copy()
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': model.optimizer.state_dict(),
                'epoch': epoch,
                'val_loss': val_loss,
                'val_age_acc': val_age_acc,
                'val_gender_acc': val_gender_acc
            }, self.model_name)
            
            print(f"📁 Saved best model: {self.model_name}")
        else:
            self.patience_counter += 1
            self.lr_patience_counter += 1
        
        # Learning rate reduction
        if self.lr_patience_counter >= self.lr_patience:
            current_lr = model.optimizer.param_groups[0]['lr']
            new_lr = max(current_lr * self.factor, self.min_lr)
            
            if new_lr < current_lr:
                for param_group in model.optimizer.param_groups:
                    param_group['lr'] = new_lr
                print(f"📉 Reduced learning rate to {new_lr:.2e}")
                self.lr_patience_counter = 0
        
        # Check early stopping
        should_stop = self.patience_counter >= self.patience
        
        return should_stop
    
    def restore_best_weights(self, model):
        """Restore the best model weights"""
        if self.best_model_state is not None:
            model.load_state_dict(self.best_model_state)
            print("🔄 Restored best model weights")
    
    def save_training_log(self, log_file='training_log.csv'):
        """Save training log to CSV"""
        if self.training_log:
            import pandas as pd
            df = pd.DataFrame(self.training_log)
            df.to_csv(log_file, index=False)
            print(f"📊 Training log saved to {log_file}")

def create_model_architecture(input_shape=(3, 128, 128), use_pretrained=True):
    """Convenience function to create model (PyTorch equivalent of original function)"""
    mt_cnn = MultiTaskCNN(input_shape=input_shape)
    model = mt_cnn.create_model(use_pretrained=use_pretrained)
    model = mt_cnn.compile_model()
    return model, mt_cnn

# Device configuration
def get_device():
    """Get the best available device"""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"🔥 Using GPU: {torch.cuda.get_device_name()}")
    else:
        device = torch.device('cpu')
        print("💻 Using CPU")
    return device

if __name__ == "__main__":
    # Test model creation
    print("Testing PyTorch model creation...")
    
    device = get_device()
    model, mt_cnn = create_model_architecture()
    model = model.to(device)
    
    # Test forward pass
    batch_size = 4
    test_input = torch.randn(batch_size, 3, 128, 128).to(device)
    
    model.eval()
    with torch.no_grad():
        age_pred, gender_pred = model(test_input)
        print(f"✅ Model test successful!")
        print(f"   Age predictions shape: {age_pred.shape}")
        print(f"   Gender predictions shape: {gender_pred.shape}")
    
    print("PyTorch model creation test completed!")