import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import VGG16
import numpy as np

class MultiTaskCNN:
    def __init__(self, input_shape=(128, 128, 3), num_age_classes=5, num_gender_classes=2):
        self.input_shape = input_shape
        self.num_age_classes = num_age_classes
        self.num_gender_classes = num_gender_classes
        
    def create_model(self, use_pretrained=True, architecture='vgg16'):
        """Create multi-task CNN for age and gender prediction"""
        print(f"Creating multi-task model...")
        print(f"   - Input shape: {self.input_shape}")
        print(f"   - Age classes: {self.num_age_classes}")
        print(f"   - Gender classes: {self.num_gender_classes}")
        print(f"   - Using pretrained: {use_pretrained}")
        
        # Input layer
        inputs = keras.Input(shape=self.input_shape, name='input_image')
        
        if use_pretrained and architecture == 'vgg16':
            # Use VGG16 as backbone (pre-trained on ImageNet)
            backbone = VGG16(
                weights='imagenet',
                include_top=False,
                input_tensor=inputs,
                pooling=None
            )
            
            # Freeze early layers, fine-tune later layers
            for layer in backbone.layers[:-6]:
                layer.trainable = False
            
            print(f"   - Using VGG16 backbone with {sum([1 for layer in backbone.layers if layer.trainable])} trainable layers")
            x = backbone.output
            
        else:
            # Custom CNN architecture
            print("   - Using custom CNN architecture")
            x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
            x = layers.BatchNormalization()(x)
            x = layers.MaxPooling2D((2, 2))(x)
            x = layers.Dropout(0.25)(x)
            
            x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
            x = layers.BatchNormalization()(x)
            x = layers.MaxPooling2D((2, 2))(x)
            x = layers.Dropout(0.25)(x)
            
            x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
            x = layers.BatchNormalization()(x)
            x = layers.MaxPooling2D((2, 2))(x)
            x = layers.Dropout(0.25)(x)
            
            x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
            x = layers.BatchNormalization()(x)
            x = layers.MaxPooling2D((2, 2))(x)
            x = layers.Dropout(0.25)(x)
        
        # Global Average Pooling
        x = layers.GlobalAveragePooling2D(name='global_avg_pool')(x)
        x = layers.Dropout(0.5)(x)
        
        # Shared dense layers
        x = layers.Dense(512, activation='relu', name='shared_dense_1')(x)
        x = layers.BatchNormalization(name='shared_bn_1')(x)
        x = layers.Dropout(0.4)(x)
        
        x = layers.Dense(256, activation='relu', name='shared_dense_2')(x)
        x = layers.BatchNormalization(name='shared_bn_2')(x)
        x = layers.Dropout(0.3)(x)
        
        # Age prediction branch
        age_branch = layers.Dense(128, activation='relu', name='age_dense_1')(x)
        age_branch = layers.BatchNormalization(name='age_bn_1')(age_branch)
        age_branch = layers.Dropout(0.3)(age_branch)
        
        age_branch = layers.Dense(64, activation='relu', name='age_dense_2')(age_branch)
        age_branch = layers.Dropout(0.2)(age_branch)
        
        age_output = layers.Dense(
            self.num_age_classes, 
            activation='softmax', 
            name='age_output'
        )(age_branch)
        
        # Gender prediction branch  
        gender_branch = layers.Dense(64, activation='relu', name='gender_dense_1')(x)
        gender_branch = layers.BatchNormalization(name='gender_bn_1')(gender_branch)
        gender_branch = layers.Dropout(0.3)(gender_branch)
        
        gender_branch = layers.Dense(32, activation='relu', name='gender_dense_2')(gender_branch)
        gender_branch = layers.Dropout(0.2)(gender_branch)
        
        gender_output = layers.Dense(
            self.num_gender_classes, 
            activation='softmax', 
            name='gender_output'
        )(gender_branch)
        
        # Create model
        model = Model(
            inputs=inputs,
            outputs=[age_output, gender_output],
            name='MultiTask_Age_Gender_CNN'
        )
        
        print(f"Model created successfully!")
        return model
    
    def compile_model(self, model, learning_rate=0.001, age_weight=1.0, gender_weight=1.0):
        """Compile the multi-task model with EXPLICIT multi-output configuration"""
        print(f"Compiling model...")
        print(f"   - Learning rate: {learning_rate}")
        print(f"   - Age weight: {age_weight}")
        print(f"   - Gender weight: {gender_weight}")
        
        # Define optimizer
        optimizer = keras.optimizers.Adam(
            learning_rate=learning_rate,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-07
        )
        
        losses = {
            'age_output': 'sparse_categorical_crossentropy',
            'gender_output': 'sparse_categorical_crossentropy'
        }
        
        metrics = {
            'age_output': ['accuracy'],
            'gender_output': ['accuracy']
        }
        
        # Loss weights for balancing tasks
        loss_weights = {
            'age_output': age_weight,
            'gender_output': gender_weight
        }
        
        model.compile(
            optimizer=optimizer,
            loss=losses,              # ← Explicit dict format
            loss_weights=loss_weights,
            metrics=metrics           # ← Explicit dict format
        )
        
        print("Model compiled with explicit multi-output configuration!")
        return model
    
    def create_callbacks(self, model_name='best_multitask_model.keras', patience=15):
        """Create training callbacks"""
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=patience,
                restore_best_weights=True,
                verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=patience//2,
                min_lr=1e-7,
                verbose=1
            ),
            keras.callbacks.ModelCheckpoint(
                model_name,
                monitor='val_loss',
                save_best_only=True,
                save_weights_only=False,
                verbose=1
            ),
            keras.callbacks.CSVLogger(
                'training_log.csv',
                append=True
            )
        ]
        
        return callbacks

def create_model_architecture(input_shape=(128, 128, 3), use_pretrained=True):
    """Convenience function to create model"""
    mt_cnn = MultiTaskCNN(input_shape=input_shape)
    model = mt_cnn.create_model(use_pretrained=use_pretrained)
    model = mt_cnn.compile_model(model)
    return model, mt_cnn

if __name__ == "__main__":
    # Test model creation
    print("Testing model creation...")
    model, mt_cnn = create_model_architecture()
    print("Model creation test completed!")