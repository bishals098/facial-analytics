import os
import sys
import cv2
import numpy as np
import pandas as pd
from scipy.io import loadmat
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pickle

# Fix Unicode encoding for Windows
if sys.platform.startswith('win'):
    os.environ['PYTHONIOENCODING'] = 'utf-8'

class IMDBWIKIPreprocessor:
    def __init__(self):
        # Get the project root directory (parent of src/)
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.project_root = os.path.dirname(self.script_dir)
        self.data_root = os.path.join(self.project_root, 'data')
        
        self.imdb_path = os.path.join(self.data_root, 'imdb_crop')
        self.wiki_path = os.path.join(self.data_root, 'wiki_crop')
        self.processed_data_path = os.path.join(self.data_root, 'processed_data')
        self.target_size = (128, 128)
        
        print(f"Project root: {self.project_root}")
        print(f"Data root: {self.data_root}")
        print(f"IMDB path: {self.imdb_path}")
        print(f"WIKI path: {self.wiki_path}")
        
    def load_metadata(self, mat_file_path):
        """Load and process metadata from .mat files"""
        try:
            print(f"Loading metadata from {mat_file_path}...")
            mat_data = loadmat(mat_file_path)
            
            # Extract metadata from nested structure
            if 'imdb' in mat_data:
                data = mat_data['imdb'][0, 0]
                print("Loaded IMDB metadata")
            elif 'wiki' in mat_data:
                data = mat_data['wiki'][0, 0]
                print("Loaded WIKI metadata")
            else:
                print("Unknown metadata format")
                return None
            
            # Extract relevant fields
            metadata = {
                'dob': data['dob'][0],
                'photo_taken': data['photo_taken'][0],
                'full_path': data['full_path'][0],
                'gender': data['gender'][0],
                'name': data['name'][0],
                'face_location': data['face_location'][0] if 'face_location' in data.dtype.names else None
            }
            
            print(f"Loaded {len(metadata['dob'])} records")
            return metadata
            
        except Exception as e:
            print(f"Error loading metadata from {mat_file_path}: {e}")
            return None
    
    def calculate_age(self, dob, photo_taken):
        """Calculate age from date of birth and photo taken year"""
        try:
            # Convert MATLAB datenum to years
            birth_year = 1 + (dob - 1) / 365.25
            age = photo_taken - birth_year
            return int(age) if 0 <= age <= 100 else None
        except:
            return None
    
    def create_age_groups(self, age):
        """Convert age to age groups for better accuracy"""
        if age is None:
            return None
        elif 0 <= age <= 12:
            return 0  # Child
        elif 13 <= age <= 19:
            return 1  # Teen
        elif 20 <= age <= 35:
            return 2  # Young Adult
        elif 36 <= age <= 55:
            return 3  # Adult
        elif 56 <= age <= 100:
            return 4  # Senior
        else:
            return None
    
    def preprocess_image(self, img_path):
        """Preprocess individual image"""
        try:
            img = cv2.imread(img_path)
            if img is None:
                return None
            
            # Resize image
            img_resized = cv2.resize(img, self.target_size)
            
            # Normalize pixel values
            img_normalized = img_resized.astype(np.float32) / 255.0
            
            return img_normalized
        except Exception as e:
            return None
    
    def process_dataset(self, dataset_path, mat_file, max_images=25000):
        """Process entire dataset"""
        print(f"\n{'='*50}")
        print(f"Processing {dataset_path}...")
        print(f"{'='*50}")
        
        # Check if dataset exists
        print(f"Checking if dataset exists: {dataset_path}")
        if not os.path.exists(dataset_path):
            print(f"Dataset path not found: {dataset_path}")
            return None, None, None
            
        # Load metadata
        mat_file_full_path = os.path.join(dataset_path, mat_file)
        print(f"Looking for metadata file: {mat_file_full_path}")
        if not os.path.exists(mat_file_full_path):
            print(f"Metadata file not found: {mat_file_full_path}")
            return None, None, None
            
        metadata = self.load_metadata(mat_file_full_path)
        if metadata is None:
            return None, None, None
        
        images = []
        ages = []
        genders = []
        valid_count = 0
        invalid_count = 0
        
        total_records = len(metadata['dob'])
        print(f"Processing up to {max_images} images from {total_records} records...")
        
        for i in range(min(total_records, max_images * 2)):  # Process more to get enough valid images
            if valid_count >= max_images:
                break
                
            try:
                dob = metadata['dob'][i]
                photo_taken = metadata['photo_taken'][i]
                path = metadata['full_path'][i]
                gender = metadata['gender'][i]
                
                # Skip if path is not a string
                if not isinstance(path, str):
                    if hasattr(path, '__len__') and len(path) > 0:
                        path = path[0]
                    else:
                        invalid_count += 1
                        continue
                
                # Calculate age
                age = self.calculate_age(dob, photo_taken)
                age_group = self.create_age_groups(age)
                
                # Skip invalid data
                if age_group is None or np.isnan(gender):
                    invalid_count += 1
                    continue
                
                # Process image - construct full path
                img_path = os.path.join(dataset_path, path)
                processed_img = self.preprocess_image(img_path)
                
                if processed_img is not None:
                    images.append(processed_img)
                    ages.append(age_group)
                    genders.append(int(gender))  # 0: female, 1: male
                    valid_count += 1
                    
                    if valid_count % 1000 == 0:
                        print(f"Processed {valid_count} valid images, skipped {invalid_count} invalid")
                else:
                    invalid_count += 1
                    
            except Exception as e:
                invalid_count += 1
                continue
        
        print(f"\nDataset Processing Summary:")
        print(f"Valid images processed: {len(images)}")
        print(f"Invalid/skipped images: {invalid_count}")
        if len(images) + invalid_count > 0:
            print(f"Success rate: {len(images)/(len(images)+invalid_count)*100:.1f}%")
        
        if len(images) == 0:
            print("No valid images found!")
            return None, None, None
            
        return np.array(images), np.array(ages), np.array(genders)
    
    def balance_dataset(self, images, ages, genders):
        """Balance gender and age distribution"""
        print("\nBalancing dataset...")
        
        df = pd.DataFrame({
            'age': ages,
            'gender': genders,
            'index': range(len(images))
        })
        
        print("Original distribution:")
        print("Gender distribution:", df['gender'].value_counts().sort_index())
        print("Age distribution:", df['age'].value_counts().sort_index())
        
        # Balance genders first
        min_gender_count = df['gender'].value_counts().min()
        print(f"\nBalancing to {min_gender_count} samples per gender...")
        
        balanced_indices = []
        for gender in [0, 1]:  # female, male
            gender_data = df[df['gender'] == gender]
            if len(gender_data) >= min_gender_count:
                selected_indices = np.random.choice(
                    gender_data['index'].values, 
                    size=min_gender_count, 
                    replace=False
                )
                balanced_indices.extend(selected_indices)
        
        balanced_indices = np.array(balanced_indices)
        np.random.shuffle(balanced_indices)
        
        # Apply balancing
        balanced_images = images[balanced_indices]
        balanced_ages = ages[balanced_indices]
        balanced_genders = genders[balanced_indices]
        
        # Show final distribution
        print("\nFinal balanced distribution:")
        print("Gender distribution:", np.bincount(balanced_genders))
        print("Age distribution:", np.bincount(balanced_ages))
        
        return balanced_images, balanced_ages, balanced_genders
    
    def save_processed_data(self, images, ages, genders):
        """Save processed data"""
        os.makedirs(self.processed_data_path, exist_ok=True)
        
        print(f"\nSaving processed data to {self.processed_data_path}/...")
        
        # Save arrays
        np.save(os.path.join(self.processed_data_path, 'images.npy'), images)
        np.save(os.path.join(self.processed_data_path, 'ages.npy'), ages)
        np.save(os.path.join(self.processed_data_path, 'genders.npy'), genders)
        
        # Save metadata
        metadata = {
            'num_samples': len(images),
            'image_shape': images.shape[1:],
            'num_age_classes': len(np.unique(ages)),
            'num_gender_classes': len(np.unique(genders)),
            'age_labels': ['Child (0-12)', 'Teen (13-19)', 'Young Adult (20-35)', 
                          'Adult (36-55)', 'Senior (56+)'],
            'gender_labels': ['Female', 'Male']
        }
        
        with open(os.path.join(self.processed_data_path, 'metadata.pkl'), 'wb') as f:
            pickle.dump(metadata, f)
        
        print(f"Data saved successfully!")
        print(f"Final dataset statistics:")
        print(f"   - Total samples: {len(images)}")
        print(f"   - Image shape: {images.shape[1:]}")
        print(f"   - Age distribution: {dict(zip(range(5), np.bincount(ages)))}")
        print(f"   - Gender distribution: {dict(zip(['Female', 'Male'], np.bincount(genders)))}")

def main():
    """Main preprocessing function"""
    print("Starting IMDB-WIKI Dataset Preprocessing...")
    
    # Initialize preprocessor
    preprocessor = IMDBWIKIPreprocessor()
    
    # Process datasets
    all_images = []
    all_ages = []
    all_genders = []
    
    # Process IMDB dataset
    imdb_mat_path = os.path.join(preprocessor.imdb_path, 'imdb.mat')
    print(f"\nChecking IMDB dataset...")
    print(f"IMDB path exists: {os.path.exists(preprocessor.imdb_path)}")
    print(f"IMDB mat file exists: {os.path.exists(imdb_mat_path)}")
    
    if os.path.exists(preprocessor.imdb_path) and os.path.exists(imdb_mat_path):
        print("Found IMDB dataset")
        imdb_images, imdb_ages, imdb_genders = preprocessor.process_dataset(
            preprocessor.imdb_path, 'imdb.mat', max_images=1000
        )
        if imdb_images is not None:
            all_images.append(imdb_images)
            all_ages.append(imdb_ages)
            all_genders.append(imdb_genders)
    else:
        print("IMDB dataset not found or incomplete")
    
    # Process WIKI dataset
    wiki_mat_path = os.path.join(preprocessor.wiki_path, 'wiki.mat')
    print(f"\nChecking WIKI dataset...")
    print(f"WIKI path exists: {os.path.exists(preprocessor.wiki_path)}")
    print(f"WIKI mat file exists: {os.path.exists(wiki_mat_path)}")
    
    if os.path.exists(preprocessor.wiki_path) and os.path.exists(wiki_mat_path):
        print("Found WIKI dataset")
        wiki_images, wiki_ages, wiki_genders = preprocessor.process_dataset(
            preprocessor.wiki_path, 'wiki.mat', max_images=1000
        )
        if wiki_images is not None:
            all_images.append(wiki_images)
            all_ages.append(wiki_ages)
            all_genders.append(wiki_genders)
    else:
        print("WIKI dataset not found or incomplete")
    
    if not all_images:
        print("\nNo datasets found! Please download IMDB-WIKI dataset first.")
        print("Expected structure:")
        print(f"  {preprocessor.imdb_path}/imdb.mat")
        print(f"  {preprocessor.imdb_path}/00/ (folders with images)")
        print(f"  {preprocessor.wiki_path}/wiki.mat")
        print(f"  {preprocessor.wiki_path}/00/ (folders with images)")
        return
    
    # Combine datasets
    print("\nCombining datasets...")
    combined_images = np.vstack(all_images)
    combined_ages = np.hstack(all_ages)
    combined_genders = np.hstack(all_genders)
    
    # Balance dataset
    balanced_images, balanced_ages, balanced_genders = preprocessor.balance_dataset(
        combined_images, combined_ages, combined_genders
    )
    
    # Save processed data
    preprocessor.save_processed_data(balanced_images, balanced_ages, balanced_genders)
    
    print("\nPreprocessing completed successfully!")
    print("Next step: Run train.py to train the model")

if __name__ == "__main__":
    main()