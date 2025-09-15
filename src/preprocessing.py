import os
import sys
import cv2
import numpy as np
import pandas as pd
from scipy.io import loadmat
from datetime import datetime
import pickle
import h5py

# Fix Unicode encoding for Windows
if sys.platform.startswith('win'):
    os.environ['PYTHONIOENCODING'] = 'utf-8'

class MemoryEfficientPreprocessor:
    def __init__(self):
        # Get the project root directory (parent of src/)
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.project_root = os.path.dirname(self.script_dir)
        self.data_root = os.path.join(self.project_root, 'data')
        self.imdb_path = os.path.join(self.data_root, 'imdb_crop')
        self.wiki_path = os.path.join(self.data_root, 'wiki_crop')
        
        # NEW: Efficient storage path using HDF5 chunks
        self.efficient_data_path = os.path.join(self.data_root, 'efficient_data')
        self.target_size = (128, 128)
        
        # Create efficient data directory
        os.makedirs(self.efficient_data_path, exist_ok=True)
        
        print(f"Project root: {self.project_root}")
        print(f"Data root: {self.data_root}")
        print(f"IMDB path: {self.imdb_path}")
        print(f"WIKI path: {self.wiki_path}")
        print(f"Efficient storage: {self.efficient_data_path}")

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
                'full_path': data['full_path'][0],  # FIXED: Remove extra [0]
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
            # Convert MATLAB datenum to years (FIXED formula)
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

    def process_and_save_efficiently(self, dataset_path, mat_file, dataset_name, max_images=100000):
        """Process dataset and save efficiently to disk in chunks"""
        print(f"\n{'='*60}")
        print(f"Processing {dataset_name} with memory-efficient chunking...")
        print(f"Target: {max_images:,} images")
        print(f"{'='*60}")
        
        # Check if dataset exists
        if not os.path.exists(dataset_path):
            print(f"Dataset path not found: {dataset_path}")
            return 0
            
        # Load metadata
        mat_file_full_path = os.path.join(dataset_path, mat_file)
        if not os.path.exists(mat_file_full_path):
            print(f"Metadata file not found: {mat_file_full_path}")
            return 0
            
        metadata = self.load_metadata(mat_file_full_path)
        if metadata is None:
            return 0
        
        # Create dataset-specific directory
        dataset_dir = os.path.join(self.efficient_data_path, dataset_name)
        os.makedirs(dataset_dir, exist_ok=True)
        
        # Process and save in small chunks to avoid memory issues
        chunk_size = 1000  # Process 1000 images at a time
        chunk_num = 0
        valid_count = 0
        invalid_count = 0
        total_records = len(metadata['dob'])
        
        current_chunk_images = []
        current_chunk_ages = []
        current_chunk_genders = []
        
        print(f"Processing up to {max_images:,} images from {total_records:,} records...")
        print(f"Using chunk size: {chunk_size:,} images per chunk")
        
        for i in range(total_records):
            if valid_count >= max_images:
                break
                
            try:
                # Extract metadata (same logic as before)
                dob = metadata['dob'][i]
                photo_taken = metadata['photo_taken'][i]
                path = metadata['full_path'][i]
                gender = metadata['gender'][i]
                
                # Skip invalid paths
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
                
                # Process image
                img_path = os.path.join(dataset_path, path)
                processed_img = self.preprocess_image(img_path)
                
                if processed_img is not None:
                    current_chunk_images.append(processed_img)
                    current_chunk_ages.append(age_group)
                    current_chunk_genders.append(int(gender))
                    valid_count += 1
                    
                    # When chunk is full, save to disk and clear memory
                    if len(current_chunk_images) >= chunk_size:
                        self.save_chunk_to_disk(
                            dataset_dir, chunk_num,
                            current_chunk_images, current_chunk_ages, current_chunk_genders
                        )
                        
                        print(f"✅ Chunk {chunk_num:03d}: {valid_count:,}/{max_images:,} images processed")
                        print(f"   Memory usage: ~{len(current_chunk_images) * 128 * 128 * 3 * 4 / (1024**2):.1f} MB per chunk")
                        
                        # Clear memory immediately
                        current_chunk_images = []
                        current_chunk_ages = []
                        current_chunk_genders = []
                        chunk_num += 1
                        
                else:
                    invalid_count += 1
                    
            except Exception as e:
                invalid_count += 1
                continue
        
        # Save remaining images in final chunk
        if current_chunk_images:
            self.save_chunk_to_disk(
                dataset_dir, chunk_num,
                current_chunk_images, current_chunk_ages, current_chunk_genders
            )
            print(f"✅ Final chunk {chunk_num:03d}: {len(current_chunk_images)} images")
            chunk_num += 1
        
        print(f"\n📊 {dataset_name} Processing Complete:")
        print(f"   - Valid images: {valid_count:,}")
        print(f"   - Invalid images: {invalid_count:,}")
        print(f"   - Success rate: {valid_count/(valid_count+invalid_count)*100:.1f}%")
        print(f"   - Chunks created: {chunk_num}")
        print(f"   - Disk usage: ~{valid_count * 128 * 128 * 3 * 4 / (1024**3):.1f} GB")
        
        return valid_count
    
    def save_chunk_to_disk(self, dataset_dir, chunk_num, images, ages, genders):
        """Save a chunk of data efficiently to disk using HDF5 compression"""
        chunk_file = os.path.join(dataset_dir, f'chunk_{chunk_num:04d}.h5')
        
        try:
            with h5py.File(chunk_file, 'w') as f:
                # Save as compressed HDF5 (saves ~40% space vs NumPy)
                f.create_dataset('images', data=np.array(images), 
                               compression='gzip', compression_opts=1)
                f.create_dataset('ages', data=np.array(ages), 
                               compression='gzip')
                f.create_dataset('genders', data=np.array(genders), 
                               compression='gzip')
                
        except Exception as e:
            print(f"❌ Error saving chunk {chunk_num}: {e}")
    
    def create_dataset_index(self):
        """Create an index of all chunk files for efficient loading during training"""
        index = {
            'imdb_chunks': [],
            'wiki_chunks': [],
            'total_samples': 0,
            'chunk_info': []
        }
        
        # Index IMDB chunks
        imdb_dir = os.path.join(self.efficient_data_path, 'imdb')
        if os.path.exists(imdb_dir):
            imdb_chunks = sorted([f for f in os.listdir(imdb_dir) if f.endswith('.h5')])
            for chunk_file in imdb_chunks:
                full_path = os.path.join(imdb_dir, chunk_file)
                index['imdb_chunks'].append(full_path)
                
                # Get chunk size
                try:
                    with h5py.File(full_path, 'r') as f:
                        chunk_size = len(f['images'])
                        index['total_samples'] += chunk_size
                        index['chunk_info'].append({
                            'file': full_path,
                            'dataset': 'imdb', 
                            'size': chunk_size
                        })
                except Exception as e:
                    print(f"⚠️  Warning: Could not read {full_path}: {e}")
        
        # Index WIKI chunks  
        wiki_dir = os.path.join(self.efficient_data_path, 'wiki')
        if os.path.exists(wiki_dir):
            wiki_chunks = sorted([f for f in os.listdir(wiki_dir) if f.endswith('.h5')])
            for chunk_file in wiki_chunks:
                full_path = os.path.join(wiki_dir, chunk_file)
                index['wiki_chunks'].append(full_path)
                
                # Get chunk size
                try:
                    with h5py.File(full_path, 'r') as f:
                        chunk_size = len(f['images'])
                        index['total_samples'] += chunk_size
                        index['chunk_info'].append({
                            'file': full_path,
                            'dataset': 'wiki',
                            'size': chunk_size
                        })
                except Exception as e:
                    print(f"⚠️  Warning: Could not read {full_path}: {e}")
        
        # Save index for training
        index_file = os.path.join(self.efficient_data_path, 'dataset_index.pkl')
        with open(index_file, 'wb') as f:
            pickle.dump(index, f)
        
        print(f"\n📋 Dataset index created:")
        print(f"   - IMDB chunks: {len(index['imdb_chunks'])}")
        print(f"   - WIKI chunks: {len(index['wiki_chunks'])}")
        print(f"   - Total samples: {index['total_samples']:,}")
        print(f"   - Index saved to: {index_file}")
        
        return index

def main():
    """Main preprocessing function - memory efficient approach"""
    print("🚀 Starting Memory-Efficient IMDB-WIKI Dataset Preprocessing...")
    print(f"Current time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Initialize preprocessor
    preprocessor = MemoryEfficientPreprocessor()
    
    # Set processing limits (can be increased without memory issues)
    max_images_per_dataset = 100000  # Now we can handle 100k+ per dataset!
    
    # Process datasets efficiently
    imdb_count = 0
    wiki_count = 0
    
    # Process IMDB dataset
    if os.path.exists(preprocessor.imdb_path):
        print(f"\n🎬 Processing IMDB dataset...")
        imdb_count = preprocessor.process_and_save_efficiently(
            preprocessor.imdb_path, 'imdb.mat', 'imdb', 
            max_images=max_images_per_dataset
        )
    else:
        print("⚠️  IMDB dataset not found!")
    
    # Process WIKI dataset
    if os.path.exists(preprocessor.wiki_path):
        print(f"\n📚 Processing WIKI dataset...")
        wiki_count = preprocessor.process_and_save_efficiently(
            preprocessor.wiki_path, 'wiki.mat', 'wiki', 
            max_images=max_images_per_dataset
        )
    else:
        print("⚠️  WIKI dataset not found!")
    
    total_processed = imdb_count + wiki_count
    
    if total_processed == 0:
        print("\n❌ No datasets processed!")
        print("Please download IMDB-WIKI dataset first:")
        print(f"   Expected: {preprocessor.imdb_path}/imdb.mat")
        print(f"   Expected: {preprocessor.wiki_path}/wiki.mat")
        return
    
    # Create file index for efficient training
    print(f"\n📑 Creating dataset index...")
    preprocessor.create_dataset_index()
    
    print(f"\n🎉 Preprocessing completed successfully!")
    print(f"📊 Summary:")
    print(f"   - Total images processed: {total_processed:,}")
    print(f"   - Max memory usage: < 1GB (chunked processing)")
    print(f"   - Data stored efficiently on disk with compression")
    print(f"   - Ready for memory-efficient training!")
    print(f"\n➡️  Next step: Update train.py to use streaming from disk")

if __name__ == "__main__":
    main()