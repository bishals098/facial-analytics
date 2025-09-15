import os
import sys
import time
import subprocess
from datetime import datetime
import argparse

class PipelineRunner:
    def __init__(self):
        self.start_time = None
        self.src_dir = "src"
        self.data_dir = "data"
        self.models_dir = "models"

    def check_prerequisites(self):
        """Check if all required files exist"""
        print("\n🔍 Checking prerequisites...")
        
        # Create directories if they don't exist
        required_dirs = [self.src_dir, self.data_dir, self.models_dir]
        for dir_name in required_dirs:
            if not os.path.exists(dir_name):
                print(f"📁 Creating directory: {dir_name}/")
                os.makedirs(dir_name, exist_ok=True)

        # Check required source files
        required_files = [
            f'{self.src_dir}/preprocessing.py',
            f'{self.src_dir}/train.py',
            f'{self.src_dir}/model.py',
            f'{self.src_dir}/detection.py',
            'streamlit_app.py'
        ]

        missing_files = []
        for file_path in required_files:
            if not os.path.exists(file_path):
                missing_files.append(file_path)

        if missing_files:
            print(f"❌ ERROR: Missing required files: {', '.join(missing_files)}")
            return False

        # Check for dataset directories
        dataset_paths = [f'{self.data_dir}/imdb_crop', f'{self.data_dir}/wiki_crop']
        dataset_found = any(os.path.exists(path) for path in dataset_paths)

        if not dataset_found:
            print("⚠️  WARNING: Dataset directories not found!")
            print(f"Please download IMDB-WIKI dataset to:")
            print(f"  - {self.data_dir}/imdb_crop/")
            print(f"  - {self.data_dir}/wiki_crop/")
            response = input("\nContinue anyway? (y/n): ").lower().strip()
            if response != 'y':
                return False

        print("✅ Prerequisites check completed successfully")
        return True

    def run_script(self, script_path, step_name):
        """Run a Python script with error handling"""
        print(f"\n{'='*60}")
        print(f"🚀 Running {step_name}")
        print(f"📄 Script: {script_path}")
        print(f"{'='*60}")
        step_start_time = time.time()

        try:
            # Set environment with proper Python path
            env = os.environ.copy()
            pythonpath = env.get('PYTHONPATH', '')
            src_path = os.path.join(os.getcwd(), self.src_dir)
            
            if pythonpath:
                env['PYTHONPATH'] = f"{src_path}{os.pathsep}{pythonpath}"
            else:
                env['PYTHONPATH'] = src_path

            # Run the script
            result = subprocess.run(
                [sys.executable, script_path],
                check=True,
                env=env,
                cwd=os.getcwd()
            )

            step_duration = time.time() - step_start_time
            print(f"\n✅ SUCCESS: {step_name} completed in {step_duration:.2f} seconds")
            return True

        except subprocess.CalledProcessError as e:
            step_duration = time.time() - step_start_time
            print(f"\n❌ ERROR: {step_name} failed after {step_duration:.2f} seconds")
            print(f"Exit code: {e.returncode}")
            return False

    def check_results(self):
        """Check if pipeline results exist"""
        print(f"\n{'='*60}")
        print("🔍 Checking Results")
        print(f"{'='*60}")

        results = {}

        # Check processed data (legacy .npy format)
        processed_data_dir = f"{self.data_dir}/processed_data"
        if os.path.exists(processed_data_dir):
            data_files = ['images.npy', 'ages.npy', 'genders.npy', 'metadata.pkl']
            processed_files = [f for f in data_files if os.path.exists(f'{processed_data_dir}/{f}')]
            results['legacy_processed_data'] = len(processed_files) == len(data_files)
        else:
            results['legacy_processed_data'] = False

        # Check HDF5 efficient data (new format)
        efficient_data_dir = f"{self.data_dir}/efficient_data"
        if os.path.exists(efficient_data_dir):
            index_file = f"{efficient_data_dir}/dataset_index.pkl"
            results['hdf5_processed_data'] = os.path.exists(index_file)
        else:
            results['hdf5_processed_data'] = False

        # Overall processed data status
        results['processed_data'] = results['legacy_processed_data'] or results['hdf5_processed_data']

        if results['hdf5_processed_data']:
            print("📊 Processed data: OK (HDF5 format - memory efficient)")
        elif results['legacy_processed_data']:
            print("📊 Processed data: OK (legacy .npy format)")
        else:
            print("📊 Processed data: MISSING")

        # Check trained models (both PyTorch .pth and legacy .keras)
        model_files = []
        if os.path.exists(self.models_dir):
            model_files = [f for f in os.listdir(self.models_dir) 
                          if f.endswith('.pth') or f.endswith('.keras')]

        results['trained_model'] = len(model_files) > 0

        print(f"🤖 Trained model: {'OK' if results['trained_model'] else 'MISSING'}")
        if results['trained_model']:
            pytorch_models = [f for f in model_files if f.endswith('.pth')]
            keras_models = [f for f in model_files if f.endswith('.keras')]
            
            if pytorch_models:
                print(f"   🔥 PyTorch models: {', '.join(pytorch_models)}")
            if keras_models:
                print(f"   📄 Keras models: {', '.join(keras_models)}")

        return results

    def show_next_steps(self, results):
        """Show what to do next"""
        print(f"\n{'='*60}")
        print("📋 Next Steps")
        print(f"{'='*60}")

        if all(results.values()):
            print("🎉 Pipeline completed successfully!")
            print("\n🚀 You can now:")
            print("1. 📱 Launch web app: streamlit run streamlit_app.py")
            print("2. 🎥 Test detection: python src/detection.py")
            print("3. 🔬 Test live camera: python src/detection.py (with model loaded)")
            
            if results.get('hdf5_processed_data'):
                print("\n💡 Your data is in memory-efficient HDF5 format!")
                print("   This enables training on large datasets (100k+ images)")
        else:
            print("⚠️  Pipeline completed with issues:")
            if not results['processed_data']:
                print("   - 📊 Run preprocessing: python src/preprocessing.py")
                print("     (This will create HDF5 chunked data for memory efficiency)")
            if not results['trained_model']:
                print("   - 🏋️  Run training: python src/train.py")
                print("     (This will train a PyTorch model with better performance)")

    def run_pipeline(self, skip_preprocessing=False, skip_training=False):
        """Run the complete PyTorch pipeline"""
        self.start_time = time.time()
        
        print("🎯 Age & Gender Detection System - PyTorch Pipeline Runner")
        print("=" * 65)
        print(f"🕐 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("🔥 Framework: PyTorch (migrated from TensorFlow)")
        print("📊 Data Format: HDF5 streaming for memory efficiency")

        # Check prerequisites
        if not self.check_prerequisites():
            print("❌ Prerequisites check failed. Exiting.")
            return False

        success = True

        # Step 1: Data Preprocessing (HDF5 efficient format)
        if not skip_preprocessing:
            if not self.run_script(f"{self.src_dir}/preprocessing.py", "HDF5 Data Preprocessing"):
                print("❌ Preprocessing failed. Stopping pipeline.")
                success = False
        else:
            print("\n⏭️  Skipping preprocessing step")

        # Step 2: Model Training
        if success and not skip_training:
            if not self.run_script(f"{self.src_dir}/train.py", "PyTorch Model Training"):
                print("❌ Training failed. Stopping pipeline.")
                success = False
        elif skip_training:
            print("\n⏭️  Skipping training step")

        # Pipeline Summary
        total_duration = time.time() - self.start_time
        print(f"\n{'='*60}")
        print("📊 Pipeline Summary")
        print(f"{'='*60}")
        print(f"🎯 Status: {'✅ SUCCESS' if success else '❌ FAILED'}")
        print(f"⏱️  Duration: {total_duration:.2f} seconds ({total_duration/60:.1f} minutes)")
        print(f"🕐 Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"🔥 Framework: PyTorch")
        print(f"💾 Memory Usage: Optimized with HDF5 streaming")

        # Check and show results
        results = self.check_results()
        self.show_next_steps(results)

        return success

def main():
    """Main function for PyTorch pipeline runner"""
    parser = argparse.ArgumentParser(
        description="Age & Gender Detection PyTorch Pipeline Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                           # Run full pipeline
  python main.py --skip-preprocessing     # Skip preprocessing, only train
  python main.py --skip-training         # Only preprocess data
  python main.py --skip-preprocessing --skip-training  # Just check status
        """
    )
    
    parser.add_argument(
        '--skip-preprocessing', 
        action='store_true', 
        help='Skip data preprocessing step (use existing processed data)'
    )
    
    parser.add_argument(
        '--skip-training', 
        action='store_true', 
        help='Skip model training step (use existing trained model)'
    )

    args = parser.parse_args()

    # Initialize and run pipeline
    runner = PipelineRunner()

    try:
        success = runner.run_pipeline(
            skip_preprocessing=args.skip_preprocessing,
            skip_training=args.skip_training
        )

        if success:
            print("\n🎉 Pipeline completed successfully!")
            print("🚀 Ready to run age & gender detection with PyTorch!")
            sys.exit(0)
        else:
            print("\n❌ Pipeline failed. Check error messages above.")
            sys.exit(1)

    except KeyboardInterrupt:
        print("\n\n⏹️  Pipeline interrupted by user")
        print("💡 You can resume by running: python main.py --skip-preprocessing")
        sys.exit(130)

    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()