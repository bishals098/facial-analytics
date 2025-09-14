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
        print("\nChecking prerequisites...")
        
        # Create directories if they don't exist
        required_dirs = [self.src_dir, self.data_dir, self.models_dir]
        for dir_name in required_dirs:
            if not os.path.exists(dir_name):
                print(f"Creating directory: {dir_name}/")
                os.makedirs(dir_name, exist_ok=True)
        
        # Check required source files
        required_files = [
            f'{self.src_dir}/preprocessing.py',
            f'{self.src_dir}/train.py', 
            f'{self.src_dir}/model.py',
            'streamlit_app.py'
        ]
        
        missing_files = []
        for file in required_files:
            if not os.path.exists(file):
                missing_files.append(file)
        
        if missing_files:
            print(f"ERROR: Missing required files: {', '.join(missing_files)}")
            return False
        
        # Check for dataset directories
        dataset_paths = [f'{self.data_dir}/imdb_crop', f'{self.data_dir}/wiki_crop']
        dataset_found = any(os.path.exists(path) for path in dataset_paths)
        
        if not dataset_found:
            print("WARNING: Dataset directories not found!")
            print(f"Please download IMDB-WIKI dataset to {self.data_dir}/imdb_crop/ or {self.data_dir}/wiki_crop/")
            response = input("\nContinue anyway? (y/n): ").lower().strip()
            if response != 'y':
                return False
        
        print("Prerequisites check completed successfully")
        return True
    
    def run_script(self, script_path, step_name):
        """Run a Python script with error handling"""
        print(f"\n{'='*50}")
        print(f"Running {step_name}")
        print(f"Script: {script_path}")
        print(f"{'='*50}")
        
        step_start_time = time.time()
        
        try:
            # Set environment
            env = os.environ.copy()
            pythonpath = env.get('PYTHONPATH', '')
            if pythonpath:
                env['PYTHONPATH'] = f"{os.path.join(os.getcwd(), self.src_dir)};{pythonpath}"
            else:
                env['PYTHONPATH'] = os.path.join(os.getcwd(), self.src_dir)
            
            # Run the script
            result = subprocess.run(
                [sys.executable, script_path],
                check=True,
                env=env,
                cwd=os.getcwd()
            )
            
            step_duration = time.time() - step_start_time
            print(f"\nSUCCESS: {step_name} completed in {step_duration:.2f} seconds")
            return True
            
        except subprocess.CalledProcessError as e:
            step_duration = time.time() - step_start_time
            print(f"\nERROR: {step_name} failed after {step_duration:.2f} seconds")
            print(f"Exit code: {e.returncode}")
            return False
    
    def check_results(self):
        """Check if pipeline results exist"""
        print(f"\n{'='*50}")
        print("Checking Results")
        print(f"{'='*50}")
        
        results = {}
        
        # Check processed data
        processed_data_dir = f"{self.data_dir}/processed_data"
        if os.path.exists(processed_data_dir):
            data_files = ['images.npy', 'ages.npy', 'genders.npy', 'metadata.pkl']
            processed_files = [f for f in data_files if os.path.exists(f'{processed_data_dir}/{f}')]
            results['processed_data'] = len(processed_files) == len(data_files)
        else:
            results['processed_data'] = False
        
        print(f"Processed data: {'OK' if results['processed_data'] else 'MISSING'}")
        
        # Check trained model
        model_files = []
        if os.path.exists(self.models_dir):
            model_files = [f for f in os.listdir(self.models_dir) if f.endswith('.h5')]
        
        results['trained_model'] = len(model_files) > 0
        print(f"Trained model: {'OK' if results['trained_model'] else 'MISSING'}")
        if results['trained_model']:
            print(f"Model files: {', '.join(model_files)}")
        
        return results
    
    def show_next_steps(self, results):
        """Show what to do next"""
        print(f"\n{'='*50}")
        print("Next Steps")
        print(f"{'='*50}")
        
        if all(results.values()):
            print("Pipeline completed successfully!")
            print("\nYou can now:")
            print("1. Launch web app: streamlit run streamlit_app.py")
            print("2. Test detection: python src/detection.py")
        else:
            print("Pipeline completed with issues:")
            if not results['processed_data']:
                print("- Run preprocessing: python src/preprocessing.py")
            if not results['trained_model']:
                print("- Run training: python src/train.py")
    
    def run_pipeline(self, skip_preprocessing=False, skip_training=False):
        """Run the complete pipeline"""
        self.start_time = time.time()
        
        print("Age & Gender Detection System - Pipeline Runner")
        print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Check prerequisites
        if not self.check_prerequisites():
            print("Prerequisites check failed. Exiting.")
            return False
        
        success = True
        
        # Step 1: Preprocessing
        if not skip_preprocessing:
            if not self.run_script(f"{self.src_dir}/preprocessing.py", "Data Preprocessing"):
                print("Preprocessing failed. Stopping.")
                success = False
        else:
            print("\nSkipping preprocessing step")
        
        # Step 2: Training
        if success and not skip_training:
            if not self.run_script(f"{self.src_dir}/train.py", "Model Training"):
                print("Training failed. Stopping.")
                success = False
        elif skip_training:
            print("\nSkipping training step")
        
        # Summary
        total_duration = time.time() - self.start_time
        print(f"\n{'='*50}")
        print("Pipeline Summary")
        print(f"{'='*50}")
        print(f"Status: {'SUCCESS' if success else 'FAILED'}")
        print(f"Duration: {total_duration:.2f} seconds ({total_duration/60:.1f} minutes)")
        print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Check results
        results = self.check_results()
        self.show_next_steps(results)
        
        return success

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Age & Gender Detection Pipeline Runner")
    parser.add_argument('--skip-preprocessing', action='store_true', help='Skip preprocessing step')
    parser.add_argument('--skip-training', action='store_true', help='Skip training step')
    
    args = parser.parse_args()
    
    runner = PipelineRunner()
    
    try:
        success = runner.run_pipeline(
            skip_preprocessing=args.skip_preprocessing,
            skip_training=args.skip_training
        )
        
        if success:
            print("\nPipeline completed successfully!")
            sys.exit(0)
        else:
            print("\nPipeline failed. Check error messages above.")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n\nPipeline interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()