import cv2
import numpy as np
from tensorflow.keras.models import load_model
import tensorflow as tf

class FaceDetectionPipeline:
    def __init__(self, model_path=None):
        print("🔧 Initializing Face Detection Pipeline...")
        
        # Initialize face detection
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        # Alternative face detection methods
        self.face_cascade_profile = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_profileface.xml'
        )
        
        # Age and gender labels
        self.age_labels = [
            'Child (0-12)', 'Teen (13-19)', 'Young Adult (20-35)', 
            'Adult (36-55)', 'Senior (56+)'
        ]
        self.gender_labels = ['Female', 'Male']
        
        # Colors for visualization
        self.colors = {
            'high_conf': (0, 255, 0),    # Green
            'medium_conf': (0, 255, 255), # Yellow
            'low_conf': (0, 0, 255)      # Red
        }
        
        # Load trained model if provided
        self.model = None
        if model_path:
            self.load_model(model_path)
        
        print("✅ Face Detection Pipeline initialized!")
    
    def load_model(self, model_path):
        """Load pre-trained model"""
        try:
            print(f"📦 Loading model from {model_path}...")
            self.model = load_model(model_path)
            print("✅ Model loaded successfully!")
            
            # Print model info
            print(f"   - Input shape: {self.model.input.shape}")
            print(f"   - Output shapes: {[output.shape for output in self.model.outputs]}")
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            self.model = None
    
    def detect_faces(self, image, scale_factor=1.1, min_neighbors=5, min_size=(30, 30)):
        """Detect faces in image using Haar Cascade"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Primary face detection (frontal faces)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=scale_factor,
            minNeighbors=min_neighbors,
            minSize=min_size,
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        # If no faces found, try with different parameters
        if len(faces) == 0:
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.05,
                minNeighbors=3,
                minSize=(20, 20),
                flags=cv2.CASCADE_SCALE_IMAGE
            )
        
        return faces
    
    def preprocess_face(self, face_img, target_size=(128, 128)):
        """Preprocess face image for model prediction"""
        try:
            # Handle empty face
            if face_img.size == 0:
                return None
            
            # Resize to model input size
            face_resized = cv2.resize(face_img, target_size)
            
            # Convert to RGB if needed
            if len(face_resized.shape) == 3 and face_resized.shape[2] == 3:
                # Already in correct format
                pass
            else:
                face_resized = cv2.cvtColor(face_resized, cv2.COLOR_GRAY2RGB)
            
            # Normalize pixel values (0-1 range)
            face_normalized = face_resized.astype(np.float32) / 255.0
            
            # Add batch dimension
            face_batch = np.expand_dims(face_normalized, axis=0)
            
            return face_batch
        except Exception as e:
            print(f"❌ Error preprocessing face: {e}")
            return None
    
    def predict_age_gender(self, face_img):
        """Predict age and gender for a face"""
        if self.model is None:
            return "Model not loaded", "Model not loaded", 0.0, 0.0
        
        try:
            # Preprocess face
            processed_face = self.preprocess_face(face_img)
            if processed_face is None:
                return "Preprocessing Error", "Preprocessing Error", 0.0, 0.0
            
            # Make predictions
            predictions = self.model.predict(processed_face, verbose=0)
            age_pred, gender_pred = predictions
            
            # Get predicted classes and confidence
            age_class = np.argmax(age_pred[0])
            gender_class = np.argmax(gender_pred[0])
            
            age_confidence = float(np.max(age_pred[0]))
            gender_confidence = float(np.max(gender_pred[0]))
            
            # Get labels
            age_label = self.age_labels[age_class] if age_class < len(self.age_labels) else "Unknown"
            gender_label = self.gender_labels[gender_class] if gender_class < len(self.gender_labels) else "Unknown"
            
            return age_label, gender_label, age_confidence, gender_confidence
            
        except Exception as e:
            print(f"❌ Error in prediction: {e}")
            return "Prediction Error", "Prediction Error", 0.0, 0.0
    
    def get_confidence_color(self, age_conf, gender_conf):
        """Get color based on confidence levels"""
        min_conf = min(age_conf, gender_conf)
        
        if min_conf >= 0.8:
            return self.colors['high_conf']
        elif min_conf >= 0.6:
            return self.colors['medium_conf']
        else:
            return self.colors['low_conf']
    
    def draw_prediction(self, image, x, y, w, h, age_label, gender_label, 
                       age_conf, gender_conf, face_id):
        """Draw bounding box and predictions on image"""
        # Get color based on confidence
        color = self.get_confidence_color(age_conf, gender_conf)
        
        # Draw bounding box
        cv2.rectangle(image, (x, y), (x+w, y+h), color, 2)
        
        # Prepare text
        min_conf = min(age_conf, gender_conf)
        if min_conf > 0.5:
            main_text = f"{gender_label}, {age_label}"
            conf_text = f"Conf: A={age_conf:.2f}, G={gender_conf:.2f}"
        else:
            main_text = f"Low Confidence"
            conf_text = f"A={age_conf:.2f}, G={gender_conf:.2f}"
        
        face_id_text = f"Face {face_id + 1}"
        
        # Calculate text sizes
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        
        (text_w, text_h), _ = cv2.getTextSize(main_text, font, font_scale, thickness)
        (conf_w, conf_h), _ = cv2.getTextSize(conf_text, font, 0.4, 1)
        (id_w, id_h), _ = cv2.getTextSize(face_id_text, font, 0.5, 1)
        
        # Calculate background rectangle size
        bg_width = max(text_w, conf_w, id_w) + 10
        bg_height = text_h + conf_h + id_h + 20
        
        # Draw background rectangle
        cv2.rectangle(image, (x, y - bg_height - 5), (x + bg_width, y), color, -1)
        
        # Draw texts
        y_offset = y - bg_height + 15
        cv2.putText(image, face_id_text, (x + 5, y_offset), font, 0.5, (255, 255, 255), 1)
        
        y_offset += id_h + 5
        cv2.putText(image, main_text, (x + 5, y_offset), font, font_scale, (255, 255, 255), thickness)
        
        y_offset += text_h + 5
        cv2.putText(image, conf_text, (x + 5, y_offset), font, 0.4, (255, 255, 255), 1)
    
    def process_frame(self, frame, draw_predictions=True, confidence_threshold=0.5):
        """Process a single frame with face detection and prediction"""
        # Create a copy for processing
        processed_frame = frame.copy()
        
        # Detect faces
        faces = self.detect_faces(frame)
        
        results = []
        
        for i, (x, y, w, h) in enumerate(faces):
            # Extract face region with some padding
            padding = max(5, min(w, h) // 10)
            face_x1 = max(0, x - padding)
            face_y1 = max(0, y - padding)
            face_x2 = min(frame.shape[1], x + w + padding)
            face_y2 = min(frame.shape[0], y + h + padding)
            
            face_img = frame[face_y1:face_y2, face_x1:face_x2]
            
            if face_img.size == 0:
                continue
            
            # Predict age and gender
            age_label, gender_label, age_conf, gender_conf = self.predict_age_gender(face_img)
            
            # Store results
            face_result = {
                'face_id': i,
                'bbox': (x, y, w, h),
                'age_prediction': age_label,
                'gender_prediction': gender_label,
                'age_confidence': age_conf,
                'gender_confidence': gender_conf
            }
            results.append(face_result)
            
            # Draw predictions if requested
            if draw_predictions:
                self.draw_prediction(
                    processed_frame, x, y, w, h,
                    age_label, gender_label, age_conf, gender_conf, i
                )
        
        # Draw frame info
        info_text = f"Faces detected: {len(results)}"
        cv2.putText(processed_frame, info_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Draw confidence legend
        y_pos = processed_frame.shape[0] - 80
        cv2.putText(processed_frame, "Confidence:", (10, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Color legend
        legend_items = [
            ("High (>0.8)", self.colors['high_conf']),
            ("Medium (>0.6)", self.colors['medium_conf']),
            ("Low (<0.6)", self.colors['low_conf'])
        ]
        
        for i, (label, color) in enumerate(legend_items):
            y_pos += 20
            cv2.rectangle(processed_frame, (10, y_pos - 10), (30, y_pos + 5), color, -1)
            cv2.putText(processed_frame, label, (35, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return processed_frame, results
    
    def process_video_stream(self, video_source=0, save_video=False, 
                           output_path="output.avi", confidence_threshold=0.5):
        """Process live video stream"""
        print(f"🎥 Starting video stream from source: {video_source}")
        
        cap = cv2.VideoCapture(video_source)
        
        if not cap.isOpened():
            print(f"❌ Error: Could not open video source {video_source}")
            return
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"📹 Video properties: {width}x{height} @ {fps} FPS")
        
        if save_video:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            print(f"💾 Saving video to: {output_path}")
        
        frame_count = 0
        total_faces = 0
        
        try:
            print("🎬 Press 'q' to quit, 'p' to pause, 's' to save screenshot")
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("📺 End of video stream")
                    break
                
                # Process frame
                processed_frame, results = self.process_frame(
                    frame, confidence_threshold=confidence_threshold
                )
                
                frame_count += 1
                total_faces += len(results)
                
                # Add frame counter
                counter_text = f"Frame: {frame_count} | Total faces: {total_faces}"
                cv2.putText(processed_frame, counter_text, (10, height - 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Display frame
                cv2.imshow('Age & Gender Detection', processed_frame)
                
                if save_video:
                    out.write(processed_frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('p'):
                    print("⏸️  Paused. Press any key to continue...")
                    cv2.waitKey(0)
                elif key == ord('s'):
                    screenshot_path = f"screenshot_{frame_count}.jpg"
                    cv2.imwrite(screenshot_path, processed_frame)
                    print(f"📸 Screenshot saved: {screenshot_path}")
                    
        except KeyboardInterrupt:
            print("\n⏹️  Video processing interrupted by user")
        
        finally:
            cap.release()
            if save_video:
                out.release()
            cv2.destroyAllWindows()
            
            print(f"📊 Processing complete:")
            print(f"   - Frames processed: {frame_count}")
            print(f"   - Total faces detected: {total_faces}")
            print(f"   - Average faces per frame: {total_faces/max(1, frame_count):.2f}")

if __name__ == "__main__":
    # Test detection pipeline
    print("🧪 Testing Face Detection Pipeline...")
    
    # Initialize without model
    detector = FaceDetectionPipeline()
    
    # Test with webcam (if available)
    try:
        detector.process_video_stream(video_source=0)
    except KeyboardInterrupt:
        print("Test completed!")