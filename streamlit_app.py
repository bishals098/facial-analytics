import streamlit as st
import cv2
import numpy as np
import tempfile
from PIL import Image
import io
from detection import FaceDetectionPipeline
import os
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Age & Gender Detection",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
.main-header {
    text-align: center;
    color: #1e88e5;
    font-size: 3rem;
    font-weight: bold;
    margin-bottom: 2rem;
}

.sub-header {
    text-align: center;
    color: #666;
    font-size: 1.2rem;
    margin-bottom: 3rem;
}

.result-box {
    background-color: #f0f2f6;
    padding: 1rem;
    border-radius: 0.5rem;
    border-left: 4px solid #1e88e5;
    margin: 1rem 0;
}

.metric-card {
    background-color: #ffffff;
    padding: 1rem;
    border-radius: 0.5rem;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'detector' not in st.session_state:
    st.session_state.detector = None

@st.cache_resource
def load_detector(model_path):
    """Load the face detection model"""
    try:
        detector = FaceDetectionPipeline(model_path=model_path)
        return detector
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def process_uploaded_image(uploaded_file, detector, confidence_threshold):
    """Process uploaded image"""
    try:
        # Convert uploaded file to OpenCV image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)
        
        if image is None:
            st.error("Could not decode image")
            return None, None
        
        # Process image
        processed_image, results = detector.process_frame(
            image, 
            draw_predictions=True,
            confidence_threshold=confidence_threshold
        )
        
        return processed_image, results
        
    except Exception as e:
        st.error(f"Error processing image: {e}")
        return None, None

def display_results(results):
    """Display detection results in a formatted way"""
    if not results:
        st.info("No faces detected in the image.")
        return
    
    st.success(f"✅ Detected {len(results)} face(s)")
    
    # Create columns for results
    cols = st.columns(min(len(results), 3))
    
    for i, result in enumerate(results):
        col_idx = i % 3
        with cols[col_idx]:
            st.markdown(f"""
            <div class="result-box">
                <h4>👤 Face {result['face_id'] + 1}</h4>
                <p><strong>Gender:</strong> {result['gender_prediction']}</p>
                <p><strong>Age:</strong> {result['age_prediction']}</p>
                <p><strong>Confidence:</strong></p>
                <ul>
                    <li>Age: {result['age_confidence']:.2%}</li>
                    <li>Gender: {result['gender_confidence']:.2%}</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

def main():
    # Header
    st.markdown('<h1 class="main-header">🎯 Age & Gender Detection</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">AI-powered facial recognition system using deep learning</p>', unsafe_allow_html=True)
    
    # Sidebar configuration
    st.sidebar.header("⚙️ Configuration")
    
    # Model selection
    # Model selection - look in models directory
    models_dir = 'models'
    if os.path.exists(models_dir):
        model_files = [f for f in os.listdir(models_dir) if f.endswith('.keras')]
    else:
        model_files = []

    if not model_files:
        st.error("❌ No trained model found! Please train a model first using train.py")
        st.info("💡 Make sure you have .keras model files in the models/ directory")
        return

    selected_model = st.sidebar.selectbox("Select Model", model_files)
    model_path = os.path.join(models_dir, selected_model)

    # Load detector
    if st.session_state.detector is None or st.sidebar.button("🔄 Reload Model"):
        with st.spinner("Loading model..."):
            st.session_state.detector = load_detector(model_path)  # Use full path
        
        if st.session_state.detector is None:
            st.error("Failed to load model")
            return
        else:
            st.sidebar.success("✅ Model loaded successfully!")
    
    detector = st.session_state.detector
    
    # Detection parameters
    st.sidebar.subheader("🎛️ Detection Parameters")
    confidence_threshold = st.sidebar.slider(
        "Confidence Threshold", 
        min_value=0.1, 
        max_value=1.0, 
        value=0.7, 
        step=0.05,
        help="Minimum confidence level for predictions"
    )
    
    # Mode selection
    st.sidebar.subheader("📱 Detection Mode")
    mode = st.sidebar.radio(
        "Choose input method:",
        ["📸 Upload Image", "🎥 Upload Video", "📹 Live Camera"]
    )
    
    # Main content area
    if mode == "📸 Upload Image":
        st.header("📸 Image Analysis")
        
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=['jpg', 'jpeg', 'png', 'bmp', 'tiff'],
            help="Upload a clear image with visible faces"
        )
        
        if uploaded_file is not None:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📤 Original Image")
                image = Image.open(uploaded_file)
                st.image(image, use_column_width=True)
                
                # Image info
                st.info(f"Image size: {image.size[0]}x{image.size[1]} pixels")
            
            with col2:
                st.subheader("🎯 Detection Results")
                
                with st.spinner("Processing image..."):
                    processed_image, results = process_uploaded_image(
                        uploaded_file, detector, confidence_threshold
                    )
                
                if processed_image is not None:
                    # Convert BGR to RGB for display
                    processed_image_rgb = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)
                    st.image(processed_image_rgb, use_column_width=True)
                    
                    # Download button for processed image
                    processed_pil = Image.fromarray(processed_image_rgb)
                    buf = io.BytesIO()
                    processed_pil.save(buf, format='PNG')
                    
                    st.download_button(
                        label="💾 Download Processed Image",
                        data=buf.getvalue(),
                        file_name=f"processed_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                        mime="image/png"
                    )
            
            # Display detailed results
            if 'results' in locals():
                st.header("📊 Detailed Results")
                display_results(results)
    
    elif mode == "🎥 Upload Video":
        st.header("🎥 Video Analysis")
        
        uploaded_video = st.file_uploader(
            "Choose a video file",
            type=['mp4', 'avi', 'mov', 'mkv'],
            help="Upload a video file for frame-by-frame analysis"
        )
        
        if uploaded_video is not None:
            # Save uploaded video to temporary file
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            temp_file.write(uploaded_video.read())
            temp_file_path = temp_file.name
            temp_file.close()
            
            st.info("📹 Video uploaded successfully! Click 'Process Video' to analyze.")
            
            if st.button("🚀 Process Video"):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                try:
                    # Open video
                    cap = cv2.VideoCapture(temp_file_path)
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    fps = int(cap.get(cv2.CAP_PROP_FPS))
                    
                    st.info(f"📊 Video info: {total_frames} frames @ {fps} FPS")
                    
                    # Process video frames
                    frame_count = 0
                    total_faces = 0
                    sample_results = []
                    
                    # Create placeholder for sample frames
                    sample_frames_container = st.container()
                    
                    while cap.isOpened():
                        ret, frame = cap.read()
                        if not ret:
                            break
                        
                        # Process every 30th frame for efficiency
                        if frame_count % 30 == 0:
                            processed_frame, results = detector.process_frame(
                                frame, confidence_threshold=confidence_threshold
                            )
                            
                            total_faces += len(results)
                            
                            # Store sample results
                            if len(sample_results) < 5 and results:
                                sample_results.append({
                                    'frame': frame_count,
                                    'image': cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB),
                                    'results': results
                                })
                        
                        frame_count += 1
                        progress = frame_count / total_frames
                        progress_bar.progress(progress)
                        status_text.text(f"Processing frame {frame_count}/{total_frames}")
                    
                    cap.release()
                    
                    # Display results
                    st.success(f"✅ Video processing complete!")
                    st.info(f"📊 Found {total_faces} faces across {frame_count} frames")
                    
                    # Show sample frames
                    if sample_results:
                        st.header("🖼️ Sample Frames")
                        for i, sample in enumerate(sample_results):
                            st.subheader(f"Frame {sample['frame']}")
                            col1, col2 = st.columns([2, 1])
                            
                            with col1:
                                st.image(sample['image'], use_column_width=True)
                            
                            with col2:
                                display_results(sample['results'])
                    
                except Exception as e:
                    st.error(f"Error processing video: {e}")
                
                finally:
                    # Cleanup
                    os.unlink(temp_file_path)
    
    elif mode == "📹 Live Camera":
        st.header("📹 Live Camera Detection")
        
        st.info("🚧 Live camera feature requires additional setup for web deployment.")
        st.markdown("""
        **For local development:**
        1. Run the detection pipeline directly using `detection.py`
        2. Use the `process_video_stream()` method
        3. Set `video_source=0` for default webcam
        
        **Code example:**
        ```
        from detection import FaceDetectionPipeline
        
        detector = FaceDetectionPipeline('your_model.h5')
        detector.process_video_stream(video_source=0)
        ```
        """)
        
        if st.button("🎥 Launch Camera (Local)"):
            st.info("This will launch camera in a separate OpenCV window...")
            try:
                detector.process_video_stream(video_source=0)
            except Exception as e:
                st.error(f"Camera error: {e}")
    
    # Sidebar info
    st.sidebar.markdown("---")
    st.sidebar.subheader("📋 Model Information")
    st.sidebar.info(f"**Model:** {selected_model}")
    st.sidebar.info("**Age Groups:**\n- Child (0-12)\n- Teen (13-19)\n- Young Adult (20-35)\n- Adult (36-55)\n- Senior (56+)")
    st.sidebar.info("**Gender Classes:**\n- Female\n- Male")
    
    # Performance info
    st.sidebar.subheader("⚡ Expected Performance")
    st.sidebar.success("**Accuracy:**\n- Gender: ~90-95%\n- Age Groups: ~80-85%")
    st.sidebar.success("**Speed:**\n- Real-time processing\n- Multi-face detection")
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<p style='text-align: center; color: #666;'>"
        "🤖 Powered by TensorFlow & OpenCV | Built with Streamlit"
        "</p>", 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()