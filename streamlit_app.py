import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os
from src.detection import FaceDetectionPipeline
import time

# Page configuration
st.set_page_config(
    page_title="Live Age & Gender Detection",
    page_icon="🎯",
    layout="wide"
)

# Initialize session state
if 'detector' not in st.session_state:
    st.session_state.detector = None
if 'camera_active' not in st.session_state:
    st.session_state.camera_active = False

@st.cache_resource
def load_detector(model_path):
    """Load the face detection model"""
    try:
        detector = FaceDetectionPipeline(model_path=model_path)
        return detector
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def main():
    """Main Streamlit application - Live Camera Only"""
    
    # Header
    st.title("🎯 Live Age & Gender Detection")
    st.markdown("**Real-time face detection using your webcam**")
    
    # Sidebar for model selection and settings
    st.sidebar.header("⚙️ Configuration")
    
    # Model selection
    models_dir = 'models'
    if os.path.exists(models_dir):
        model_files = [f for f in os.listdir(models_dir) if f.endswith('.keras')]
    else:
        model_files = []
    
    if not model_files:
        st.error("❌ No trained model found!")
        st.info("💡 Train a model first using: `python main.py`")
        return
    
    selected_model = st.sidebar.selectbox("Select Model", model_files)
    model_path = os.path.join(models_dir, selected_model)
    
    # Load detector
    if st.session_state.detector is None:
        with st.spinner("Loading model..."):
            st.session_state.detector = load_detector(model_path)
        
        if st.session_state.detector is None:
            st.error("Failed to load model")
            return
        else:
            st.sidebar.success("✅ Model loaded!")
    
    detector = st.session_state.detector
    
    # Detection settings
    st.sidebar.subheader("🎛️ Detection Settings")
    confidence_threshold = st.sidebar.slider(
        "Confidence Threshold", 
        min_value=0.1, 
        max_value=1.0, 
        value=0.7, 
        step=0.05
    )
    
    # Camera controls
    st.sidebar.subheader("📹 Camera Controls")
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📹 Live Camera Feed")
        
        # Camera control buttons
        start_button = st.button("🎥 Start Camera", type="primary")
        stop_button = st.button("⏹️ Stop Camera")
        
        # Placeholder for camera feed
        camera_placeholder = st.empty()
        
        if start_button:
            st.session_state.camera_active = True
            
        if stop_button:
            st.session_state.camera_active = False
            camera_placeholder.empty()
        
        # Live camera processing
        if st.session_state.camera_active:
            st.info("🔴 Camera is active - Press 'Stop Camera' to end")
            
            # Initialize camera
            cap = cv2.VideoCapture(0)
            
            if not cap.isOpened():
                st.error("❌ Could not access camera. Please check your webcam.")
                st.session_state.camera_active = False
            else:
                # Camera processing loop
                frame_count = 0
                fps_counter = time.time()
                
                while st.session_state.camera_active:
                    ret, frame = cap.read()
                    
                    if not ret:
                        st.error("❌ Failed to read from camera")
                        break
                    
                    # Process frame for face detection
                    processed_frame, results = detector.process_frame(
                        frame, 
                        draw_predictions=True,
                        confidence_threshold=confidence_threshold
                    )
                    
                    # Convert BGR to RGB for Streamlit
                    processed_frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                    
                    # Display frame
                    camera_placeholder.image(
                        processed_frame_rgb, 
                        channels="RGB", 
                        use_column_width=True
                    )
                    
                    # Update statistics
                    frame_count += 1
                    
                    # Small delay to prevent overwhelming
                    time.sleep(0.033)  # ~30 FPS
                    
                    # Check if stop was pressed (streamlit limitation - manual check)
                    if frame_count % 30 == 0:  # Check every 30 frames
                        # Force a brief pause to allow Streamlit to process stop button
                        time.sleep(0.001)
                
                # Clean up camera
                cap.release()
                st.info("📹 Camera stopped")
    
    with col2:
        st.subheader("📊 Detection Info")
        
        # Model information
        st.info(f"**Model:** {selected_model}")
        
        # Age groups
        st.markdown("**Age Groups:**")
        st.markdown("""
        - 👶 Child (0-12)
        - 🧒 Teen (13-19)  
        - 👨 Young Adult (20-35)
        - 👨‍💼 Adult (36-55)
        - 👴 Senior (56+)
        """)
        
        # Gender classes
        st.markdown("**Gender Classes:**")
        st.markdown("""
        - 👩 Female
        - 👨 Male
        """)
        
        # Performance info
        st.success("""
        **Expected Accuracy:**
        - Gender: ~90-95%
        - Age Groups: ~80-85%
        """)
        
        # Instructions
        st.markdown("---")
        st.markdown("**Instructions:**")
        st.markdown("""
        1. Click 'Start Camera' to begin
        2. Position your face in view
        3. See real-time predictions
        4. Click 'Stop Camera' to end
        """)
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<p style='text-align: center; color: gray;'>"
        "🤖 Powered by TensorFlow & OpenCV"
        "</p>", 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()