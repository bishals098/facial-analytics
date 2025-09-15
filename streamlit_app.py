import streamlit as st
import cv2
import numpy as np
import os
import time
import torch
from src.detection import FaceDetectionPipeline

# Page configuration
st.set_page_config(
    page_title="Live Age & Gender Detection - PyTorch",
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
    """Load the PyTorch face detection model"""
    try:
        detector = FaceDetectionPipeline(model_path=model_path)
        return detector
    except Exception as e:
        st.error(f"Error loading PyTorch model: {e}")
        return None

def get_device_info():
    """Get device information for display"""
    if torch.cuda.is_available():
        device = f"GPU: {torch.cuda.get_device_name()}"
        gpu_memory = f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB"
        return device, gpu_memory
    else:
        return "CPU", "No GPU detected"

def main():
    """Main Streamlit application - Live Camera Only with PyTorch"""
    
    # Header
    st.title("🎯 Live Age & Gender Detection")
    st.markdown("**🔥 Real-time face detection powered by PyTorch**")
    
    # Device info
    device, memory = get_device_info()
    st.markdown(f"**Device:** {device} | **{memory}**")

    # Sidebar for model selection and settings
    st.sidebar.header("⚙️ Configuration")
    
    # Model selection - Look for PyTorch models
    models_dir = 'models'
    if os.path.exists(models_dir):
        # Look for both .pth and .keras files for transition period
        pytorch_models = [f for f in os.listdir(models_dir) if f.endswith('.pth')]
        keras_models = [f for f in os.listdir(models_dir) if f.endswith('.keras')]
        
        model_files = pytorch_models + keras_models
    else:
        model_files = []

    if not model_files:
        st.error("❌ No trained model found!")
        st.info("💡 Train a model first using: `python main.py`")
        st.info("📋 Expected model formats: `.pth` or `.keras`")
        return

    # Show model types
    if pytorch_models:
        st.sidebar.success(f"🔥 Found {len(pytorch_models)} PyTorch model(s)")
    if keras_models:
        st.sidebar.warning(f"📄 Found {len(keras_models)} legacy Keras model(s)")

    selected_model = st.sidebar.selectbox("Select Model", model_files)
    model_path = os.path.join(models_dir, selected_model)
    
    # Show model type
    if selected_model.endswith('.pth'):
        st.sidebar.info("🔥 **PyTorch Model** (Recommended)")
    else:
        st.sidebar.warning("📄 **Keras Model** (Consider upgrading)")

    # Load detector
    if st.session_state.detector is None or st.sidebar.button("🔄 Reload Model"):
        with st.spinner("Loading model..."):
            st.session_state.detector = load_detector(model_path)

        if st.session_state.detector is None:
            st.error("Failed to load model")
            return
        else:
            model_info = st.session_state.detector.get_model_info()
            if isinstance(model_info, dict):
                st.sidebar.success(f"✅ {model_info['framework']} model loaded!")
                st.sidebar.info(f"Device: {model_info['device']}")
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
        step=0.05,
        help="Higher values = more confident predictions only"
    )

    # Performance settings
    st.sidebar.subheader("⚡ Performance")
    fps_target = st.sidebar.selectbox(
        "Target FPS",
        [15, 20, 25, 30],
        index=2,  # Default to 25 FPS
        help="Lower FPS = better performance on slower devices"
    )
    
    frame_delay = 1.0 / fps_target

    # Camera controls
    st.sidebar.subheader("📹 Camera Controls")

    # Main content area
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("📹 Live Camera Feed")

        # Camera control buttons
        camera_col1, camera_col2 = st.columns(2)
        
        with camera_col1:
            start_button = st.button("🎥 Start Camera", type="primary", use_container_width=True)
        
        with camera_col2:
            stop_button = st.button("⏹️ Stop Camera", use_container_width=True)

        # Placeholder for camera feed
        camera_placeholder = st.empty()
        status_placeholder = st.empty()

        if start_button:
            st.session_state.camera_active = True

        if stop_button:
            st.session_state.camera_active = False
            camera_placeholder.empty()
            status_placeholder.empty()

        # Live camera processing
        if st.session_state.camera_active:
            status_placeholder.info("🔴 Camera is active - Press 'Stop Camera' to end")

            # Initialize camera
            cap = cv2.VideoCapture(0)

            if not cap.isOpened():
                st.error("❌ Could not access camera. Please check your webcam.")
                st.session_state.camera_active = False
            else:
                # Camera processing loop
                frame_count = 0
                total_faces = 0
                fps_counter = time.time()
                fps_display = 0

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

                    # Add performance info to frame
                    if frame_count % 30 == 0:  # Update FPS every 30 frames
                        current_time = time.time()
                        if frame_count > 0:
                            fps_display = 30 / (current_time - fps_counter)
                        fps_counter = current_time

                    # Add FPS and face count overlay
                    cv2.putText(processed_frame_rgb, 
                              f"FPS: {fps_display:.1f} | Faces: {len(results)}", 
                              (10, processed_frame_rgb.shape[0] - 50),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                    # Display frame
                    camera_placeholder.image(
                        processed_frame_rgb, 
                        channels="RGB", 
                        use_container_width=True
                    )

                    # Update statistics
                    frame_count += 1
                    total_faces += len(results)

                    # Control frame rate
                    time.sleep(frame_delay)

                    # Check if stop was pressed (Streamlit limitation - periodic check)
                    if frame_count % 15 == 0:  # Check every 15 frames
                        time.sleep(0.001)  # Allow Streamlit to process stop button

                # Clean up camera
                cap.release()
                status_placeholder.success(f"📹 Camera stopped | Processed {frame_count} frames | Found {total_faces} faces")

    with col2:
        st.subheader("📊 Detection Info")

        # Model information
        st.info(f"**Model:** {selected_model}")
        
        if hasattr(detector, 'device') and detector.device:
            device_name = str(detector.device).upper()
            st.info(f"**Device:** {device_name}")

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
        2. Position your face clearly in view
        3. See real-time predictions with confidence
        4. Adjust confidence threshold if needed
        5. Click 'Stop Camera' to end
        """)

        # Performance tips
        st.markdown("---")
        st.markdown("**Performance Tips:**")
        st.markdown("""
        - 🔥 Use GPU for faster inference
        - 💡 Good lighting improves accuracy
        - 📐 Face should be clearly visible
        - 🎯 Higher confidence = more accurate
        """)

    # Footer
    st.markdown("---")
    
    footer_col1, footer_col2, footer_col3 = st.columns(3)
    
    with footer_col1:
        st.markdown("🔥 **Powered by PyTorch**")
    
    with footer_col2:
        st.markdown("📷 **OpenCV Vision**") 
        
    with footer_col3:
        if torch.cuda.is_available():
            st.markdown("⚡ **GPU Accelerated**")
        else:
            st.markdown("💻 **CPU Processing**")

if __name__ == "__main__":
    main()