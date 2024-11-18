import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
from datetime import datetime
import io
import zipfile
import os

# Function to create a downloadable ZIP file
def create_downloadable_zip(cropped_images):
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w") as zip_file:
        for idx, image in enumerate(cropped_images):
            image_bytes = cv2.imencode('.png', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))[1].tobytes()
            zip_file.writestr(f"mango_{idx + 1}.png", image_bytes)
    zip_buffer.seek(0)
    return zip_buffer

# Load your YOLO models
model_path = os.path.join('models', 'extraction.pt')
model = YOLO(model_path)

variety_tracking_model_path = os.path.join('models', 'classification.pt')
variety_tracking_model = YOLO(variety_tracking_model_path)

# Streamlit sidebar
st.sidebar.title("About This Tool")
st.sidebar.markdown(
    """
    ### 🍃 Tree Classification and Mango Detection  
    This tool uses Yolo v11 models to perform:  
    - **Tree Variety Classification**  
    - **Mango Detection and Cropping**  
    
    Upload an image to analyze the type of tree and identify individual mangoes.  
    Processed images and cropped mangoes can be downloaded for further analysis.  
    
    """
)
st.sidebar.info("Use the main interface to upload an image and run the analysis.")

# Streamlit app interface
st.title("🍃 Tree Classification and Mango Detection 🍃")
st.markdown(
    """
    Welcome to the Tree Classification and Mango Detection tool!  
    Upload an image to classify tree varieties and detect mangoes. 
    """
)
st.divider()

# File uploader
uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])
if uploaded_file:
    st.subheader("Uploaded Image")
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    st.divider()
    if st.button("Run Detection"):
        try:
            # Convert the image to a NumPy array and ensure it's in BGR format
            image_np = np.array(image)
            image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

            # Perform inference
            with st.spinner("Classifying tree variety..."):
                variety_results = variety_tracking_model(image_np)
            with st.spinner("Detecting mangoes..."):
                results = model(image_np)

            # Display the variety classification
            st.subheader("🌳 Level 1: Tree Variety Classification")
            for result in variety_results:
                variety_labelled = result.plot()
            variety_labelled_rgb = cv2.cvtColor(variety_labelled, cv2.COLOR_BGR2RGB)
            st.image(variety_labelled_rgb, caption="Variety Classification", use_column_width=True)

            # Display mango detection
            st.subheader("🥭 Level 2: Mango Detection")
            for result in results:
                annotated_frame = result.plot()
            annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            st.image(annotated_frame_rgb, caption="Mango Detection", use_column_width=True)

            # Extract and display cropped mangoes
            cropped_mangoes = []
            padding = 10
            st.subheader("📸 Individual Mango Crops")
            for result in results:
                img = np.copy(result.orig_img)
                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    x1, y1, x2, y2 = max(0, x1 - padding), max(0, y1 - padding), min(img.shape[1], x2 + padding), min(img.shape[0], y2 + padding)
                    mango_crop = img[y1:y2, x1:x2]
                    cropped_mangoes.append(cv2.cvtColor(mango_crop, cv2.COLOR_BGR2RGB))

            if cropped_mangoes:
                cols = st.columns(3)
                for idx, cropped_mango in enumerate(cropped_mangoes):
                    with cols[idx % 3]:
                        st.image(cropped_mango, caption=f"Mango {idx + 1}", use_column_width=True)

            # Create a downloadable ZIP file
            zip_buffer = create_downloadable_zip(cropped_mangoes)
            st.download_button(
                label=f"📥 Download {len(cropped_mangoes)} Cropped Mangoes",
                data=zip_buffer,
                file_name=f"cropped_mangoes_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
                mime="application/zip"
            )
        except Exception as e:
            st.error(f"An error occurred: {e}")
