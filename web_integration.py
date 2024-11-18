import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
from datetime import datetime
import matplotlib.pyplot as plt
import io
import zipfile
import os

# Function to create a downloadable ZIP file
def create_downloadable_zip(cropped_images):
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w") as zip_file:
        for idx, image in enumerate(cropped_images):
            # Convert the image to a format suitable for saving
            image_bytes = cv2.imencode('.png', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))[1].tobytes()
            # Write the image to the ZIP file
            zip_file.writestr(f"mango_{idx + 1}.png", image_bytes)
    zip_buffer.seek(0)
    return zip_buffer

# Load your YOLO models


model_path = os.path.join('models', 'extraction.pt')
model = YOLO(model_path)

variety_tracking_model_path = os.path.join('models', 'classification.pt')
variety_tracking_model = YOLO(variety_tracking_model_path)


# Streamlit app interface
st.title("Tree Classification and Mango Detection")
st.write("Upload an image to detect mangoes.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        # Load the uploaded image
        image = Image.open(uploaded_file)

        # Convert the image to a NumPy array and ensure it's in BGR format
        image_np = np.array(image)
        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

        # Perform inference
        variety_results = variety_tracking_model(image_np)
        results = model(image_np)

        # Display the variety classification
        for result in variety_results:
            variety_labelled = result.plot()
        
        variety_labelled_rgb = cv2.cvtColor(variety_labelled, cv2.COLOR_BGR2RGB)
        st.write("Level 1 Segmentation")
        st.image(variety_labelled_rgb, caption="Variety Classification", use_column_width=True)

        # Display mango detection
        for result in results:
            annotated_frame = result.plot()
        
        annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        st.write("Level 2 Segmentation")
        st.image(annotated_frame_rgb, caption="Mango Detection", use_column_width=True)

        # Extract and display cropped mangoes
        cropped_mangoes = []
        st.write("Individual Mango Crops:")
        padding = 10

        for result in results:
            img = np.copy(result.orig_img)
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                x1, y1, x2, y2 = max(0, x1 - padding), max(0, y1 - padding), min(img.shape[1], x2 + padding), min(img.shape[0], y2 + padding)
                mango_crop = img[y1:y2, x1:x2]
                cropped_mangoes.append(cv2.cvtColor(mango_crop, cv2.COLOR_BGR2RGB))

        # Display cropped mangoes in a grid
        if cropped_mangoes:
            num_cropped = len(cropped_mangoes)
            cols = 3
            rows = (num_cropped // cols) + (1 if num_cropped % cols > 0 else 0)
            fig, axes = plt.subplots(rows, cols, figsize=(12, 4 * rows))
            axes = axes.flatten()
            for i, cropped_mango in enumerate(cropped_mangoes):
                axes[i].imshow(cropped_mango)
                axes[i].axis('off')
                axes[i].set_title(f"Mango {i + 1}")
            for j in range(i + 1, len(axes)):
                axes[j].axis('off')
            st.pyplot(fig)

        # Create a downloadable ZIP file
        zip_buffer = create_downloadable_zip(cropped_mangoes)
        st.download_button(
            label="Download Cropped Mangoes",
            data=zip_buffer,
            file_name=f"cropped_mangoes_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
            mime="application/zip"
        )

    except Exception as e:
        st.error(f"Error during model inference: {e}")
