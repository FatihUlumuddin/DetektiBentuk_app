# -*- coding: utf-8 -*-
import streamlit as st
import cv2
import numpy as np
from PIL import Image

def detect_shapes(image):
    """
    Detect shapes in the provided image.
    Draw bounding boxes around shapes and label them.
    """
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 0.04 * cv2.arcLength(contour, True), True)
        x, y, w, h = cv2.boundingRect(approx)

        # Detect shape based on the number of vertices
        if len(approx) == 3:
            shape = "Triangle"
        elif len(approx) == 4:
            shape = "Rectangle"
        elif len(approx) > 4:
            shape = "Circle"
        else:
            shape = "Unknown"

        # Draw the shape and label on the image
        cv2.putText(image, shape, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return image

# Streamlit app
st.set_page_config(
    page_title="Shape Detection App",
    layout="wide",  # Wide layout for responsiveness
)

# Header
st.title("Shape Detection Web App")
st.markdown("Upload an image to detect shapes, such as circles, rectangles, or triangles.")

# Sidebar for additional options
with st.sidebar:
    st.header("Settings")
    st.info("Use this sidebar for additional detection settings.")

# Main content layout
col1, col2 = st.columns([1, 1])

# File uploader in the first column
with col1:
    uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

# Process and display images
if uploaded_file:
    # Read the uploaded image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    # Convert the image to RGB for Streamlit display
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Display the uploaded image
    with col1:
        st.subheader("Uploaded Image")
        st.image(image_rgb, caption="Original Image", use_column_width=True)

    # Detect shapes
    processed_image = detect_shapes(image)

    # Convert processed image to RGB for Streamlit display
    processed_image_rgb = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)

    # Display the processed image
    with col2:
        st.subheader("Processed Image")
        st.image(processed_image_rgb, caption="Image with Detected Shapes", use_column_width=True)
else:
    with col2:
        st.info("Upload an image to start detecting shapes.")
