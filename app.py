import streamlit as st
import cv2
import numpy as np
import joblib
import os
from PIL import Image
from skimage.feature import hog

st.set_page_config(
    page_title="FaceGuard AI",
    page_icon="😷",
    layout="wide"
)

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #0b1220, #172d49);
    color: white;
}
h1 { color: #62d5ff; }
</style>
""", unsafe_allow_html=True)

st.title("😷 FaceGuard AI")
st.write("Face Mask Detection using OpenCV + HOG + SVM")
st.caption("Traditional Machine Learning Project")
st.divider()

MODEL_PATH = "models/mask_svm.pkl"
CASCADE_PATH = "haarcascade_frontalface_default.xml"

if not os.path.exists(MODEL_PATH):
    st.error("Model file nahi mili: models/mask_svm.pkl")
    st.stop()

if not os.path.exists(CASCADE_PATH):
    st.error("Haar Cascade XML file nahi mili.")
    st.stop()

@st.cache_resource
def load_model_and_detector():
    model = joblib.load(MODEL_PATH)
    detector = cv2.CascadeClassifier(CASCADE_PATH)
    if detector.empty():
        raise ValueError("Face detector load nahi hua.")
    return model, detector

try:
    model, detector = load_model_and_detector()
except Exception as e:
    st.error(f"Loading error: {e}")
    st.stop()

st.sidebar.header("About this project")
st.sidebar.write("1. Upload a face image")
st.sidebar.write("2. OpenCV detects faces")
st.sidebar.write("3. HOG extracts features")
st.sidebar.write("4. SVM predicts the class")

uploaded = st.file_uploader(
    "Upload a clear face image",
    type=["jpg", "jpeg", "png"]
)

if uploaded:
    pil_image = Image.open(uploaded).convert("RGB")
    rgb = np.array(pil_image)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    faces = detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=6,
        minSize=(60, 60)
    )

    result = rgb.copy()
    mask_count = 0
    no_mask_count = 0

    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (64, 64))

        features = hog(
            face,
            orientations=9,
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2),
            block_norm="L2-Hys"
        ).reshape(1, -1)

        prediction = int(model.predict(features)[0])

        if prediction == 1:
            label = "WITH MASK"
            color = (40, 220, 100)
            mask_count += 1
        else:
            label = "WITHOUT MASK"
            color = (255, 70, 70)
            no_mask_count += 1

        cv2.rectangle(result, (x, y), (x+w, y+h), color, 3)
        cv2.putText(
            result, label, (x, max(y-10, 25)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2
        )

    left, right = st.columns(2)

    with left:
        st.subheader("Original Image")
        st.image(rgb, use_container_width=True)

    with right:
        st.subheader("Detection Result")
        st.image(result, use_container_width=True)

    st.divider()
    a, b, c = st.columns(3)
    a.metric("Faces Detected", len(faces))
    b.metric("With Mask", mask_count)
    c.metric("Without Mask", no_mask_count)

    if len(faces) == 0:
        st.warning("Face detect nahi hua. Clear, front-facing photo try karo.")
    else:
        st.success("Image processing complete!")

    st.info(
        "Note: Predictions are estimates. Model performance may vary "
        "with image quality, face angle and lighting."
    )
else:
    st.info("Upload an image to start detection.")
