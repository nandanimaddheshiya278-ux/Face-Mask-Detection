# FaceGuard AI 🛡️
### Face Mask Detection using OpenCV + HOG + SVM

FaceGuard AI is a traditional Machine Learning and Computer Vision project that detects faces in uploaded images and predicts whether a person is wearing a face mask.

## Features
- Face detection using OpenCV.
- HOG feature extraction.
- SVM-based mask classification.
- Image upload through Streamlit.
- Detection results with labelled bounding boxes.
- Count of faces with and without masks.

## Technologies Used
- Python
- OpenCV
- HOG (Histogram of Oriented Gradients)
- Scikit-learn (SVM)
- Streamlit
- NumPy

## Project Structure
```text
face_mask_detection/
├── dataset/
│   ├── with_mask/
│   └── without_mask/
├── models/
├── app.py
├── app.ipynb
├── train_model.ipynb
├── haarcascade_frontalface_default.xml
├── requirements.txt
└── README.md
```

## Installation

Install the required libraries:

```bash
pip install -r requirements.txt
```

## Run the Application

```bash
streamlit run app.py
```

Open the local URL displayed in the terminal.

## Methodology
1. Prepare and preprocess the image.
2. Detect faces using OpenCV.
3. Extract HOG features.
4. Predict mask status using the trained SVM model.
5. Display the annotated image and detection counts.

## Dataset
The project uses a face mask image dataset with two categories:
- With Mask
- Without Mask

## Limitations
Predictions may vary depending on image quality, lighting, face angle, and dataset quality.

## Future Scope
- Improve detection accuracy.
- Support real-time webcam detection.
- Improve performance on multiple faces.
- Deploy the application online.

## Author
Kumari Nandani Maddheshiya  
B.Tech CSE (AI & ML)  
Bansal Institute of Engineering & Technology, Lucknow