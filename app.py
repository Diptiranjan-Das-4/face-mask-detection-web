import os
import cv2
import numpy as np

from flask import Flask, render_template, request, jsonify, Response
from tensorflow.keras.models import load_model


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "mask_detector_model.h5")


CASCADE_PATH = os.path.join(
    BASE_DIR,
    "haarcascade_frontalface_default.xml"
)


# ============================================================
# SETTINGS
# ============================================================

IMG_SIZE = 250


# ============================================================
# CHECK REQUIRED FILES
# ============================================================

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        "mask_detector_model.h5 was not found.\n"
        f"Expected location: {MODEL_PATH}"
    )

if not os.path.exists(CASCADE_PATH):
    raise FileNotFoundError(
        "haarcascade_frontalface_default.xml was not found.\n"
        f"Expected location: {CASCADE_PATH}"
    )


# ============================================================
# LOAD MODEL
# ============================================================

print("Loading mask detection model...")

model = load_model(MODEL_PATH)

print("Mask detection model loaded successfully.")


# ============================================================
# LOAD HAAR CASCADE
# ============================================================

face_cascade = cv2.CascadeClassifier(CASCADE_PATH)

if face_cascade.empty():
    raise RuntimeError(
        "Could not load haarcascade_frontalface_default.xml"
    )

print("Haar Cascade loaded successfully.")


# ============================================================
# FLASK APPLICATION
# ============================================================

app = Flask(__name__)


# ============================================================
# FACE MASK DETECTION
# ============================================================

def detect_mask(frame):

    # Make a copy so we don't modify the original frame
    output = frame.copy()
    

    # --------------------------------------------------------
    # Convert image to grayscale for Haar Cascade
    # --------------------------------------------------------

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    # --------------------------------------------------------
    # Detect faces
    # --------------------------------------------------------

    faces = face_cascade.detectMultiScale(
    gray,
    scaleFactor=1.05,
    minNeighbors=3,
    minSize=(50, 50)
)

    detections = []

    # --------------------------------------------------------
    # Process every detected face
    # --------------------------------------------------------

    for (x, y, w, h) in faces:

        # Crop face
        face = frame[
            y:y + h,
            x:x + w
        ]

        if face.size == 0:
            continue

        # ----------------------------------------------------
        # Resize to the same size used by your notebook
        # ----------------------------------------------------

        face = cv2.resize(
            face,
            (IMG_SIZE, IMG_SIZE)
        )

        # ----------------------------------------------------
        # Normalize
        # Same as your notebook:
        #
        # face = face / 255.0
        # ----------------------------------------------------

        face = face / 255.0

        # ----------------------------------------------------
        # Reshape for CNN
        # ----------------------------------------------------

        face = np.reshape(
            face,
            (1, IMG_SIZE, IMG_SIZE, 3)
        )

        # ----------------------------------------------------
        # Prediction
        # ----------------------------------------------------

        prediction = model.predict(
            face,
            verbose=0
        )[0][0]

        prediction = float(prediction)

        # ----------------------------------------------------
        # Classification
        #
        # Your notebook:
        #
        # pred > 0.5 -> Mask
        # pred <= 0.5 -> No Mask
        # ----------------------------------------------------

        if prediction > 0.5:
            label = "Mask"
            confidence = prediction
            color = (0, 255, 0)

        else:
            label = "No Mask"
            confidence = 1.0 - prediction
            color = (0, 0, 255)

        # ----------------------------------------------------
        # Convert confidence to percentage
        # ----------------------------------------------------

        confidence_percentage = confidence * 100

        # ----------------------------------------------------
        # Draw bounding box
        # ----------------------------------------------------

        cv2.rectangle(
            output,
            (x, y),
            (x + w, y + h),
            color,
            2
        )

        # ----------------------------------------------------
        # Label
        # ----------------------------------------------------

        text = (
            f"{label} "
            f"({confidence_percentage:.1f}%)"
        )

        text_y = max(
            y - 10,
            25
        )

        cv2.putText(
            output,
            text,
            (x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2
        )

        # Store detection information
        detections.append({
            "label": label,
            "confidence": round(
                confidence_percentage,
                1
            ),
            "box": [
                int(x),
                int(y),
                int(w),
                int(h)
            ]
        })

    return output, detections


# ============================================================
# HOME PAGE
# ============================================================

@app.route("/")
def home():

    return render_template(
        "index.html"
    )


# ============================================================
# PREDICTION API
# ============================================================

@app.route(
    "/predict",
    methods=["POST"]
)
def predict():

    # Check if frame exists
    if "frame" not in request.files:

        return jsonify({
            "error": "No frame received."
        }), 400

    # --------------------------------------------------------
    # Read uploaded frame
    # --------------------------------------------------------

    file = request.files["frame"]

    image_bytes = file.read()

    # Convert bytes to NumPy array
    np_array = np.frombuffer(
        image_bytes,
        np.uint8
    )

    # Decode image
    frame = cv2.imdecode(
        np_array,
        cv2.IMREAD_COLOR
    )

    if frame is None:

        return jsonify({
            "error": "Could not decode image."
        }), 400

    # --------------------------------------------------------
    # Detect mask
    # --------------------------------------------------------

    output, detections = detect_mask(frame)

    # --------------------------------------------------------
    # Encode processed frame as JPEG
    # --------------------------------------------------------

    success, encoded_image = cv2.imencode(
        ".jpg",
        output,
        [
            cv2.IMWRITE_JPEG_QUALITY,
            85
        ]
    )

    if not success:

        return jsonify({
            "error": "Could not encode image."
        }), 500

    # --------------------------------------------------------
    # Send JPEG back to browser
    # --------------------------------------------------------

    response = Response(
        encoded_image.tobytes(),
        mimetype="image/jpeg"
    )

    return response


# ============================================================
# RUN SERVER
# ============================================================

if __name__ == "__main__":

    print("")
    print("======================================")
    print(" FACE MASK DETECTION WEB APPLICATION")
    print("======================================")
    print("")
    print("Open this URL in your browser:")
    print("http://127.0.0.1:5000")
    print("")

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )