import cvzone
import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import google.generativeai as genai
from PIL import Image
import streamlit as st

# Configure the Streamlit page layout for a wide display
st.set_page_config(layout="wide")
st.image('Math.png')  # Display a title image for the application

# Creating two columns for UI layout
col1, col2 = st.columns([3, 2])

# Left column: Handles video capture and processing
def initialize_ui():
    with col1:
        run = st.checkbox('Run', value=True)  # Checkbox to control webcam processing
        FRAME_WINDOW = st.image([])  # Placeholder for video frame rendering
    return run, FRAME_WINDOW

# Right column: Displays AI-generated output
with col2:
    st.title("Result:")
    output_text_area = st.subheader("")

# Configure Generative AI API
from dotenv import load_dotenv
import os

load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel('gemini-2.5-flash-lite')

# Initialize webcam for video capture
cap = cv2.VideoCapture(0)
cap.set(3, 1080)  # Increase resolution width
cap.set(4, 720)   # Increase resolution height
cap.set(cv2.CAP_PROP_FPS, 60)  # Increase frame rate

# Initialize HandDetector with optimized parameters
detector = HandDetector(
    staticMode=False,
    maxHands=1,
    modelComplexity=1,  # Keep 1 for real-time; use 2 if more accuracy is needed
    detectionCon=0.8,   # Increased confidence for better detection
    minTrackCon=0.7     # Higher confidence to ensure stable tracking
)

def getHandInfo(img):
    """Detects hands in the frame and returns finger state along with landmark coordinates."""
    hands, img = detector.findHands(img, draw=True, flipType=True)
    if hands:
        hand = hands[0]  # Process the first detected hand
        lmList = hand["lmList"]  # Extract landmark points
        fingers = detector.fingersUp(hand)  # Identify which fingers are raised
        return fingers, lmList
    return None

def draw(info, prev_pos, canvas, preview_canvas):
    """Handles drawing gestures and adds a cursor when necessary."""
    fingers, lmList = info
    current_pos = None

    # Drawing Mode (Index Finger Up)
    if fingers == [0, 1, 0, 0, 0]:
        current_pos = lmList[8][0:2]
        if prev_pos is None:
            prev_pos = current_pos
        cv2.line(canvas, prev_pos, current_pos, (255, 255, 0), 8)  # Yellow line for drawing

    # Erasing Mode (Three Fingers Up)
    elif fingers == [0, 1, 1, 1, 0]:
        current_pos = lmList[8][0:2]
        if prev_pos is None:
            prev_pos = current_pos
        cv2.line(canvas, prev_pos, current_pos, (0, 0, 0), 15)  # Eraser

    # Clear Canvas (Four Fingers Up)
    elif fingers == [0, 1, 1, 1, 1]:
        canvas[:] = 0  # Reset the canvas
        preview_canvas[:] = 0  # Reset the preview overlay
        return None, canvas, preview_canvas  # Reset tracking

    # Cursor Functionality: Show a dot when index + middle fingers are up or before erasing
    if fingers == [0, 1, 1, 0, 0] or fingers == [0, 1, 1, 1, 0]:
        cursor_pos = tuple(lmList[8][0:2])  # Use index fingertip position
        preview_canvas[:] = 0  # Clear previous preview
        cv2.circle(preview_canvas, cursor_pos, 10, (0, 255, 0), -1)  # Green cursor dot

    else:
        preview_canvas[:] = 0  # Clear preview if not in cursor mode

    return current_pos, canvas, preview_canvas

def sendToAI(model, canvas, fingers):
    """Sends drawn content to AI for processing when the thumb and pinky are extended."""
    if fingers == [1, 0, 0, 0, 1]:
        pil_image = Image.fromarray(canvas)
        response = model.generate_content([ "You are an AI that analyzes handwritten sketches of geometric shapes and mathematical equations.\n"

            "### If the image contains a shape:\n"

            "- *Object Name*\n"

            "- *Type (e.g., Right Triangle, Regular Polygon, etc.)*\n"

            "- *Mathematical Properties (only if applicable):*\n"

            "  - *Area, Perimeter, Side Lengths, Angles, Altitude*\n"

            "  - *Radius, Diameter, Circumference, Chords*\n"

            "  - *Center Coordinates, Inscribed/Circumscribed Circles*\n\n"

            "### If it's an equation:\n"

            "1. *Identify the type*\n"

            "2. *Rewrite it clearly*\n"

            "3. *Solve step by step*\n"

            "4. *Explain the reasoning*\n"

            "5. *Give the final answer*\n"

            "Note:If a figure contains conflicting dimensions (e.g., a square with sides labeled 3 and 4), treat it as a rough sketch and give precedence to the length."

            " *Only include applicable details—omit 'Not applicable' fields.*"

, pil_image])
        return response.text
    return ""

# Initialize variables for tracking drawing state
prev_pos = None
canvas = None
preview_canvas = None
image_combined = None
output_text = ""
alpha = 0.3  # Transparency level for overlay
run, FRAME_WINDOW = initialize_ui()

# Process webcam frames if 'Run' is checked
while run:
    success, img = cap.read()
    img = cv2.flip(img, 1)  # Flip image horizontally

    if canvas is None:
        canvas = np.zeros_like(img)
    if preview_canvas is None:
        preview_canvas = np.zeros_like(img)

    info = getHandInfo(img)
    if info:
        fingers, lmList = info
        prev_pos, canvas, preview_canvas = draw(info, prev_pos, canvas, preview_canvas)
        output_text = sendToAI(model, canvas, fingers)
        alpha = 0.05  # Make canvas black when hand is detected
    else:
        alpha = 0.5  # Reset transparency when no hand is detected
        preview_canvas[:] = 0  # Clear preview when no hand is detected

    # Combine original frame with drawing overlay
    image_combined = cv2.addWeighted(img, alpha, canvas, 1, 0)
    image_combined = cv2.addWeighted(image_combined, 1, preview_canvas, 0.95, 0)  # Overlay preview
    FRAME_WINDOW.image(image_combined, channels="BGR")

    if output_text:
        output_text_area.text(output_text)

    cv2.waitKey(1)
