# =========================================================
# REAL-TIME OBJECT RECOGNITION SYSTEM
# Raspberry Pi 5 Optimized
#
# Features:
# - YOLOv8 Nano Object Detection
# - OpenCV Camera Stream
# - Free LLM API
# - Offline Text To Speech
# - Error Handling
#
# Press Q to Exit
# =========================================================

# =========================
# IMPORT REQUIRED LIBRARIES
# =========================

import cv2
import requests
import pyttsx3

from ultralytics import YOLO

# =========================================================
# LOAD YOLOv8 NANO MODEL
# =========================================================
# This model is responsible for object detection.
#
# Possible Problems:
# 1) yolov8n.pt file missing
# 2) corrupted model file
#
# Solution:
# Download yolov8n.pt again and place it
# in the same folder as this script.
# =========================================================

try:

    model = YOLO("yolov8n.pt")

    print("[INFO] YOLOv8 model loaded successfully")

except Exception as e:

    print(f"[MODEL ERROR] {e}")

    print("Make sure yolov8n.pt exists")

    exit()

# =========================================================
# INITIALIZE TEXT TO SPEECH ENGINE
# =========================================================
# pyttsx3 works offline.
#
# Possible Problems:
# 1) No audio output
# 2) Audio driver issues
#
# Solution:
# Check speaker/headphone connection.
# =========================================================

try:

    engine = pyttsx3.init()

    # Speech speed
    engine.setProperty('rate', 150)

    # Volume level
    engine.setProperty('volume', 1.0)

    print("[INFO] TTS engine initialized")

except Exception as e:

    print(f"[TTS INIT ERROR] {e}")

    exit()

# =========================================================
# FREE LLM API CONFIGURATION
# =========================================================
# This API generates educational responses.
#
# Free API:
# https://openrouter.ai/
#
# Possible Problems:
# 1) Invalid API key
# 2) No internet connection
# 3) API rate limit exceeded
#
# Solution:
# - Check API key
# - Check internet
# - Wait and retry
# =========================================================

API_KEY = "YOUR_API_KEY"

API_URL = "https://openrouter.ai/api/v1/chat/completions"

MODEL_NAME = "mistralai/mistral-7b-instruct:free"

# =========================================================
# OBJECT DETECTION FUNCTION
# =========================================================
# This function:
# 1) Takes a frame from camera
# 2) Runs YOLO detection
# 3) Filters weak detections
# 4) Returns detected objects
#
# Possible Problems:
# 1) Slow FPS
# 2) Detection lag
#
# Solution:
# - Reduce imgsz
# - Process fewer frames
# =========================================================

def detect_objects(frame, conf_threshold=0.5):

    detected_objects = []

    try:

        # Run YOLO inference
        # imgsz=320 is optimized for Raspberry Pi 5
        results = model(frame, imgsz=320)

        # Loop through all results
        for result in results:

            # Loop through detected boxes
            for box in result.boxes:

                # Get confidence score
                confidence = float(box.conf[0])

                # Ignore weak detections
                if confidence >= conf_threshold:

                    # Get class ID
                    class_id = int(box.cls[0])

                    # Convert class ID to class name
                    class_name = model.names[class_id]

                    # Add object name to list
                    detected_objects.append(class_name)

        # Remove duplicates
        detected_objects = list(set(detected_objects))

        return detected_objects

    except Exception as e:

        print(f"[DETECTION ERROR] {e}")

        return []

# =========================================================
# PROMPT BUILDER FUNCTION
# =========================================================
# This function creates ONE prompt
# containing ALL detected objects.
#
# Why?
# - Faster
# - Better for LLM
# - Reduces API usage
# =========================================================

def build_prompt(objects):

    try:

        # Convert list to readable text
        objects_text = ", ".join(objects)

        # Create educational prompt
        prompt = f"""
You are a helpful educational AI assistant.

Explain these detected objects simply for students.

Objects:
{objects_text}

For each object explain:
1. What it is
2. What it is used for
3. One interesting fact

Keep the explanation short and simple.
"""

        return prompt

    except Exception as e:

        print(f"[PROMPT ERROR] {e}")

        return None

# =========================================================
# LLM RESPONSE FUNCTION
# =========================================================
# Sends prompt to free AI API.
#
# Possible Problems:
# 1) API timeout
# 2) Invalid response
# 3) No internet
#
# Solution:
# - Retry later
# - Check internet connection
# =========================================================

def generate_response(prompt):

    try:

        # API request headers
        headers = {
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json"
        }

        # Request body
        data = {
            "model": MODEL_NAME,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        }

        # Send POST request
        response = requests.post(
            API_URL,
            headers=headers,
            json=data,
            timeout=30
        )

        # Convert response to JSON
        result = response.json()

        # Extract AI response text
        ai_response = result["choices"][0]["message"]["content"]

        return ai_response

    except Exception as e:

        print(f"[LLM ERROR] {e}")

        return "Sorry, I could not generate a response."

# =========================================================
# TEXT TO SPEECH FUNCTION
# =========================================================
# Converts AI text response into speech.
#
# Possible Problems:
# 1) Long speech delay
# 2) Audio overlap
#
# Solution:
# - Use shorter responses
# - Reduce processing frequency
# =========================================================

def speak_text(text):

    try:

        engine.say(text)

        engine.runAndWait()

    except Exception as e:

        print(f"[TTS ERROR] {e}")

# =========================================================
# MAIN SYSTEM FUNCTION
# =========================================================
# This is the main controller.
#
# System Flow:
#
# Camera
#   ↓
# Frame
#   ↓
# YOLO Detection
#   ↓
# Prompt Builder
#   ↓
# AI Response
#   ↓
# Text To Speech
# =========================================================

def main():

    print("\n===================================")
    print("OBJECT RECOGNITION SYSTEM STARTED")
    print("===================================\n")

    # =========================
    # OPEN CAMERA
    # =========================
    # 0 = default webcam
    #
    # Possible Problems:
    # 1) Camera not connected
    # 2) Camera busy
    #
    # Solution:
    # - Reconnect camera
    # - Close other camera apps
    # =========================

    cap = cv2.VideoCapture(0)

    # Check camera status
    if not cap.isOpened():

        print("[CAMERA ERROR] Cannot open camera")

        return

    print("[INFO] Camera started successfully")

    print("[INFO] Press Q to quit\n")

    # Frame counter
    frame_count = 0

    # Store previous objects
    # Prevent repeated speaking
    last_objects = []

    # =====================================================
    # MAIN LOOP
    # =====================================================

    while True:

        # Read frame from camera
        ret, frame = cap.read()

        # Check frame status
        if not ret:

            print("[FRAME ERROR] Failed to read frame")

            break

        # Increase frame counter
        frame_count += 1

        # =================================================
        # PROCESS EVERY 5TH FRAME
        # =================================================
        # Why?
        # Reduces CPU usage on Raspberry Pi 5
        # =================================================

        if frame_count % 5 == 0:

            # =============================================
            # OBJECT DETECTION
            # =============================================

            objects = detect_objects(frame)

            # Skip if no objects found
            if not objects:
                continue

            # Avoid repeated speaking
            if objects == last_objects:
                continue

            # Save current objects
            last_objects = objects

            print("\n===================================")

            print("[DETECTED OBJECTS]")

            print(objects)

            # =============================================
            # BUILD PROMPT
            # =============================================

            prompt = build_prompt(objects)

            print("\n[PROMPT GENERATED]")

            print(prompt)

            # =============================================
            # GENERATE AI RESPONSE
            # =============================================

            response = generate_response(prompt)

            print("\n[AI RESPONSE]")

            print(response)

            # =============================================
            # SPEAK RESPONSE
            # =============================================

            speak_text(response)

        # =================================================
        # SHOW CAMERA WINDOW
        # =================================================

        cv2.imshow("Object Recognition System", frame)

        # =================================================
        # EXIT SYSTEM
        # =================================================
        # Press Q to quit
        # =================================================

        if cv2.waitKey(1) & 0xFF == ord('q'):

            print("\n[INFO] Exiting system...")

            break

    # =====================================================
    # CLEANUP
    # =====================================================

    cap.release()

    cv2.destroyAllWindows()

    print("[INFO] System closed successfully")

# =========================================================
# START PROGRAM
# =========================================================

if __name__ == "__main__":

    main()