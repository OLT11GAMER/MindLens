import face_recognition
import cv2
import numpy as np
import pyttsx3
import json
import csv
import os
import time
from datetime import datetime
from tkinter import Tk, Button, Label, messagebox
import threading
import speech_recognition as sr
import uuid
import queue # New import for thread-safe queue

# Ensure 'faces' directory exists
if not os.path.exists("faces"):
    os.makedirs("faces")

# ----------------------------
# GLOBAL CONSTANTS (Moved to top for global access)
# ----------------------------
WINDOW_NAME = "Alzheimer Face Assistant"
CORNER_RADIUS = 15 # For rounded rectangles

# ----------------------------
# HELPER FUNCTION TO DRAW ROUNDED RECTANGLE (Moved here for global access)
# ----------------------------
def draw_rounded_rectangle(image, top_left, bottom_right, color, thickness, radius, filled=False):
    x1, y1 = top_left
    x2, y2 = bottom_right
    
    width = x2 - x1
    height = y2 - y1

    # Ensure radius is not too large for the rectangle dimensions
    radius = min(radius, width // 2, height // 2)
    if radius < 0: # Ensure non-negative radius
        radius = 0
    
    # Draw the main rectangle parts
    if filled:
        cv2.rectangle(image, (x1 + radius, y1), (x2 - radius, y2), color, -1)
        cv2.rectangle(image, (x1, y1 + radius), (x2, y2 - radius), color, -1)
    else:
        cv2.line(image, (x1 + radius, y1), (x2 - radius, y1), color, thickness) # Top
        cv2.line(image, (x1 + radius, y2), (x2 - radius, y2), color, thickness) # Bottom
        cv2.line(image, (x1, y1 + radius), (x1, y2 - radius), color, thickness) # Left
        cv2.line(image, (x2, y1 + radius), (x2, y2 - radius), color, thickness) # Right

    # Draw circles for corners
    if radius > 0: # Only draw ellipses/circles if radius is positive
        if filled:
            cv2.circle(image, (x1 + radius, y1 + radius), radius, color, -1)
            cv2.circle(image, (x2 - radius, y1 + radius), radius, color, -1)
            cv2.circle(image, (x1 + radius, y2 - radius), radius, color, -1)
            cv2.circle(image, (x2 - radius, y2 - radius), radius, color, -1)
        else:
            cv2.ellipse(image, (x1 + radius, y1 + radius), (radius, radius), 180, 0, 90, color, thickness)
            cv2.ellipse(image, (x2 - radius, y1 + radius), (radius, radius), 270, 0, 90, color, thickness)
            cv2.ellipse(image, (x1 + radius, y2 - radius), (radius, radius), 90, 0, 90, color, thickness)
            cv2.ellipse(image, (x2 - radius, y2 - radius), (radius, radius), 0, 0, 90, color, thickness)


# ----------------------------
# INIT TEXT-TO-SPEECH (Centralized for multi-threading)
# ----------------------------
_tts_engine = None
_tts_queue = queue.Queue()
_tts_thread = None
_tts_thread_active = False

def _tts_worker():
    """Dedicated thread worker for Text-to-Speech."""
    global _tts_engine, _tts_thread_active
    try:
        # Initialize engine once at the start of the worker thread
        _tts_engine = pyttsx3.init()
        _tts_engine.setProperty('rate', 160)
        print("[TTS_WORKER] TTS engine initialized.")
        
        while _tts_thread_active:
            try:
                # Get text and event from the queue, blocking indefinitely until an item is available
                text, event = _tts_queue.get(block=True, timeout=None) 
                if text == "STOP_SIGNAL": # Check for stop signal
                    break # Exit loop
                
                print(f"[TTS_WORKER] Speaking: {text}")
                _tts_engine.say(text)
                _tts_engine.runAndWait() # This processes the current queue
                print(f"[TTS_WORKER] Finished speaking: {text}")

                # After speaking, explicitly stop and re-initialize the engine for robustness
                # This ensures a clean state for the next utterance and prevents engine becoming unresponsive
                _tts_engine.stop()
                # Add a small sleep here to allow resources to be fully released
                time.sleep(0.05) # Small delay (e.g., 50 milliseconds)
                _tts_engine = pyttsx3.init()
                _tts_engine.setProperty('rate', 160)
                
                if event: # Only set event if one was provided (for blocking calls)
                    event.set() 
            except queue.Empty:
                # This block should ideally not be reached with block=True, but included for robustness.
                continue
            except Exception as e:
                print(f"[TTS_WORKER_ERROR] Error during speech: {e}")
                if event: # If an event was provided, set it even on error to unblock caller
                    event.set()
                # If an error occurs, try to re-initialize the engine to recover
                print("[TTS_WORKER_RECOVERY] Attempting to recover TTS engine after error.")
                if _tts_engine:
                    try:
                        _tts_engine.stop()
                        del _tts_engine # Explicitly delete to force cleanup
                        _tts_engine = None # Mark for re-initialization
                        time.sleep(0.1) # Give a small delay for resource release
                    except Exception as stop_e:
                        print(f"[TTS_WORKER_RECOVERY_ERROR] Error stopping old engine: {stop_e}")
                try:
                    _tts_engine = pyttsx3.init()
                    _tts_engine.setProperty('rate', 160)
                    print("[TTS_WORKER_RECOVERY] TTS engine re-initialized successfully.")
                except Exception as init_e:
                    print(f"[TTS_WORKER_RECOVERY_FATAL] Failed to re-initialize TTS engine: {init_e}")
                    _tts_engine = None # Still failed, keep None
    except Exception as e:
        print(f"[TTS_WORKER_FATAL] Could not initialize TTS engine in worker: {e}")
        messagebox.showerror("TTS Error", f"Could not initialize Text-to-Speech engine: {e}")
    finally:
        if _tts_engine:
            try:
                _tts_engine.stop()
                print("[TTS_WORKER] TTS engine stopped.")
            except Exception as e:
                print(f"[TTS_WORKER_WARNING] Error stopping TTS engine: {e}")

def speak_and_wait(text, wait_for_completion=True): # Added wait_for_completion
    """Speaks the given text and waits for it to finish."""
    global last_spoken_text
    temp_speaker = None
    try:
        temp_speaker = pyttsx3.init()
        temp_speaker.setProperty('rate', 160)
        print(f"[TTS] Speaking: {text}")
        temp_speaker.say(text)
        temp_speaker.runAndWait()
        print(f"[TTS] Finished speaking: {text}")
        last_spoken_text = text
    except Exception as e:
        print(f"[ERROR] Voice output failed: {e}")
        last_spoken_text = text
    finally:
        if temp_speaker:
            try:
                temp_speaker.stop()
                del temp_speaker
                time.sleep(0.1)
            except Exception as e:
                print(f"[WARNING] Error during TTS cleanup: {e}")


# ----------------------------
# INIT SPEECH-TO-TEXT
# ----------------------------
r = sr.Recognizer()

# ----------------------------
# DATABASE OF KNOWN PEOPLE
# ----------------------------
PEOPLE_INFO_FILE = "people_info.json"
people_info = {}

def load_people_info():
    """Loads known people information from a JSON file."""
    global people_info
    if os.path.exists(PEOPLE_INFO_FILE):
        try:
            with open(PEOPLE_INFO_FILE, "r") as f:
                people_info = json.load(f)
            print(f"Loaded {len(people_info)} people from {PEOPLE_INFO_FILE}")
        except json.JSONDecodeError as e:
            print(f"[WARNING] Could not decode {PEOPLE_INFO_FILE}: {e}. Starting with empty data.")
            people_info = {}
        except Exception as e:
            print(f"[ERROR] Loading {PEOPLE_INFO_FILE}: {e}. Starting with empty data.")
            people_info = {}
    else:
        print(f"'{PEOPLE_INFO_FILE}' not found. Starting with empty data.")

def save_people_info():
    """Saves known people information to a JSON file."""
    try:
        with open(PEOPLE_INFO_FILE, "w") as f:
            json.dump(people_info, f, indent=4)
        print(f"Saved {len(people_info)} people to {PEOPLE_INFO_FILE}")
    except Exception as e:
        print(f"[ERROR] Saving {PEOPLE_INFO_FILE}: {e}")

# Load initial data
load_people_info()

# ----------------------------
# LOAD KNOWN FACE IMAGES AND ENCODINGS
# ----------------------------
known_face_encodings = []
known_face_names = [] # This will still store the human-readable names

def load_known_faces():
    """Loads face images and their encodings for known people."""
    global known_face_encodings, known_face_names
    known_face_encodings = []
    known_face_names = []
    for person_name, info in people_info.items():
        # Retrieve the face_image_id (UUID)
        face_image_id = info.get("face_image_id")
        if not face_image_id:
            print(f"[WARNING] No face_image_id found for {person_name}. Skipping.")
            continue

        try:
            image_path = os.path.join("faces", f"{face_image_id}.jpg")
            if not os.path.exists(image_path):
                print(f"[WARNING] Image for {person_name} (ID: {face_image_id}) not found at {image_path}. Skipping.")
                continue
            image = face_recognition.load_image_file(image_path)
            
            # Find face locations in the reloaded image.
            reloaded_face_locations = face_recognition.face_locations(image)

            if reloaded_face_locations:
                # Get the encoding for the first (and likely only) face found in the reloaded image
                encodings = face_recognition.face_encodings(image, reloaded_face_locations)
                if encodings:
                    encoding = encodings[0]
                    known_face_encodings.append(encoding)
                    known_face_names.append(person_name)
                else:
                    print(f"[WARNING] Could not compute encoding for {person_name} (ID: {face_image_id}) from loaded image. Skipping.")
            else:
                print(f"[WARNING] No face found in image for {person_name} (ID: {face_image_id}) after reloading. Skipping.")
                
        except Exception as e:
            print(f"[ERROR] Could not load face for {person_name} (ID: {face_image_id}): {e}")
    print(f"Loaded {len(known_face_names)} known face encodings.")

# Load known faces at startup
load_known_faces()

# ----------------------------
# SAVE TO LOG FILES
# ----------------------------
def log_recognition(name):
    """Logs recognized person data to JSON and CSV files."""
    try:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        data = {
            "name": name,
            "relationship": people_info.get(name, {}).get("relationship", "Unknown"),
            "phone": people_info.get(name, {}).get("phone", "Unknown"),
            "work_hours": people_info.get(name, {}).get("work_hours", "Unknown"),
            "time": timestamp
        }

        # Log to JSON
        with open("recognized_log.json", "a") as json_file:
            json.dump(data, json_file)
            json_file.write("\n")

        # Log to CSV
        csv_exists = os.path.exists("recognized_log.csv")
        with open("recognized_log.csv", "a", newline='') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=data.keys())
            if not csv_exists:
                writer.writeheader()
            writer.writerow(data)
    except Exception as e:
        print(f"[ERROR] Logging recognition: {e}")

# ----------------------------
# GLOBAL STATE VARIABLES FOR RECOGNITION LOOP
# ----------------------------
video_capture = None
recognition_active = False
is_prompting_unknown = False
unknown_person_detected_time = None
current_unknown_face_frame = None # Store the frame of the unknown person
current_unknown_face_location = None # Store the face location of the unknown person
stt_subtitle_text = ""
last_spoken_text = "" # To prevent repeating questions immediately
recognized_set = set() # To track who has been announced in the current session (reset per start_recognition)

# New global state variables for UI interactions
current_recognized_person_name = None # Stores the name of the last person announced
is_registering_new_face = False # Flag for "Register New Face" flow
is_updating_person_info = False # Flag for "Update Person Info" flow


# ----------------------------
# NEW PERSON REGISTRATION FLOW
# ----------------------------
def add_new_person(name, relationship, phone, work_hours, face_frame, face_location):
    """
    Adds a new person to the known people database and saves their face image.
    Uses UUID for robust image file management.
    This function is also used for reintroduction/updating existing person's face.
    """
    global people_info, known_face_encodings, known_face_names

    if not name:
        print("[WARNING] Cannot add person without a name.")
        return False # Indicate failure

    is_update = False
    if name in people_info:
        is_update = True
        speak_and_wait(f"A person named {name} already exists in my records. Updating their information and face image.", wait_for_completion=True)
        existing_face_image_id = people_info[name].get("face_image_id")
        if existing_face_image_id:
            old_image_path = os.path.join("faces", f"{existing_face_image_id}.jpg")
            if os.path.exists(old_image_path):
                try:
                    os.remove(old_image_path)
                    print(f"Removed old face image: {old_image_path}")
                except Exception as e:
                    print(f"[ERROR] Could not remove old face image {old_image_path}: {e}")

    # Generate a unique ID for the face image
    face_image_id = str(uuid.uuid4())
    image_filename = os.path.join("faces", f"{face_image_id}.jpg")

    # Extract the face region from the full frame
    top, right, bottom, left = face_location
    face_image_cut = face_frame[top:bottom, left:right]

    # Ensure the cropped image is valid before saving
    if face_image_cut.size == 0 or face_image_cut.shape[0] == 0 or face_image_cut.shape[1] == 0:
        print(f"[ERROR] Cropped face image is empty or invalid for {name}. Cannot save or encode.")
        speak_and_wait(f"I could not capture a clear image of your face, {name}. Please try again.", wait_for_completion=True)
        return False # Indicate failure

    # --- Start of new error handling for saving image ---
    try:
        # Attempt to save the image with a quality setting (optional, but can sometimes help)
        # For JPEG, quality can be 0-100. 95 is a good balance.
        cv2.imwrite(image_filename, face_image_cut, [cv2.IMWRITE_JPEG_QUALITY, 95])
        print(f"Saved new face image: {image_filename}")
    except cv2.error as e: # Catch specific OpenCV errors for saving
        print(f"[ERROR] OpenCV failed to save face image {image_filename}: {e}")
        speak_and_wait(f"There was an error saving your face image (OpenCV issue), {name}. Please try again.", wait_for_completion=True)
        return False # Exit if saving fails
    except Exception as e: # Catch any other general exceptions during saving
        print(f"[ERROR] Failed to save face image {image_filename}: {e}")
        speak_and_wait(f"There was an unexpected error saving your face image, {name}. Please try again.", wait_for_completion=True)
        return False # Exit if saving fails
    # --- End of new error handling for saving image ---

    # Add/Update to people_info with the UUID
    people_info[name] = {
        "relationship": relationship,
        "phone": phone,
        "work_hours": work_hours,
        "face_image_id": face_image_id # Store the UUID
    }
    save_people_info() # Save updated people_info to JSON

    # --- Start of new error handling for loading and encoding image ---
    reloaded_image = None
    try:
        reloaded_image = face_recognition.load_image_file(image_filename)
    except Exception as e:
        print(f"[ERROR] Failed to load saved face image {image_filename} for encoding: {e}")
        speak_and_wait(f"I could not re-process your saved face image, {name}. Please try again later.", wait_for_completion=True)
        return False # Exit if loading fails

    # Find face locations in the reloaded image.
    reloaded_face_locations = face_recognition.face_locations(reloaded_image)

    if reloaded_face_locations:
        try:
            # Get the encoding for the first (and likely only) face found in the reloaded image
            new_encodings = face_recognition.face_encodings(reloaded_image, reloaded_face_locations)
            if new_encodings:
                new_encoding = new_encodings[0]
                
                # Update known_face_encodings and known_face_names lists
                # Reload all known faces to ensure lists are consistent after adding/updating
                load_known_faces() # This is simpler than trying to update incrementally
                
                print(f"Successfully added/updated {name} to known faces.")
                log_recognition(name) # Log the new person as recognized
                speak_and_wait(f"Thank you, {name}. I have {'updated' if is_update else 'added'} you to my records.", wait_for_completion=True)
                recognized_set.add(name) # Add to current session's recognized set
                return True # Indicate success
            else:
                print(f"[WARNING] Could not compute encoding for {name} from saved image (no encodings found). Not added to recognition list.")
                speak_and_wait(f"I could not process your face from the saved image, {name}. Please try again later.", wait_for_completion=True)
                return False # Indicate failure
        except Exception as e:
            print(f"[ERROR] Failed to compute encoding for {name} from saved image: {e}. This might be a dlib/face_recognition issue.")
            speak_and_wait(f"I encountered an error processing your face for recognition, {name}. Please try again later.", wait_for_completion=True)
            return False # Indicate failure
    else:
        print(f"[WARNING] No face found in the saved image for {name} after reloading. Not added to recognition list.")
        speak_and_wait(f"I could not process your face from the saved image, {name}. Please try again later.", wait_for_completion=True)
        return False # Indicate failure
    # --- End of new error handling for loading and encoding image ---

def handle_unknown_person_flow():
    """Manages the interaction to register an unknown person."""
    global is_prompting_unknown, stt_subtitle_text, current_unknown_face_frame, current_unknown_face_location, last_spoken_text

    # Check explicitly for None, not truthiness of NumPy array
    if current_unknown_face_frame is None or current_unknown_face_location is None:
        print("[WARNING] No valid unknown face frame/location to process for new person flow.")
        is_prompting_unknown = False
        return

    # Reset last spoken text to ensure the initial greeting is spoken
    last_spoken_text = "" 
    speak_and_wait("Hello. I don't recognize you.", wait_for_completion=True)
    time.sleep(0.5) # Small pause

    name = ""
    for _ in range(3): # Try to get name 3 times
        question = "What is your name?"
        if last_spoken_text != question: # Only ask if not just asked
            speak_and_wait(question, wait_for_completion=True)
        name = listen_for_input().strip()
        if name:
            break
        else:
            speak_and_wait("I didn't catch that. Please tell me your name.", wait_for_completion=True)
    
    if not name:
        speak_and_wait("Okay, I will mark you as unknown for now.", wait_for_completion=True)
        is_prompting_unknown = False
        stt_subtitle_text = ""
        return

    speak_and_wait(f"Nice to meet you, {name}.", wait_for_completion=True)
    
    relationship = ""
    question = "What is your relationship to the person living here? For example, friend, family, or caretaker. This is optional."
    if last_spoken_text != question:
        speak_and_wait(question, wait_for_completion=True)
    relationship = listen_for_input().strip()
    if not relationship:
        relationship = "Unknown"
        speak_and_wait("No relationship provided.", wait_for_completion=True)

    phone = ""
    question = "Can I get your phone number? This is optional."
    if last_spoken_text != question:
        speak_and_wait(question, wait_for_completion=True)
    phone = listen_for_input().strip()
    if not phone:
        phone = "Unknown"
        speak_and_wait("No phone number provided.", wait_for_completion=True)

    work_hours = ""
    question = "What are your typical work hours? This is optional."
    if last_spoken_text != question:
        speak_and_wait(question, wait_for_completion=True)
    work_hours = listen_for_input().strip()
    if not work_hours:
        work_hours = "Unknown"
        speak_and_wait("No work hours provided.", wait_for_completion=True)

    # Add the new person to the database
    add_new_person(name, relationship, phone, work_hours, current_unknown_face_frame, current_unknown_face_location)

    is_prompting_unknown = False
    stt_subtitle_text = "" # Clear subtitle after interaction
    last_spoken_text = "" # Reset last spoken text after interaction

# ----------------------------
# VideoStream Class for smooth camera feed
# ----------------------------
class VideoStream:
    def __init__(self, src=0): # Removed width, height parameters
        self.stream = cv2.VideoCapture(src)
        if not self.stream.isOpened():
            raise Exception(f"Could not open video stream from source {src}. Make sure camera is connected and not in use.")
        
        # Removed explicit resolution setting, using camera default
        # self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        # self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        # Verify resolution set (now actual camera default)
        actual_width = int(self.stream.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self.stream.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"[DIAGNOSTIC] Camera opened, actual resolution: {actual_width}x{actual_height} (using default).")


        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False
        self.lock = threading.Lock()

    def start(self):
        threading.Thread(target=self.update, args=(), daemon=True).start()
        return self

    def update(self):
        while True:
            if self.stopped:
                return
            (grabbed, frame) = self.stream.read()
            with self.lock:
                self.grabbed = grabbed
                self.frame = frame

    def read(self):
        with self.lock:
            return self.grabbed, self.frame.copy()

    def stop(self):
        with self.lock:
            self.stopped = True
        self.stream.release()

# ----------------------------
# RECOGNITION LOOP FUNCTION
# ----------------------------
def run_recognition_loop(): # Renamed to avoid confusion with Tkinter start_recognition_process
    """Main function for running face recognition and displaying video."""
    global video_capture, recognition_active, is_prompting_unknown, \
           unknown_person_detected_time, current_unknown_face_frame, \
           current_unknown_face_location, stt_subtitle_text, recognized_set, last_spoken_text, \
           is_registering_new_face, is_updating_person_info, current_recognized_person_name # Added new global flags

    # Initialize TTS worker thread
    global _tts_thread, _tts_thread_active
    if not _tts_thread_active:
        _tts_thread_active = True
        _tts_thread = threading.Thread(target=_tts_worker, daemon=True)
        _tts_thread.start()
        print("[DIAGNOSTIC] TTS worker thread started.")
        time.sleep(0.5) # Give TTS engine a moment to initialize

    # Initialize VideoStream with default camera resolution
    vs = None
    try:
        vs = VideoStream(src=0).start()
        time.sleep(1.0)
        if not vs.grabbed:
            raise Exception("Failed to grab initial frame from video stream.")
        print(f"[DIAGNOSTIC] Camera opened successfully via VideoStream (using default resolution).")

        recognition_active = True
        print("[DIAGNOSTIC] Recognition loop starting...")
        
        # Reset recognized set and last spoken text for a new session
        recognized_set = set()
        last_spoken_text = ""
        current_recognized_person_name = None # Reset
        is_registering_new_face = False # Reset
        is_updating_person_info = False # Reset

        cv2.namedWindow(WINDOW_NAME) # Removed WINDOW_NORMAL for default sizing
        # Removed cv2.setMouseCallback as UI is separate

        while recognition_active:
            grabbed, frame = vs.read()
            if not grabbed:
                print("[WARNING] Failed to read frame from VideoStream. Retrying...")
                time.sleep(0.1)
                continue
            
            # Flip frame horizontally for a mirror effect (common for webcams)
            frame = cv2.flip(frame, 1)

            # --- OPTIMIZATION: Reduce Image Resolution for Processing ---
            # The original 'frame' is used for display, but a smaller 'small_frame' is used for face detection and encoding.
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            # --- END OPTIMIZATION ---

            # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
            # Ensure the array is contiguous for dlib
            rgb_frame = np.ascontiguousarray(small_frame[:, :, ::-1])

            face_locations = face_recognition.face_locations(rgb_frame)
            # print(f"[DIAGNOSTIC] Faces detected in small_frame: {len(face_locations)}")

            face_encodings = []
            
            if face_locations:
                try:
                    # Compute encodings directly using the image and the face locations.
                    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
                    # print(f"[DIAGNOSTIC] Face encodings computed: {len(face_encodings)}")
                except Exception as e:
                    # Catching the specific dlib error for better user guidance
                    if "compute_face_descriptor()" in str(e) or "error: 3" in str(e):
                        print(f"[CRITICAL ERROR] Failed to compute face encodings: {e}")
                        print("This often indicates an issue with your dlib/face_recognition installation or C++ build tools.")
                        print("Please try: `pip uninstall face_recognition dlib` then `pip install face_recognition`.")
                        print("On Windows, ensure you have 'Desktop development with C++' installed via Visual Studio Installer.")
                    else:
                        print(f"[ERROR] Failed to compute face encodings in main loop: {e}. This might be a transient issue or a problem with face quality.")
                    face_encodings = []


            current_frame_has_unknown = False
            
            for i, (top, right, bottom, left) in enumerate(face_locations):
                name = "Unknown"
                
                if i < len(face_encodings): # Ensure we have an encoding for this face
                    face_encoding = face_encodings[i]

                    # Compare with known faces
                    if known_face_encodings: # Only compare if there are known faces
                        matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.6) # Adjust tolerance as needed
                        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                        
                        if face_distances.size > 0:
                            best_match_index = np.argmin(face_distances)
                            if matches[best_match_index]:
                                name = known_face_names[best_match_index]

                # Scale back up face locations to original frame size
                original_top = top * 4
                original_right = right * 4
                original_bottom = bottom * 4
                original_left = left * 4

                # print(f"[DIAGNOSTIC] Scaled face coordinates for drawing: (L:{original_left}, T:{original_top}) (R:{original_right}, B:{original_bottom}) for {name}")

                # Draw bounding box and name
                color = (0, 255, 0) if name != "Unknown" else (0, 0, 255) # Green for known, Red for unknown
                # Draw rounded rectangle for face bounding box
                draw_rounded_rectangle(frame, (original_left, original_top), (original_right, original_bottom), color, 2, CORNER_RADIUS, filled=False)
                
                # Add semi-transparent overlay inside the bounding box for contrast
                overlay = frame.copy()
                alpha = 0.2 # Transparency factor
                # Draw filled rounded rectangle for overlay
                draw_rounded_rectangle(overlay, (original_left, original_top), (original_right, original_bottom), color, -1, CORNER_RADIUS, filled=True)
                frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

                # Display name
                cv2.putText(frame, name, (original_left, original_top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                # print(f"[DIAGNOSTIC] cv2.putText (name) called for {name}.")

                # Display detailed info for known people
                if name != "Unknown":
                    current_recognized_person_name = name # Update the currently recognized person
                    info = people_info.get(name, {})
                    label_lines = [
                        f"Relation: {info.get('relationship', 'Unknown')}",
                        f"Phone: {info.get('phone', 'Unknown')}",
                        f"Work Hours: {info.get('work_hours', 'Unknown')}"
                    ]
                    y = original_bottom + 20
                    for line in label_lines:
                        cv2.putText(frame, line, (original_left, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                        y += 20
                    # print(f"[DIAGNOSTIC] cv2.putText (info) called for {name}.")
                    
                    # Announce known person if not already announced in this session
                    if name not in recognized_set and not is_prompting_unknown and not is_registering_new_face and not is_updating_person_info:
                        recognized_set.add(name)
                        log_recognition(name)
                        info_text = f"This is {name}, your {info.get('relationship', 'friend')}. " \
                                    f"They work {info.get('work_hours', 'unknown hours')}. " \
                                    f"You can reach them at {info.get('phone', 'unknown phone number')}."
                        # Use non-blocking speech for recognition announcements
                        speak_and_wait(info_text, wait_for_completion=False) 
                else: # Handle unknown person detection
                    current_frame_has_unknown = True
                    # Store the full frame and exact location for potential registration
                    # Only update these if we are not already prompting or in a specific interaction mode
                    if not is_prompting_unknown and not is_registering_new_face and not is_updating_person_info:
                        current_unknown_face_frame = frame.copy() 
                        current_unknown_face_location = (original_top, original_right, original_bottom, original_left) # Store scaled coordinates
            
            # Logic for triggering unknown person registration flow OR manual registration
            if (current_frame_has_unknown and not is_prompting_unknown and not is_registering_new_face and not is_updating_person_info):
                if unknown_person_detected_time is None:
                    unknown_person_detected_time = time.time()
                elif (time.time() - unknown_person_detected_time) > 3: # If unknown person persists for 3 seconds
                    is_prompting_unknown = True
                    # Start the interaction in a separate thread
                    threading.Thread(target=handle_unknown_person_flow, daemon=True).start()
            elif is_registering_new_face and not is_prompting_unknown and face_locations: # Manual registration initiated and face is present
                # Use the first detected face for registration
                first_face_location = (face_locations[0][0]*4, face_locations[0][1]*4, face_locations[0][2]*4, face_locations[0][3]*4)
                current_unknown_face_frame = frame.copy()
                current_unknown_face_location = first_face_location
                is_prompting_unknown = True # Set this to block other interactions
                is_registering_new_face = False # Reset this flag after triggering
                threading.Thread(target=handle_unknown_person_flow, daemon=True).start()
            elif not current_frame_has_unknown and not is_prompting_unknown and not is_registering_new_face and not is_updating_person_info: # Reset timer if unknown person leaves and no prompt is active
                unknown_person_detected_time = None 


            # Display STT subtitles on the camera feed if active
            if stt_subtitle_text:
                text_size = cv2.getTextSize(stt_subtitle_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
                text_x = (frame.shape[1] - text_size[0]) // 2 # Center the text
                text_y = frame.shape[0] - 30 # 30 pixels from bottom
                cv2.rectangle(frame, (0, frame.shape[0] - 60), (frame.shape[1], frame.shape[0]), (0,0,0), -1) # Black background for subtitle
                cv2.putText(frame, stt_subtitle_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2) # Yellow text
                # print(f"[DIAGNOSTIC] cv2.putText (subtitle) called: '{stt_subtitle_text}'.")

            cv2.imshow(WINDOW_NAME, frame)
            
            # Break loop on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("[DIAGNOSTIC] 'q' pressed, breaking loop.")
                break

    except Exception as e:
        print(f"[FATAL ERROR] Recognition crashed: {e}")
        messagebox.showerror("Fatal Error", f"The recognition process crashed: {e}")
    finally:
        if vs: # Ensure VideoStream is stopped
            print("[DIAGNOSTIC] Stopping VideoStream.")
            vs.stop()
        print("[DIAGNOSTIC] Destroying all OpenCV windows.")
        cv2.destroyAllWindows()
        recognition_active = False # Ensure flag is reset on exit
        print("[DIAGNOSTIC] Recognition active flag reset.")

# ----------------------------
# GUI SETUP (Tkinter)
# ----------------------------
status_label = None # Global reference for Tkinter status label

def start_recognition_thread():
    """Starts the recognition loop in a separate thread."""
    global recognition_active # Corrected to global
    if not recognition_active:
        print("[GUI] Attempting to start recognition thread.")
        threading.Thread(target=run_recognition_loop, daemon=True).start()
        status_label.config(text="Status: Recognition Running...", fg="green")
    else:
        messagebox.showinfo("Info", "Recognition is already running.")

def stop_recognition_thread():
    """Stops the recognition loop."""
    global recognition_active # Corrected to global
    if recognition_active:
        recognition_active = False
        status_label.config(text="Status: Recognition Stopped.", fg="red")
        print("[GUI] Signaled recognition thread to stop.")
    else:
        messagebox.showinfo("Info", "Recognition is not running.")

def repeat_last_info_thread():
    """Repeats the information for the last recognized person."""
    global current_recognized_person_name, is_prompting_unknown
    if is_prompting_unknown:
        messagebox.showinfo("Info", "I am currently in an interaction. Please wait.")
        return

    if current_recognized_person_name and current_recognized_person_name in people_info:
        info = people_info.get(current_recognized_person_name, {})
        info_text = f"This is {current_recognized_person_name}, your {info.get('relationship', 'friend')}. " \
                    f"They work {info.get('work_hours', 'unknown hours')}. " \
                    f"You can reach them at {info.get('phone', 'unknown phone number')}."
        threading.Thread(target=speak_and_wait, args=(info_text,)).start()
    else:
        messagebox.showinfo("Info", "No person has been recognized yet to repeat information.")

def remove_person_flow():
    """Initiates the flow to remove a person."""
    global is_prompting_unknown, stt_subtitle_text
    if is_prompting_unknown:
        messagebox.showinfo("Info", "I am currently in an interaction. Please wait.")
        return
    
    is_prompting_unknown = True # Block other interactions
    stt_subtitle_text = "Who would you like to remove?"
    speak_and_wait("Who would you like to remove?", wait_for_completion=True)
    name_to_remove = listen_for_input().strip()

    if not name_to_remove:
        speak_and_wait("No name provided. Cancelling removal.", wait_for_completion=True)
    elif name_to_remove not in people_info:
        speak_and_wait(f"I don't have {name_to_remove} in my records.", wait_for_completion=True)
    else:
        try:
            # Remove image file
            face_image_id = people_info[name_to_remove].get("face_image_id")
            if face_image_id:
                image_path = os.path.join("faces", f"{face_image_id}.jpg")
                if os.path.exists(image_path):
                    os.remove(image_path)
                    print(f"Removed face image: {image_path}")
            
            # Remove from people_info
            del people_info[name_to_remove]
            save_people_info()
            load_known_faces() # Reload in-memory data
            speak_and_wait(f"{name_to_remove} has been removed from my records.", wait_for_completion=True)
            global recognized_set
            if name_to_remove in recognized_set:
                recognized_set.remove(name_to_remove) # Remove from current session's recognized set
        except Exception as e:
            speak_and_wait(f"An error occurred while trying to remove {name_to_remove}: {e}", wait_for_completion=True)
            print(f"[ERROR] Error removing person: {e}")
    
    is_prompting_unknown = False
    stt_subtitle_text = ""

def change_name_flow():
    """Initiates the flow to change the name of a person (by deleting and re-registering)."""
    global is_prompting_unknown, is_registering_new_face, stt_subtitle_text
    if is_prompting_unknown:
        messagebox.showinfo("Info", "I am currently in an interaction. Please wait.")
        return
    
    is_prompting_unknown = True # Block other interactions
    stt_subtitle_text = "Whose name would you like to change?"
    speak_and_wait("Whose name would you like to change?", wait_for_completion=True)
    old_name = listen_for_input().strip()

    if not old_name:
        speak_and_wait("No name provided. Cancelling name change.", wait_for_completion=True)
        is_prompting_unknown = False
        stt_subtitle_text = ""
        return
    elif old_name not in people_info:
        speak_and_wait(f"I don't have {old_name} in my records.", wait_for_completion=True)
        is_prompting_unknown = False
        stt_subtitle_text = ""
        return
    
    # Remove the old entry
    try:
        face_image_id = people_info[old_name].get("face_image_id")
        if face_image_id:
            image_path = os.path.join("faces", f"{face_image_id}.jpg")
            if os.path.exists(image_path):
                os.remove(image_path)
                print(f"Removed old face image for name change: {image_path}")
        del people_info[old_name]
        save_people_info()
        load_known_faces()
        global recognized_set
        if old_name in recognized_set:
            recognized_set.remove(old_name)
        speak_and_wait(f"Removed old entry for {old_name}. Now, please show me the face you want to associate with a new name.", wait_for_completion=True)
    except Exception as e:
        speak_and_wait(f"An error occurred while preparing for name change: {e}", wait_for_completion=True)
        print(f"[ERROR] Error during name change preparation: {e}")
        is_prompting_unknown = False
        stt_subtitle_text = ""
        return
    
    is_registering_new_face = True # Signal the main loop to capture the next face for registration
    is_prompting_unknown = False # Release this flag so main loop can detect face
    stt_subtitle_text = "Waiting for face to register new name..."
    # The handle_unknown_person_flow will be triggered by the main loop when a face is detected

def update_person_info_flow():
    """Initiates the flow to update info for an existing person."""
    global is_prompting_unknown, is_updating_person_info, stt_subtitle_text, last_spoken_text
    if is_prompting_unknown:
        messagebox.showinfo("Info", "I am currently in an interaction. Please wait.")
        return
    
    is_prompting_unknown = True # Block other interactions
    is_updating_person_info = True # Set flag for this specific flow
    stt_subtitle_text = "Whose information would you like to update?"
    speak_and_wait("Whose information would you like to update?", wait_for_completion=True)
    name_to_update = listen_for_input().strip()

    if not name_to_update:
        speak_and_wait("No name provided. Cancelling update.", wait_for_completion=True)
        is_prompting_unknown = False
        is_updating_person_info = False
        stt_subtitle_text = ""
        return
    elif name_to_update not in people_info:
        speak_and_wait(f"I don't have {name_to_update} in my records.", wait_for_completion=True)
        is_prompting_unknown = False
        is_updating_person_info = False
        stt_subtitle_text = ""
        return
    
    current_info = people_info[name_to_update]
    speak_and_wait(f"Current information for {name_to_update}: Relationship is {current_info.get('relationship', 'Unknown')}, phone is {current_info.get('phone', 'Unknown')}, work hours are {current_info.get('work_hours', 'Unknown')}.", wait_for_completion=True)

    relationship = ""
    question = "What is their new relationship? Say 'skip' to keep current."
    if last_spoken_text != question:
        speak_and_wait(question, wait_for_completion=True)
    relationship_input = listen_for_input().strip()
    if relationship_input.lower() != 'skip' and relationship_input:
        relationship = relationship_input
    else:
        relationship = current_info.get('relationship', 'Unknown')
        speak_and_wait("Keeping current relationship.", wait_for_completion=True)

    phone = ""
    question = "What is their new phone number? Say 'skip' to keep current."
    if last_spoken_text != question:
        speak_and_wait(question, wait_for_completion=True)
    phone_input = listen_for_input().strip()
    if phone_input.lower() != 'skip' and phone_input:
        phone = phone_input
    else:
        phone = current_info.get('phone', 'Unknown')
        speak_and_wait("Keeping current phone number.", wait_for_completion=True)

    work_hours = ""
    question = "What are their new typical work hours? Say 'skip' to keep current."
    if last_spoken_text != question:
        speak_and_wait(question, wait_for_completion=True)
    work_hours_input = listen_for_input().strip()
    if work_hours_input.lower() != 'skip' and work_hours_input:
        work_hours = work_hours_input
    else:
        work_hours = current_info.get('work_hours', 'Unknown')
        speak_and_wait("Keeping current work hours.", wait_for_completion=True)
    
    people_info[name_to_update]['relationship'] = relationship
    people_info[name_to_update]['phone'] = phone
    people_info[name_to_update]['work_hours'] = work_hours
    save_people_info()
    load_known_faces() # Refresh in-memory data
    speak_and_wait(f"Information for {name_to_update} has been updated.", wait_for_completion=True)

    is_prompting_unknown = False
    is_updating_person_info = False
    stt_subtitle_text = ""
    last_spoken_text = ""


def start_gui():
    """Initializes and runs the Tkinter GUI."""
    global status_label

    print("[GUI] Starting GUI initialization...")
    root = Tk()
    print("[GUI] Tkinter root window created.")
    root.title("Alzheimer Face Recognition Assistant")
    root.geometry("500x450") # Increased height for new buttons
    root.resizable(False, False) # Prevent resizing

    # Attempt to bring the window to the front
    root.lift()
    root.attributes('-topmost', True)
    root.after_idle(root.attributes, '-topmost', False) # Remove topmost after a short delay

    # Styling
    root.configure(bg="#f0f0f0")
    font_large = ("Helvetica", 16, "bold")
    font_medium = ("Helvetica", 12)
    button_style = {"font": font_medium, "bg": "#4CAF50", "fg": "white", "activebackground": "#45a049", "activeforeground": "white", "relief": "raised", "bd": 3, "width": 25, "height": 2}
    exit_button_style = {"font": font_medium, "bg": "#f44336", "fg": "white", "activebackground": "#da190b", "activeforeground": "white", "relief": "raised", "bd": 3, "width": 25, "height": 2}
    action_button_style = {"font": font_medium, "bg": "#2196F3", "fg": "white", "activebackground": "#1976D2", "activeforeground": "white", "relief": "raised", "bd": 3, "width": 25, "height": 2} # Blue for general actions
    warning_button_style = {"font": font_medium, "bg": "#FF9800", "fg": "white", "activebackground": "#FB8C00", "activeforeground": "white", "relief": "raised", "bd": 3, "width": 25, "height": 2} # Orange for warning/destructive actions

    Label(root, text="Face Recognition Assistant for Alzheimer's", font=font_large, bg="#f0f0f0", fg="#333").pack(pady=20)

    status_label = Label(root, text="Status: Idle", font=font_medium, bg="#f0f0f0", fg="blue")
    status_label.pack(pady=5)

    Button(root, text="Start Recognition", command=start_recognition_thread, **button_style).pack(pady=5)
    Button(root, text="Stop Recognition", command=stop_recognition_thread, **exit_button_style).pack(pady=5)
    
    # New Buttons
    Button(root, text="Repeat Last Info", command=repeat_last_info_thread, **action_button_style).pack(pady=5)
    Button(root, text="Remove Person", command=lambda: threading.Thread(target=remove_person_flow, daemon=True).start(), **warning_button_style).pack(pady=5)
    Button(root, text="Register New Face", command=lambda: threading.Thread(target=change_name_flow, daemon=True).start(), **action_button_style).pack(pady=5)
    Button(root, text="Update Person Info", command=lambda: threading.Thread(target=update_person_info_flow, daemon=True).start(), **action_button_style).pack(pady=5)

    Button(root, text="Exit Application", command=root.destroy, **exit_button_style).pack(pady=5)

    # Handle window closing event
    def on_closing():
        stop_recognition_thread()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_closing)
    print("[GUI] Entering mainloop...")
    root.mainloop()
    print("[GUI] Exited mainloop.")

# ----------------------------
# RUN GUI
# ----------------------------
if __name__ == '__main__':
    # Start the GUI, which will then allow the user to start recognition
    start_gui()
