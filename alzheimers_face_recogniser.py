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
import uuid # New import for UUID generation

# Ensure 'faces' directory exists
if not os.path.exists("faces"):
    os.makedirs("faces")
CORNER_RADIUS = 15  # For rounded rectangles

def draw_rounded_rectangle(image, top_left, bottom_right, color, thickness, radius, filled=False):
    x1, y1 = top_left
    x2, y2 = bottom_right
    width = x2 - x1
    height = y2 - y1
    radius = min(radius, width // 2, height // 2)
    if radius < 0:
        radius = 0
    if filled:
        cv2.rectangle(image, (x1 + radius, y1), (x2 - radius, y2), color, -1)
        cv2.rectangle(image, (x1, y1 + radius), (x2, y2 - radius), color, -1)
    else:
        cv2.line(image, (x1 + radius, y1), (x2 - radius, y1), color, thickness)
        cv2.line(image, (x1 + radius, y2), (x2 - radius, y2), color, thickness)
        cv2.line(image, (x1, y1 + radius), (x1, y2 - radius), color, thickness)
        cv2.line(image, (x2, y1 + radius), (x2, y2 - radius), color, thickness)
    if radius > 0:
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
# INIT TEXT-TO-SPEECH (Removed global initialization, now done per call in speak_and_wait)
# ----------------------------
# speaker = None # This is no longer needed globally
# try:
#     speaker = pyttsx3.init()
#     speaker.setProperty('rate', 160)
# except Exception as e:
#     print(f"[ERROR] Initializing TTS: {e}")
#     messagebox.showerror("TTS Error", f"Could not initialize Text-to-Speech: {e}")

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
reintroduction_active_for_name = None # New: Stores the name of the person being reintroduced

# ----------------------------
# HELPER FUNCTIONS FOR TTS/STT
# ----------------------------
def speak_and_wait(text):
    """Speaks the given text and waits for it to finish."""
    global last_spoken_text
    temp_speaker = None # Initialize temp_speaker to None
    try:
        # Initialize the speaker engine for each call
        temp_speaker = pyttsx3.init()
        temp_speaker.setProperty('rate', 160)
        
        print(f"[TTS] Speaking: {text}")
        temp_speaker.say(text)
        temp_speaker.runAndWait()
        print(f"[TTS] Finished speaking: {text}") # Diagnostic
        last_spoken_text = text # Update last spoken text
        
    except Exception as e:
        print(f"[ERROR] Voice output failed: {e}")
        # If initialization fails, ensure last_spoken_text is still updated to prevent re-asking
        last_spoken_text = text 
    finally:
        # Ensure temp_speaker is cleaned up even if an error occurs
        if temp_speaker: # Check if temp_speaker was successfully initialized
            try:
                temp_speaker.stop()
                del temp_speaker # Explicitly delete to release resources
                time.sleep(0.1) # Small delay to ensure resources are fully released
            except Exception as e:
                print(f"[WARNING] Error during TTS cleanup: {e}")


def listen_for_input():
    """Listens for voice input and returns transcribed text."""
    global stt_subtitle_text
    with sr.Microphone() as source:
        print("[STT] Adjusting for ambient noise (1 second duration)...")
        r.adjust_for_ambient_noise(source, duration=1) # Adjust for ambient noise for 1 second
        print("[STT] Listening for input (timeout 7s, phrase_time_limit 7s)...")
        stt_subtitle_text = "Listening..."
        try:
            # Increased timeout and phrase_time_limit for more flexibility
            audio = r.listen(source, timeout=7, phrase_time_limit=7) # Listen for up to 7 seconds
            print("[STT] Audio received. Processing...")
            stt_subtitle_text = "Processing..."
            text = r.recognize_google(audio)
            print(f"[STT] Heard: {text}")
            stt_subtitle_text = text # Update subtitle with recognized text
            return text
        except sr.UnknownValueError:
            print("[STT] Could not understand audio (UnknownValueError).")
            stt_subtitle_text = "Could not understand audio."
            return ""
        except sr.RequestError as e:
            print(f"[STT] Could not request results from Google Speech Recognition service; {e} (RequestError).")
            stt_subtitle_text = f"STT Error: {e}"
            return ""
        except sr.WaitTimeoutError:
            print("[STT] No speech detected within timeout (WaitTimeoutError).")
            stt_subtitle_text = "No speech detected."
            return ""
        except Exception as e:
            print(f"[ERROR] STT failed: {e} (Generic Exception).")
            stt_subtitle_text = f"STT Error: {e}"
            return ""

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
        speak_and_wait(f"A person named {name} already exists in my records. Updating their information and face image.")
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
        speak_and_wait(f"I could not capture a clear image of your face, {name}. Please try again.")
        return False # Indicate failure

    # --- Start of new error handling for saving image ---
    try:
        # Attempt to save the image with a quality setting (optional, but can sometimes help)
        # For JPEG, quality can be 0-100. 95 is a good balance.
        cv2.imwrite(image_filename, face_image_cut, [cv2.IMWRITE_JPEG_QUALITY, 95])
        print(f"Saved new face image: {image_filename}")
    except cv2.error as e: # Catch specific OpenCV errors for saving
        print(f"[ERROR] OpenCV failed to save face image {image_filename}: {e}")
        speak_and_wait(f"There was an error saving your face image (OpenCV issue), {name}. Please try again.")
        return False # Exit if saving fails
    except Exception as e: # Catch any other general exceptions during saving
        print(f"[ERROR] Failed to save face image {image_filename}: {e}")
        speak_and_wait(f"There was an unexpected error saving your face image, {name}. Please try again.")
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
        speak_and_wait(f"I could not re-process your saved face image, {name}. Please try again later.")
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
                speak_and_wait(f"Thank you, {name}. I have {'updated' if is_update else 'added'} you to my records.")
                recognized_set.add(name) # Add to current session's recognized set
                return True # Indicate success
            else:
                print(f"[WARNING] Could not compute encoding for {name} from saved image (no encodings found). Not added to recognition list.")
                speak_and_wait(f"I could not process your face from the saved image, {name}. Please try again later.")
                return False # Indicate failure
        except Exception as e:
            print(f"[ERROR] Failed to compute encoding for {name} from saved image: {e}. This might be a dlib/face_recognition issue.")
            speak_and_wait(f"I encountered an error processing your face for recognition, {name}. Please try again later.")
            return False # Indicate failure
    else:
        print(f"[WARNING] No face found in the saved image for {name} after reloading. Not added to recognition list.")
        speak_and_wait(f"I could not process your face from the saved image, {name}. Please try again later.")
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
    speak_and_wait("Hello. I don't recognize you.")
    time.sleep(0.5) # Small pause

    name = ""
    for _ in range(3): # Try to get name 3 times
        question = "What is your name?"
        if last_spoken_text != question: # Only ask if not just asked
            speak_and_wait(question)
        name = listen_for_input().strip()
        if name:
            break
        else:
            speak_and_wait("I didn't catch that. Please tell me your name.")
    
    if not name:
        speak_and_wait("Okay, I will mark you as unknown for now.")
        is_prompting_unknown = False
        stt_subtitle_text = ""
        return

    speak_and_wait(f"Nice to meet you, {name}.")
    
    relationship = ""
    question = "What is your relationship to the person living here? For example, friend, family, or caretaker. This is optional."
    if last_spoken_text != question:
        speak_and_wait(question)
    relationship = listen_for_input().strip()
    if not relationship:
        relationship = "Unknown"
        speak_and_wait("No relationship provided.")

    phone = ""
    question = "Can I get your phone number? This is optional."
    if last_spoken_text != question:
        speak_and_wait(question)
    phone = listen_for_input().strip()
    if not phone:
        phone = "Unknown"
        speak_and_wait("No phone number provided.")

    work_hours = ""
    question = "What are your typical work hours? This is optional."
    if last_spoken_text != question:
        speak_and_wait(question)
    work_hours = listen_for_input().strip()
    if not work_hours:
        work_hours = "Unknown"
        speak_and_wait("No work hours provided.")

    # Add the new person to the database
    add_new_person(name, relationship, phone, work_hours, current_unknown_face_frame, current_unknown_face_location)

    is_prompting_unknown = False
    stt_subtitle_text = "" # Clear subtitle after interaction
    last_spoken_text = "" # Reset last spoken text after interaction

# ----------------------------
# REINTRODUCTION FLOW
# ----------------------------
def reintroduction_flow():
    """Manages the interaction to reintroduce/update a known person."""
    global reintroduction_active_for_name, is_prompting_unknown, stt_subtitle_text

    if is_prompting_unknown:
        speak_and_wait("I am currently interacting with an unknown person. Please wait.")
        return

    is_prompting_unknown = True # Temporarily use this flag to prevent other interactions
    reintroduction_active_for_name = None # Reset for a new reintroduction attempt

    speak_and_wait("Who would you like to reintroduce?")
    person_name = listen_for_input().strip()

    if not person_name:
        speak_and_wait("No name provided. Cancelling reintroduction.")
        is_prompting_unknown = False
        stt_subtitle_text = ""
        return

    if person_name not in people_info:
        speak_and_wait(f"I don't have {person_name} in my records. Please try again or add them as a new person.")
        is_prompting_unknown = False
        stt_subtitle_text = ""
        return

    speak_and_wait(f"Okay, {person_name}. Please look directly at the camera so I can update your face data.")
    reintroduction_active_for_name = person_name # Set the global flag to indicate reintroduction is active for this name
    stt_subtitle_text = f"Waiting for {person_name} to look at camera..."

    # The main recognition loop will now pick up the face and call add_new_person
    # We don't block here, the main loop will handle the capture and reset reintroduction_active_for_name
    # The is_prompting_unknown flag will be reset by the main loop after processing the reintroduction.

# ----------------------------
# VideoStream Class for smooth camera feed
# ----------------------------
class VideoStream:
    def __init__(self, src=0):
        self.stream = cv2.VideoCapture(src)
        if not self.stream.isOpened():
            raise Exception(f"Could not open video stream from source {src}. Make sure camera is connected and not in use.")
        
        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False
        self.lock = threading.Lock() # For thread-safe access to frame

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
            return self.grabbed, self.frame.copy() # Return a copy to avoid external modification

    def stop(self):
        with self.lock:
            self.stopped = True
        self.stream.release()

# ----------------------------
# RECOGNITION LOOP FUNCTION
# ----------------------------
def run_recognition():
    """Main function for running face recognition and interaction."""
    global video_capture, recognition_active, is_prompting_unknown, \
           unknown_person_detected_time, current_unknown_face_frame, \
           current_unknown_face_location, stt_subtitle_text, recognized_set, last_spoken_text, \
           reintroduction_active_for_name # Added new global

    # Initialize VideoStream
    vs = None
    try:
        vs = VideoStream(src=0).start()
        # Give the video stream a moment to start buffering frames
        time.sleep(1.0) 
        if not vs.grabbed:
            raise Exception("Failed to grab initial frame from video stream.")
        print("[DIAGNOSTIC] Camera opened successfully via VideoStream.")

        recognition_active = True
        print("[DIAGNOSTIC] Recognition loop starting...")
        
        # Reset recognized set and last spoken text for a new session
        recognized_set = set()
        last_spoken_text = ""

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
                draw_rounded_rectangle(frame, (original_left, original_top), (original_right, original_bottom), color, 2, 10) # Thicker border
                # print(f"[DIAGNOSTIC] cv2.rectangle called for {name}.")
                
                # Add a semi-transparent overlay inside the bounding box for contrast
                overlay = frame.copy()
                alpha = 0.2 # Transparency factor
                cv2.rectangle(overlay, (original_left, original_top), (original_right, original_bottom), color, -1) # Filled rectangle
                frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

                # Display name
                cv2.putText(frame, name, (original_left, original_top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                # print(f"[DIAGNOSTIC] cv2.putText (name) called for {name}.")

                # Display detailed info for known people
                if name != "Unknown":
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
                    if name not in recognized_set and not is_prompting_unknown and reintroduction_active_for_name is None: # Do not announce known people during unknown interaction or reintroduction
                        recognized_set.add(name)
                        log_recognition(name)
                        info_text = f"This is {name}, your {info.get('relationship', 'friend')}. " \
                                    f"They work {info.get('work_hours', 'unknown hours')}. " \
                                    f"You can reach them at {info.get('phone', 'unknown phone number')}."
                        # Use a small delay for speaking to prevent blocking main loop too much if many people are detected quickly
                        threading.Thread(target=speak_and_wait, args=(info_text,)).start()
                    
                    # Handle reintroduction capture
                    if reintroduction_active_for_name == name and not is_prompting_unknown:
                        print(f"[REINTRODUCTION] Face of {name} detected for reintroduction.")
                        # Capture the current frame and location for updating
                        # Note: current_unknown_face_frame/location are already updated by the loop for any detected face
                        success = add_new_person(name, info.get('relationship', 'Unknown'), 
                                                 info.get('phone', 'Unknown'), info.get('work_hours', 'Unknown'), 
                                                 frame.copy(), (original_top, original_right, original_bottom, original_left))
                        if success:
                            speak_and_wait(f"Thank you, {name}. Your face data has been updated.")
                        else:
                            speak_and_wait(f"Failed to update face data for {name}. Please try again.")
                        reintroduction_active_for_name = None # Reset reintroduction flag
                        is_prompting_unknown = False # Release the prompt flag

                else: # Handle unknown person detection
                    current_frame_has_unknown = True
                    # Store the full frame and exact location for potential registration
                    # Only update these if we are not already prompting or in reintroduction mode to avoid stale data
                    if not is_prompting_unknown and reintroduction_active_for_name is None:
                        current_unknown_face_frame = frame.copy() 
                        current_unknown_face_location = (original_top, original_right, original_bottom, original_left) # Store scaled coordinates

            # Logic for triggering unknown person registration flow
            if current_frame_has_unknown and not is_prompting_unknown and reintroduction_active_for_name is None:
                if unknown_person_detected_time is None:
                    unknown_person_detected_time = time.time()
                elif (time.time() - unknown_person_detected_time) > 3: # If unknown person persists for 3 seconds
                    is_prompting_unknown = True
                    # Start the interaction in a separate thread
                    threading.Thread(target=handle_unknown_person_flow, daemon=True).start()
            elif not current_frame_has_unknown and not is_prompting_unknown and reintroduction_active_for_name is None: # Reset timer if unknown person leaves and no prompt is active
                unknown_person_detected_time = None 


            # Display STT subtitles at the bottom of the screen
            if stt_subtitle_text:
                text_size = cv2.getTextSize(stt_subtitle_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
                text_x = (frame.shape[1] - text_size[0]) // 2 # Center the text
                text_y = frame.shape[0] - 30 # 30 pixels from bottom
                cv2.putText(frame, stt_subtitle_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2) # Yellow text
                # print(f"[DIAGNOSTIC] cv2.putText (subtitle) called: '{stt_subtitle_text}'.")

            cv2.imshow('Alzheimer Face Assistant', frame)
            
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
# GUI SETUP
# ----------------------------
def start_gui():
    """Initializes and runs the Tkinter GUI."""
    global status_label # Make status_label accessible globally for updates

    print("[GUI] Starting GUI initialization...")
    root = Tk()
    print("[GUI] Tkinter root window created.")
    root.title("Alzheimer Face Recognition Assistant")
    root.geometry("500x350") # Increased height for new button
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
    reintroduce_button_style = {"font": font_medium, "bg": "#FFC107", "fg": "black", "activebackground": "#e0a800", "activeforeground": "black", "relief": "raised", "bd": 3, "width": 25, "height": 2}


    Label(root, text="Face Recognition Assistant for Alzheimer's", font=font_large, bg="#f0f0f0", fg="#333").pack(pady=20)

    status_label = Label(root, text="Status: Idle", font=font_medium, bg="#f0f0f0", fg="blue")
    status_label.pack(pady=5)

    # Nested functions for button commands to ensure correct scope
    def start_recognition_thread():
        """Starts the recognition loop in a separate thread."""
        global recognition_active # Corrected to global
        if not recognition_active:
            print("[GUI] Attempting to start recognition thread.")
            threading.Thread(target=run_recognition, daemon=True).start()
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

    def start_reintroduction_flow_thread():
        """Starts the reintroduction flow in a separate thread."""
        global is_prompting_unknown # Use this flag to prevent other interactions
        if not is_prompting_unknown:
            threading.Thread(target=reintroduction_flow, daemon=True).start()
        else:
            messagebox.showinfo("Info", "Already in an interaction. Please wait.")

    Button(root, text="Start Recognition", command=start_recognition_thread, **button_style).pack(pady=5)
    Button(root, text="Stop Recognition", command=stop_recognition_thread, **exit_button_style).pack(pady=5)
    Button(root, text="Reintroduce Person", command=start_reintroduction_flow_thread, **reintroduce_button_style).pack(pady=5)
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
    start_gui()

