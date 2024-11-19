import os
import cv2
import face_recognition
import time

IMAGE_DIR = r"C:\Users\LuckyChinnu\Desktop\7th\Game_Proj\DB"

known_face_encodings = []
known_face_names = {}
person_info = {}

# Loop through each folder in the image directory
for person_name in os.listdir(IMAGE_DIR):
    person_folder = os.path.join(IMAGE_DIR, person_name)
    
    if not os.path.isdir(person_folder):
        continue
    
    # Load the info from the text file
    info_file_path = os.path.join(person_folder, 'info.txt')
    if os.path.isfile(info_file_path):
        try:
            with open(info_file_path, 'r') as file:
                info = file.read().strip()  # Strip to remove extra whitespace
            person_info[person_name] = info
        except Exception as e:
            print(f"Error reading info file for {person_name}: {e}")
    
    # Loop through each image in the person's folder
    for image_name in os.listdir(person_folder):
        image_path = os.path.join(person_folder, image_name)
        
        # Load the image if it has a valid image extension
        if image_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            try:
                image = face_recognition.load_image_file(image_path)
                face_encodings = face_recognition.face_encodings(image)
                if face_encodings:
                    known_face_encodings.append(face_encodings[0])
                    known_face_names[person_name] = person_name
            except Exception as e:
                print(f"Error loading image {image_path}: {e}")

# Start the webcam
video_capture = cv2.VideoCapture(0)

detected_count = 0
start_time = time.time()

while True:
    ret, frame = video_capture.read()
    if not ret:
        print("Failed to capture frame. Exiting...")
        break
    
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for face_encoding, face_location in zip(face_encodings, face_locations):
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        
        name = "Unknown"

        # Ensure that best_match_index is valid
        if len(face_distances) > 0:
            best_match_index = face_distances.argmin()
            
            if matches[best_match_index] and best_match_index < len(known_face_names):
                name = list(known_face_names.keys())[best_match_index]

        top, right, bottom, left = face_location
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)
        
        if name != "Unknown":
            print(f"Match found in folder: {name}")
            print(f"Details: {person_info.get(name, 'No details available')}")
            detected_count += 1
            time.sleep(10)

            if detected_count >= 5:
                print("Detected 5 persons. Exiting...")
                time.sleep(1)  # Optional delay before exit
                break

    cv2.imshow('Video', frame)

    # Check for 2 minutes elapsed
    if time.time() - start_time >= 120:
        print("2 minutes elapsed. Exiting...")
        break

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
