import cv2
import numpy as np
import os
from datetime import datetime
from mtcnn.mtcnn import MTCNN
from deepface import DeepFace

# Load known faces
def load_known_faces(known_faces_dir):
    known_faces = []
    known_names = []
    for filename in os.listdir(known_faces_dir):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            image = cv2.imread(os.path.join(known_faces_dir, filename))
            known_faces.append(image)
            known_names.append(os.path.splitext(filename)[0])
            print(f'Loaded {filename} as {known_names[-1]}')
    return known_faces, known_names

# Mark attendance
def mark_attendance(name):
    with open('attendance.csv', 'a') as f:
        f.write(f'{name},{datetime.now()}\n')

# Recognize faces using DeepFace
def recognize_face(face_image, known_faces, known_names):
    for idx, known_face in enumerate(known_faces):
        # Use DeepFace to find the closest match between detected and known faces
        result = DeepFace.verify(face_image, known_face, model_name='Facenet', enforce_detection=False)
        
        if result['verified']:
            return known_names[idx]
    
    return "Unknown"

# Main function
def main():
    known_faces_dir = 'known_faces'
    known_faces, known_names = load_known_faces(known_faces_dir)

    # Initialize the MTCNN detector
    detector = MTCNN()

    # Start video capture
    video_capture = cv2.VideoCapture("input_data/video.mp4")
    # Get video properties
    frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(video_capture.get(cv2.CAP_PROP_FPS))
    
    # Define the codec and create VideoWriter object to save the output
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Codec for .avi files
    output_video = cv2.VideoWriter('output_video.avi', fourcc, fps, (frame_width, frame_height))

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        # Detect faces in the frame using MTCNN
        faces = detector.detect_faces(frame)

        for face in faces:
            x, y, w, h = face['box']
            face_image = frame[y:y+h, x:x+w]

            # Recognize the face
            name = recognize_face(face_image, known_faces, known_names)

            # Mark attendance if recognized
            if name != "Unknown":
                mark_attendance(name)

            # Draw rectangle around the face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

        # Display the frame with detected faces
        cv2.imshow('Video', frame)
        output_video.write(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture and close all windows
    video_capture.release()
    output_video.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
