# Required libraries
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
import time
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path


from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import matplotlib.pyplot as plt


def draw_landmarks_on_image(rgb_image, detection_result):
  face_landmarks_list = detection_result.face_landmarks
  annotated_image = np.copy(rgb_image)

  # Loop through the detected faces to visualize.
  for idx in range(len(face_landmarks_list)):
    face_landmarks = face_landmarks_list[idx]

    # Draw the face landmarks.
    face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    face_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in face_landmarks
    ])

    solutions.drawing_utils.draw_landmarks(
        image=annotated_image,
        landmark_list=face_landmarks_proto,
        connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp.solutions.drawing_styles
        .get_default_face_mesh_tesselation_style())
    solutions.drawing_utils.draw_landmarks(
        image=annotated_image,
        landmark_list=face_landmarks_proto,
        connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp.solutions.drawing_styles
        .get_default_face_mesh_contours_style())
    solutions.drawing_utils.draw_landmarks(
        image=annotated_image,
        landmark_list=face_landmarks_proto,
        connections=mp.solutions.face_mesh.FACEMESH_IRISES,
          landmark_drawing_spec=None,
          connection_drawing_spec=mp.solutions.drawing_styles
          .get_default_face_mesh_iris_connections_style())

  return annotated_image
    
import mediapipe as mp
model_path = 'face_landmarker_v2_with_blendshapes.task'
BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
FaceLandmarkerResult = mp.tasks.vision.FaceLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

options = vision.FaceLandmarkerOptions(base_options=BaseOptions(model_asset_path=model_path),
                                           output_face_blendshapes=True,
                                           running_mode=VisionRunningMode.IMAGE
                                           )
detector = vision.FaceLandmarker.create_from_options(options)

# Argument parser for videocapture
parser = argparse.ArgumentParser()
parser.add_argument('-f', "--file_path", type=Path)
p = parser.parse_args()
if str(p.file_path) == 'cam':
    pth = 0
else:
    pth = p.file_path


cv2.namedWindow("Image", cv2.WINDOW_NORMAL) 
start_time = time.time()
cap = cv2.VideoCapture(pth)
fr=0
temp=True
# All our execution goes into this loop
while cap.isOpened():
    # start_time = time. time()
    # Reading the frames given by videoCapture
    success, frame = cap.read()
    if not success:
        break
    # Frame counter
    fr +=1
    # Converting the from BGR to RGB
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

    # Detect face landmarks from the input image.
    detection_result = detector.detect(mp_image)
    coords = detection_result.face_landmarks[0][0]
    if len(detection_result.face_blendshapes) != 0:
        # mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        annotated_image = draw_landmarks_on_image(mp_image.numpy_view(), detection_result)

        # Face emotion
        scores = [round(j.score,2) for j in detection_result.face_blendshapes[0]]

        # Happy
        if (scores[44]>0.3 or scores[45]>0.3):
            emotion="Happy"
            start_time_emotion = time.time()
            start_full_time_emotion = time.ctime()
            cv2.putText(annotated_image, emotion,(round(coords.x*1000)-250,round(coords.y*100)),cv2.FONT_HERSHEY_DUPLEX,1,(0, 255, 0),2)
        # Fear
        # elif (scores[3]>0.5 and scores[40]>0.3) and (scores[3]>0.40 and (scores[46]>0.45 or scores[47]>0.45)) and (scores[3]>0.30 and (scores[48]>0.30 or scores[49]>0.32)):
        elif (scores[3]>0.7 and (scores[4]>0.4 or scores[5]>0.25) and (scores[11]>0.25 or scores[12]>0.24) and (scores[21]>0.15 or scores[22]>0.14) and scores[25]>0.55) or (scores[3]>0.23 and (scores[4]>0.09 or scores[5]>0.16) and (scores[11]>0.29 or scores[12]>0.27) and (scores[21]>0.03 or scores[22]>0.06) and scores[25]>0.05):
            emotion="Fear"
            start_time_emotion = time.time()
            start_full_time_emotion = time.ctime()
            cv2.putText(annotated_image, emotion,(round(coords.x*1000)-250,round(coords.y*100)),cv2.FONT_HERSHEY_DUPLEX,1,(255,0,0),2)
        # Blink
        elif (scores[9]>0.55 and scores[10]>0.6 and scores[11]>0.6 and scores[12]>0.6 and scores[19]>0.45 and scores[20]>0.5):
            emotion="Blink"
            start_time_emotion = time.time()
            start_full_time_emotion = time.ctime()
            cv2.putText(annotated_image, emotion,(round(coords.x*1000)-250,round(coords.y*100)),cv2.FONT_HERSHEY_DUPLEX,1,(255,215,0),2)
        # Tired
        elif ((scores[3]>0.35 or scores[4]>0.01) and scores[5]>0.08 and (scores[9]>0.02 or scores[10]>0.05) and (scores[11]>=0.02 or scores[12]>=0.04) and (scores[19]>=0.29 or scores[20]>=0.25) and scores[25]>=0.024) or ((scores[1]>=0.01 or scores[2]>=0.04) and (scores[9]>=0.1 or scores[10]>=0.09 or scores[11]>=0.04) and (scores[19]>=0.42 or scores[20]>=0.35) and (scores[28]>=0.01 or scores[29]>=0.09)):
            emotion="Tired"
            start_time_emotion = time.time()
            start_full_time_emotion = time.ctime()
            cv2.putText(annotated_image, emotion,(round(coords.x*1000)-250,round(coords.y*100)),cv2.FONT_HERSHEY_DUPLEX,1,(0, 0, 255),2)
        else:
            emotion="Neutral"
            start_time_emotion = time.time()
            start_full_time_emotion = time.ctime()
            cv2.putText(annotated_image, emotion,(round(coords.x*1000)-250,round(coords.y*100)),cv2.FONT_HERSHEY_DUPLEX,1,(255, 255, 255),2)

        # During
        if temp:
            old_emotion=emotion
            st_time_emt=start_time_emotion
            st_full_time_emt = start_full_time_emotion
            temp=False
        if old_emotion!=emotion:
            # very_old_emotion=old_emotion
            time_counter = fr/30
            print(time_counter)
            if time_counter>300:
                with open("data.txt","a") as f:
                    f.write(f"{old_emotion},{st_full_time_emt},{fr/30+st_time_emt}\n")
                print(st_time_emt)
                old_emotion=emotion
                st_time_emt=start_time_emotion
                st_full_time_emt=start_full_time_emotion
        emt=True
    # if emt:
    if len(detection_result.face_blendshapes) == 0 and emt:
        with open("data.txt","a") as f:
            f.write(f"{old_emotion},{st_full_time_emt},{fr/30+st_time_emt}\n")
        emt=False

    # calculating  frame per seconds FPS
    # end_time = fr/30+st_time_emt
    # fps = fr/end_time
    # cv2.putText(annotated_image,f'FPS: {round(fps,1)}',(520,20),cv2.FONT_HERSHEY_DUPLEX,0.7,(0, 0, 0),2)
    # frame_counter=0
    # cv2.resizeWindow("Image", (1200, 750))
    cv2.imshow('Image',cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))

    if cv2.waitKey(25) == ord("q") or cv2.waitKey(25) == ord("Q"):
        break
cap.release()
cv2.destroyAllWindows()
