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
# For image
# def plot_face_blendshapes_bar_graph(face_blendshapes):
#   # Extract the face blendshapes category names and scores.
#   face_blendshapes_names = [face_blendshapes_category.category_name for face_blendshapes_category in face_blendshapes]
#   face_blendshapes_scores = [face_blendshapes_category.score for face_blendshapes_category in face_blendshapes]
#   # The blendshapes are ordered in decreasing score value.
#   face_blendshapes_ranks = range(len(face_blendshapes_names))

#   fig, ax = plt.subplots(figsize=(12, 12))
#   bar = ax.barh(face_blendshapes_ranks, face_blendshapes_scores, label=[str(x) for x in face_blendshapes_ranks])
#   ax.set_yticks(face_blendshapes_ranks, face_blendshapes_names)
#   ax.invert_yaxis()

#   # Label each bar with values
#   for score, patch in zip(face_blendshapes_scores, bar.patches):
#     plt.text(patch.get_x() + patch.get_width(), patch.get_y(), f"{score:.4f}", va="top")

#   ax.set_xlabel('Score')
#   ax.set_title("Face Blendshapes")
#   plt.tight_layout()
#   plt.show()
    
model_path = 'face_landmarker_v2_with_blendshapes.task'
BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
FaceLandmarkerResult = mp.tasks.vision.FaceLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode
frame_counter = 0
# Create a face landmarker instance with the live stream mode:
# def print_result(result: FaceLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
#     print('face landmarker result: {}'.format(result))

options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    output_face_blendshapes=True,
    running_mode=VisionRunningMode.IMAGE,
    num_faces=2)

detector = vision.FaceLandmarker.create_from_options(options)



cv2.namedWindow("Image", cv2.WINDOW_NORMAL) 
start_time = time.time()
cap = cv2.VideoCapture(0)

# All our execution goes into this loop
while cap.isOpened():
    # Reading the frames given by videoCapture
    success, frame = cap.read()
    if not success:
        break
    # Frame counter
    frame_counter +=1
    # Converting the from BGR to RGB
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    try:
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        detection_result = detector.detect(mp_image)
        annotated_image = draw_landmarks_on_image(mp_image.numpy_view(), detection_result)
        # Face emotion
        scores = [round(i.score,2) for i in detection_result.face_blendshapes[0]]

        # Happy
        # if (scores[19]>0.3 or scores[20]>0.3) and (scores[44]>0.3 or scores[45]>0.3) and (scores[48]>0.3 or scores[49]>0.3):
        if (scores[44]>0.3 or scores[45]>0.3):
            cv2.putText(annotated_image, "Happy",(300,70),cv2.FONT_HERSHEY_DUPLEX,1,(0, 255, 0),2)
        # Neutral
        # elif (scores[19]<0.30 or scores[20]<0.30) and (scores[23]<0.01 and scores[24]<0.01 and scores[25]<0.01 and scores[26]<0.01) and (scores[42]<0.05 and scores[43]<0.05)\
        # and (scores[9]<0.1 and scores[10]<0.1) and (scores[27]<0.01 and scores[34]<0.01 and scores[35]<0.01 and scores[44]<0.01 and scores[45]<0.01 and scores[48]<0.01 and scores[49]<0.01):
        #     cv2.putText(frame, "Neutral",(300,30),cv2.FONT_HERSHEY_DUPLEX,0.5,(255, 255, 255),2)
        # Fear
        # elif (scores[3]>0.5 and scores[40]>0.3) and (scores[3]>0.40 and (scores[46]>0.45 or scores[47]>0.45)) and (scores[3]>0.30 and (scores[48]>0.30 or scores[49]>0.32)):
        elif (scores[3]>0.7 and (scores[4]>0.4 or scores[5]>0.25) and (scores[11]>0.25 or scores[12]>0.24) and (scores[21]>0.15 or scores[22]>0.14) and scores[25]>0.55) or (scores[3]>0.23 and (scores[4]>0.09 or scores[5]>0.16) and (scores[11]>0.29 or scores[12]>0.27) and (scores[21]>0.03 or scores[22]>0.06) and scores[25]>0.05):
            cv2.putText(annotated_image, "Fear",(300,70),cv2.FONT_HERSHEY_DUPLEX,1,(255,0,0),2)
        # Blink
        elif (scores[9]>0.55 and scores[10]>0.6 and scores[11]>0.6 and scores[12]>0.6 and scores[19]>0.45 and scores[20]>0.5):
            cv2.putText(annotated_image, "Blink",(300,70),cv2.FONT_HERSHEY_DUPLEX,1,(255,215,0),2)
        # Tired
        elif ((scores[3]>0.35 or scores[4]>0.01) and scores[5]>0.08 and (scores[9]>0.02 or scores[10]>0.05) and (scores[11]>=0.02 or scores[12]>=0.04) and (scores[19]>=0.29 or scores[20]>=0.25) and scores[25]>=0.024) or ((scores[1]>=0.01 or scores[2]>=0.04) and (scores[9]>=0.1 or scores[10]>=0.09 or scores[11]>=0.04) and (scores[19]>=0.42 or scores[20]>=0.35) and (scores[28]>=0.01 or scores[29]>=0.09)):
            cv2.putText(annotated_image, "Tired",(300,70),cv2.FONT_HERSHEY_DUPLEX,1,(0, 0, 255),2)
        else:
            cv2.putText(annotated_image, "Neutral",(300,70),cv2.FONT_HERSHEY_DUPLEX,1,(255, 255, 255),2)
        # pointer to values
        # var = [f"{i.category_name}: {round(i.score,2)}" for i in detection_result.face_blendshapes[0][1:37:2]]
        # y=20
        # for item in var:
        #     cv2.putText(annotated_image, item,(20,y),cv2.FONT_HERSHEY_DUPLEX,0.6,(6,77,135),1)
        #     y+=20
    except IndexError:
        cv2.putText(annotated_image, "NO FACE",(170,250),cv2.FONT_HERSHEY_DUPLEX,2,(255, 0, 0),2)
        
    # calculating  frame per seconds FPS
    end_time = time.time()-start_time
    fps = frame_counter/end_time
    cv2.putText(annotated_image,f'FPS: {round(fps,1)}',(500,20),cv2.FONT_HERSHEY_DUPLEX,0.5,(0, 0, 0),2)
    cv2.resizeWindow("Image", (1200, 750)) 
    cv2.imshow('Image',cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
    
    if cv2.waitKey(25) == ord("q") or cv2.waitKey(25) == ord("Q"):
        break
cap.release()
cv2.destroyAllWindows()