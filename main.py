import cv2
import mediapipe as mp
import numpy as np



mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils
pose = mp_pose.Pose(enable_segmentation=True)

cap = cv2.VideoCapture("D:\\BU_FH_2023\\test.mp4")

while cap.isOpened():
    ret, img = cap.read()
    if not ret:
        break

    img = cv2.resize(img, (1280, 720))    #1280, 720
    result = pose.process(img)

    if result.pose_landmarks:
        mp_draw.draw_landmarks(img, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        h, w, c = img.shape
        opImg = np.zeros([h, w, c])
        opImg.fill(0)
        mp_draw.draw_landmarks(opImg, result.pose_landmarks, mp_pose.POSE_CONNECTIONS, mp_draw.DrawingSpec((0, 255, 0), 1, 1))
        cv2.imshow("Extracted Pose", opImg)

    cv2.imshow("Pose Estimation", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
