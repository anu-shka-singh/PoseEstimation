import cv2
import mediapipe as mp
import csv
# initialize Pose estimator
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

pose = mp_pose.Pose()
'''min_detection_confidence=0.5,
    min_tracking_confidence=0.5)'''
points=mp_pose.PoseLandmark  #key points
# create capture object
cap = cv2.VideoCapture('/Users/anushka/Desktop/video_mam.MOV')
frameNr = 0
f=open("landmarks_video.csv",'w')
f.close()
f1=open("landmarks_video.csv",'a')
fields=['X','Y','Z','Visibility','Points']
f1_writer=csv.writer(f1)
f1_writer.writerow(fields)
while cap.isOpened():
    # read frame from capture object
    success, frame = cap.read()

    try:
        # convert the frame to RGB format
        RGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # process the RGB frame to get the result
        #results = pose.process(RGB)
    
        
        results = pose.process(RGB)
        
        print((results.pose_landmarks))
        keypoints = []

        if results.pose_landmarks is None:
            f1.write("FRAME:"+str(frameNr)+"\n"+str(results.pose_landmarks))
        else:
            for data_point in results.pose_landmarks.landmark:
                keypoints.append({
                    'X': data_point.x,
                    'Y': data_point.y,
                    'Z': data_point.z,
                    'Visibility': data_point.visibility,
                         })
            f1.write("FRAME:"+str(frameNr)+"\n")
            for i,j in zip(keypoints,points):
                f1_writer.writerow([i['X'],i['Y'],i['Z'],i['Visibility'],j])
        #f1.write("FRAME:"+str(frameNr)+"\n"+str(results.pose_landmarks))

        # draw detected skeleton on the frame
        mp_drawing.draw_landmarks(
            frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # show the final output
        cv2.imshow('Output', frame)
    except:
        break
    if cv2.waitKey(1) == ord('q'):
        break
    frameNr=frameNr+1
f1.close()
cap.release()
cv2.destroyAllWindows()
