import cv2 as cv
from cv2 import VideoCapture
import matplotlib.pyplot as plt
from cv2 import imread
from cv2 import waitKey
from cv2 import imshow
import csv

net = cv.dnn.readNetFromTensorflow("graph_opt.pb")

inwidth = 368
inheight = 368
thr = 0.2

f=open("openpose.csv","w")
f.close()
BODY_PARTS = { "Nose": 0, "Neck": 1, "RightShoulder": 2, "RightElbow": 3, "Rightwrist": 4,
             "LeftShoulder": 5, "LeftElbow": 6, "LeftWrist": 7, "RightHip": 8, "RightKnee": 9, "RightAnkle": 10,
             "LeftHip": 11, "LeftKnee": 12, "LeftAnkle": 13, "RightEye": 14, "LeftEye": 15,
             "RightEar": 16, "LeftEar": 17, "Background": 18}

POSE_PAIRS = [["Nose","RightEye"],["Nose","LeftEye"],["RightEye","RightEar"],
            ["LeftEye","LeftEar"],["Neck","RightShoulder"],["Neck","LeftShoulder"],
            ["RightShoulder","RightElbow"],["RightElbow","Rightwrist"],["LeftShoulder","LeftElbow"],
            ["LeftElbow","LeftWrist"],["Neck","RightHip"],["RightHip","RightKnee"],["RightKnee","RightAnkle"]
            ,["Neck","LeftHip"],["LeftHip","LeftKnee"],["LeftKnee","LeftAnkle"]]

cap=VideoCapture('C:\\Users\\Jain\\Desktop\\video.MOV')
f1=open("openpose.csv",'a')
f1.write("Land Marks of Video Pose Estimtation\n\n")
fields = ['X','Y','key points']
f1_writer = csv.writer(f1)
f1_writer.writerow(fields)

if not cap.isOpened():
    cap=cv.VideoCapture(0)
    raise IOError("Video Cannot be Opened")
e=0

while(cv.waitKey(1)<0):
    hasFrame,frame=cap.read()
    if not hasFrame:
        cv.waitKey()
        break
    framewidth=frame.shape[1]
    frameheight=frame.shape[0]
    net.setInput(cv.dnn.blobFromImage(frame,1.0,(inwidth,inheight),(150,150,150),swapRB=False,crop=False))
    out = net.forward()
    out = out[:1, :19, :-1, :-1]  # MobileNet output [1, 57, -1, -1], we only need the first 19 elements

    assert(len(BODY_PARTS) == out.shape[1])

    points = []
    for i in range(len(BODY_PARTS)):
        # Slice heatmap of corresponging body's part.
        heatMap = out[0, i, :, :]

        # Originally, we try to find all the local maximums. To simplify a sample
        # we just find a global one. However only a single pose at the same time
        # could be detected this way.
        _, conf, _, point = cv.minMaxLoc(heatMap)
        x = (framewidth * point[0]) / out.shape[2]
        y = (frameheight * point[1]) / out.shape[3]
        # Add a point if it's confidence is higher than threshold.
        points.append((int(x), int(y)) if conf > thr else None)
        X_coordinate=[]
        Y_coordinate=[]
        for p in points:
            if(p!=None):
                for q in range(len(p)):
                    X_coordinate.append(p[0])
                    Y_coordinate.append(p[1])
        keypoints=[]
    
        for f in range(e,e+1):
            f1.write('frame'+str(f)+'\n')
            for a,b in zip(X_coordinate,Y_coordinate):
                keypoints.append({
                    'X':a,
                    'Y':b
                })
        e+=1
        for c,d in zip(keypoints,BODY_PARTS.keys()):
            f1_writer.writerow([c['X'],c['Y'],d])

    for pair in POSE_PAIRS:
        partFrom = pair[0]
        partTo = pair[1]
        assert(partFrom in BODY_PARTS)
        assert(partTo in BODY_PARTS)

        idFrom = BODY_PARTS[partFrom]
        idTo = BODY_PARTS[partTo]

        if points[idFrom] and points[idTo]:
            cv.line(frame, points[idFrom], points[idTo], (0, 255, 0), 3)
            cv.ellipse(frame, points[idFrom], (3, 3), 0, 0, 360, (0, 0, 255), cv.FILLED)
            cv.ellipse(frame, points[idTo], (3, 3), 0, 0, 360, (0, 0, 255), cv.FILLED)

    t, _ = net.getPerfProfile()
    freq = cv.getTickFrequency() / 1000
    cv.putText(frame, '%.2fms' % (t / freq), (10, 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

    cv.imshow("Video",frame)
f1.close()
# for finding frames of the video
vidcap = cv.VideoCapture('/Users/anushka/Desktop/video_mam.MOV')
success,image = vidcap.read()
count = 0
while success:
  cv.imwrite("frame%d.jpg" % count, image)           
  success,image = vidcap.read()
  count += 1
cap2 = cv.VideoCapture("/Users/anushka/Desktop/video_mam.MOV")