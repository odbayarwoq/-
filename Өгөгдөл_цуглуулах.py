import cv2
import numpy as np
import os
import torch
def visualize(input_image , faces, fps, thickness=2):
    if faces[1] is not None:
        for  idx, face in enumerate(faces[1]):
            coords = face[:-1].astype(np.int32)
            face_image = input_image[coords[1]:(coords[1]+coords[3]),coords[0]:(coords[0]+coords[2]),:]
            if face_image is not None:
               
               count_ims = len(os.listdir("./faces"))
               print(count_ims)
               cv2.imwrite("faces/test_{}.jpg".format(count_ims), face_image)
               
               cv2.rectangle(input_image, (coords[0],coords[1]), (coords[0]+coords[2],
                                                            coords[1]+coords[3]), (0, 255, 0), thickness)
            cv2.rectangle(input_image, (coords[0],coords[1]), (coords[0]+coords[2], 
                                                               coords[1]+coords[3]), (0, 255, 0), thickness) 
            cv2.circle(input_image, (coords[4], coords[5]), 2, (255, 0, 0), thickness)
            cv2.circle(input_image, (coords[6], coords[7]), 2, (0, 0, 255), thickness)
            cv2.circle(input_image, (coords[8], coords[9]), 2, (0, 255, 0), thickness)
            cv2.circle(input_image, (coords[10], coords[11]), 2, (255, 0, 255), thickness)
            cv2.circle(input_image, (coords[12], coords[13]), 2, (0, 255, 255), thickness)
    cv2.putText(input_image, 'FPS: {:.2f}'.format(fps), (1, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
cap = cv2.VideoCapture(r"C:\woq\facedetector\video\Untitled video - Made with Clipchamp.mp4")
detector = cv2.FaceDetectorYN.create(r"C:\woq\facedetector\YuNet_detector\face_detection_yunet_2023mar.onnx",
"",
    (640, 640),0.8,0.3, 5000
)
cv2.namedWindow("output", cv2.WINDOW_NORMAL)  
while True:
   ret, frame = cap.read()
   if ret:
    tm = cv2.TickMeter()
    img1Width = int(frame.shape[1])
    img1Height = int(frame.shape[0])
    img1 = cv2.resize(frame, (img1Width, img1Height))
    tm.start()
    detector.setInputSize((img1Width, img1Height))
    faces1 = detector.detect(img1)
    visualize(img1, faces1, tm.getFPS())
    tm.stop() 
    img1 = cv2.resize(img1, (1220, 680))       
    #img1 = cv2.resize(img1, (1920, 1080))           
    cv2.imshow('test', img1)
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break
cap.release() 
cv2.VideoCapture.release()
cv2.destroyAllWindows() 
