import cv2
import numpy as np
import torch
from torchvision import transforms
from efficientnet_pytorch import EfficientNet
from deep_sort_realtime.deepsort_tracker import DeepSort


model_path = r'C:\woq\facedetector\YuNet_detector\my_model_10.pth'  
#model_path = r'C:\woq\facedetector\YuNet_detector\my_model_115.pth'
model = torch.load(model_path, map_location=torch.device('cpu'))
model.eval()


preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])


class_names = ['Face1', 'Face2', 'Face3', 'Face4', 'Face5', 'Face6', 'Face7', 'Face8', 'Unknown']




def visualize(input_image, faces, predicted_names, fps, thickness=2):
    for face, name in zip(faces, predicted_names):
        x, y, w, h = face[0:4].astype(int)
        cv2.rectangle(input_image, (x, y), (x+w, y+h), (0, 255, 0), thickness)
        cv2.putText(input_image, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


face_detector_model = r'C:\woq\facedetector\YuNet_detector\face_detection_yunet_2023mar.onnx'  
detector = cv2.FaceDetectorYN.create(face_detector_model, "", (640, 640), 0.5, 0.3, 5000)


tracker = DeepSort(max_age=5)


video_path = r'C:\woq\facedetector\video\Untitled video - Made with Clipchamp.mp4'  
#video_path = r'C:\woq\facedetector\video\selina.mp4'  


cap = cv2.VideoCapture(video_path)
#cap = cv2.VideoCapture(0)
cv2.namedWindow("output", cv2.WINDOW_NORMAL)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    img1Width, img1Height = frame.shape[1], frame.shape[0]
    detector.setInputSize((img1Width, img1Height))
    
    tm = cv2.TickMeter()
    tm.start()

    # Face detection
    faces = detector.detect(frame)
    predicted_names = []
    bbs = []

    if faces[1] is not None:
        for face in faces[1]:
            x, y, w, h = face[0:4].astype(int)
            face_img = frame[y:y+h, x:x+w]
            input_tensor = preprocess(face_img)
            input_batch = input_tensor.unsqueeze(0)

            with torch.no_grad():
                output = model(input_batch)
            
            probabilities = torch.nn.functional.softmax(output, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1)
            predicted_class_idx = predicted_class.item()

            if predicted_class_idx < len(class_names):
                predicted_class_name = class_names[predicted_class_idx] + " " +  "{:.2f}".format(probabilities[0,predicted_class_idx].item())
            else:
                predicted_class_name = "Unknown"

            predicted_names.append(predicted_class_name)

            # Format for DeepSort
            confidence = 1.0  # Adjust as needed
            detection_class = 0  # Adjust if you have different classes
            bbs.append(([x, y, w, h], confidence, detection_class))

    # Update tracks with DeepSort
    tracks = tracker.update_tracks(bbs, frame=frame)

    # Process each track
    for track in tracks:
        if not track.is_confirmed():
            continue

        track_id = track.track_id
        ltrb = track.to_ltrb()

        # Draw tracking information
        #cv2.rectangle(frame, (int(ltrb[0]), int(ltrb[1])), (int(ltrb[2]), int(ltrb[3])), (255, 0, 0), 2)
        #cv2.putText(frame, f"ID: {track_id}", (int(ltrb[0]), int(ltrb[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Visualize the results
    visualize(frame, faces[1], predicted_names, tm.getFPS())

    tm.stop()
    cv2.putText(frame, 'FPS: {:.2f}'.format(tm.getFPS()), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv2.imshow('output', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
