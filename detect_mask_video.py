from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import imutils
import time
import cv2
import os

# Load our serialized face detector model from disk
prototxtPath = r"face_detector/deploy.prototxt"
weightsPath = r"face_detector/res10_300x300_ssd_iter_140000.caffemodel"

faceNet = cv2.dnn.readNet(model=weightsPath, config=prototxtPath, framework='Caffe')

# Load the face mask detector model from disk
maskNet = load_model('mask_detector.model')

min_confidence = 0.5

def detect_and_predict_mask(frame, faceNet, maskNet):
    # grab the dimensions of the frame and then construct a blob from it
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1, (224, 224), (104, 117, 123))
    
    # pass the blob through the network and obtain the face detections
    faceNet.setInput(blob)
    detections = faceNet.forward()

    # initialize our list of faces, their corresponding locations,
	# and the list of predictions from our face mask network
    faces = []
    locs = []
    preds = []

    for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with the detection
        confidence = detections[0, 0, i, 2]
        # filter out weak detections by ensuring the confidence is
		# greater than the minimum confidence
        if confidence > min_confidence:
            # compute the (x, y)-coordinates of the bounding box for
			# the object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            # ensure the bounding boxes fall within the dimensions of
			# the frame
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w-1, endX), min(h-1, endY))
            # extract the face ROI, convert it from BGR to RGB channel
			# ordering, resize it to 224x224, and preprocess it
            face = frame[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)
            
            # add the face and bounding boxes to their respective lists
            faces.append(face)
            locs.append((startX, startY, endX, endY))
            
    # only make a predictions if at least one face was detected
    if len(faces) > 0:
        # for faster inference we'll make batch predictions on *all*
		# faces at the same time rather than one-by-one predictions
		# in the above `for` loop
        faces = np.array(faces, dtype="float32")
        preds = maskNet.predict(faces, batch_size=32)
    return (locs, preds)
print("[INFO] starting video stream ... ")
vs = VideoStream(src=2).start()

while True:
    start = time.time()

    frame = vs.read()
    frame = imutils.resize(frame)
    
    locs, preds = detect_and_predict_mask(frame, faceNet, maskNet)
    
    for (box, pred) in zip(locs, preds):
        (startX, startY, endX, endY) = box
        (mask, withoutMask) = pred
        label = "Mask" if mask > withoutMask else "No Mask"
        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

        # include the probability in the label
        label = "{} - Confidence: {:.2f}%".format(label, max(mask, withoutMask)*100)

        cv2.putText(frame, label, (startX, startY-10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

    end = time.time()
    fps = "FPS:" + str(int(1/(end - start)))
    cv2.putText(frame, fps, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0, 0), 3)


    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cv2.destroyAllWindows()
vs.stop()