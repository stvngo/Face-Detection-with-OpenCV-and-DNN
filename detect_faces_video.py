from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
import cv2

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", required=True, 
                help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True, 
                help="path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
                help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# load our seerialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

# initialize the video stream and allow the camera sensor to warm up
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start() # index zero as src (computer camera)
time.sleep(2.0)

# loop over the frames from the video stream
while True:
    # grab the frame from the threaded video stream and resize it
    # to have a maximum width of 400 pixels
    frame = vs.read()
    if frame is None:
        print("[WARNING] Unable to grab frame from video stream")
        break
    frame = imutils.resize(frame, width=400)

    # grab the frame dimensions and convert it to a blob
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, 
                                (300, 300), (104.0, 177.0, 123.0))

    # pass the blob through the network and obtain the detections and
    # predictions
    net.setInput(blob)
    detections = net.forward()

    # loop over the detections (nearly identical to detect_faces image loop)
    for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with the
        # prediction
        confidence = detections[0, 0, i, 2]

        # filter out weak detections by ensuring the 'confidence' is 
        # greater than the minimum confidence
        if confidence < args["confidence"]:
            continue
        # comopute the (x, y)-coordinates of the bounding box for the 
        # object
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")

        # confidence string
        text = "{:.2f}%".format(confidence * 100)
        # shift text goes when off-image
        y = startY - 10 if startY - 10 > 10 else startY + 10 
        # draw the bounding box of the face
        cv2.rectangle(frame, (startX, startY), (endX, endY), 
                        (0, 0, 255), 2) 
        # params: (img, pt1 (top-left corner), pt2 (bottom-right corner), 
        # color as BGR tuple, thickness)
        cv2.putText(frame, text, (startX, int(y)), cv2.FONT_HERSHEY_SIMPLEX, 
                    1, (0, 0, 255), 2) # 0.45 by default
        # params: (img, text, bottom-keft corner of text string in image as 
        # tuple, fontFace, font scale, color as BGR tuple, thickness)
            
    # show the output image
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # if the 'q' or spacebar was pressed, break from the loop
    if key == ord("q") or key == 32:
        break

# perform cleanup
cv2.destroyAllWindows()
vs.stop()