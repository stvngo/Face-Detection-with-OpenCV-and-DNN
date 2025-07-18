import numpy as np
import argparse
import cv2

# construct the argument parse and parse the command line arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, 
                help="path to input image")
ap.add_argument("-p", "--prototxt", required=True, 
                help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True, 
                help="path to pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.5, 
                help="minimum probability to filter weak detections") # optional
args = vars(ap.parse_args())

# load our serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

# load the input image and construct an input blob for the image
# by resizing to a fixed 300x300 pixels and then normalizing it
image = cv2.imread(args["image"])
if image is None:
    raise FileNotFoundError(f"Image file '{args['image']}' not found or could not be opened.")
(h, w) = image.shape[:2] # extract dimensions
blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, 
                             (300, 300), (104.0, 177.0, 123.0))
''' dnn.blobFromImage function takes care of pre-processing which 
includes setting the blob dimensions and normalization'''

# pass the blob through the network and obtain the detections and
# predictions
print("[INFO] computing object detections")
net.setInput(blob)
detections = net.forward()

# loop over the detections 
for i in range(0, detections.shape[2]):
    # extract the confidence (i.e., probability) associated with the
    # prediction
    confidence = detections[0, 0, i, 2]

    # filter out weak detections by ensuring the 'confidence' is 
    # greater than the minimum confidence
    if confidence > args["confidence"]:
        # comopute the (x, y)-coordinates of the bounding box for the 
        # object
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")

        # confidence string
        text = "{:.2f}%".format(confidence * 100)
        # shift text goes when off-image
        y = startY - 10 if startY - 10 > 10 else startY + 10 
        # draw the bounding box of the face
        cv2.rectangle(image, (startX, startY), (endX, endY), 
                      (0, 0, 255), 2) 
        # params: (img, pt1 (top-left corner), pt2 (bottom-right corner), 
        # color as BGR tuple, thickness)
        cv2.putText(image, text, (startX, int(y)), cv2.FONT_HERSHEY_SIMPLEX, 
                    1, (0, 0, 255), 2) # 0.45 by default
        # params: (img, text, bottom-keft corner of text string in image as 
        # tuple, fontFace, font scale, color as BGR tuple, thickness)
        
# show the output image
cv2.imshow("Output", image)
cv2.waitKey(0)