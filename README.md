# Face Detection with OpenCV and DNN

This project implements real-time face detection using OpenCV and a pre-trained Deep Neural Network (DNN) model.

## Features

- Face detection using Caffe-based DNN model
- Configurable confidence threshold for detection filtering
- Bounding box visualization with confidence scores
- Support for various image formats

## Requirements

- Python 3.6+
- OpenCV 4.x
- NumPy

## Installation

1. Clone this repository:
```bash
git clone <your-repo-url>
cd "Face Detection with OpenCV and DNN"
```

2. Install the required dependencies:
```bash
pip install opencv-python numpy
```

3. Download the pre-trained model files:
   - Download the Caffe model files from the [OpenCV face detection model](https://github.com/opencv/opencv/tree/master/samples/dnn/face_detector)
   - Place `deploy.prototxt` and `res10_300x300_ssd_iter_140000.caffemodel` in the project directory

## Usage

Run the face detection script with the following command:

```bash
python detect_faces.py -i <path-to-image> -p <path-to-prototxt> -m <path-to-model> -c <confidence-threshold>
```

### Parameters

- `-i, --image`: Path to the input image (required)
- `-p, --prototxt`: Path to Caffe 'deploy' prototxt file (required)
- `-m, --model`: Path to pre-trained model (required)
- `-c, --confidence`: Minimum probability to filter weak detections (optional, default: 0.5)

### Example

```bash
python detect_faces.py -i sample.jpg -p deploy.prototxt -m res10_300x300_ssd_iter_140000.caffemodel -c 0.7
```

## How it Works

1. **Model Loading**: The script loads a pre-trained Caffe DNN model for face detection
2. **Image Preprocessing**: Input images are resized to 300x300 pixels and normalized
3. **Detection**: The model processes the image and returns face detection results
4. **Filtering**: Detections below the confidence threshold are filtered out
5. **Visualization**: Bounding boxes and confidence scores are drawn on the image

## Model Information

This implementation uses the Single Shot Detector (SSD) framework with a ResNet-10 backbone, trained on the WIDER FACE dataset. The model is optimized for real-time face detection with good accuracy.

## License

This project is open source and available under the MIT License.

## Contributing

Feel free to submit issues and enhancement requests! 

## Personal Notes for Improvements
- Integrate with Raspberry Pi
- Facial and emotional gesture detection with AI feedback integration
