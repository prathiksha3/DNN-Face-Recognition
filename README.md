# pip install opencv-python opencv-contrib-python

The code performs face detection and matching using ONNX models in conjunction
with OpenCV. Initially, the script utilizes ‘argparse‘ to handle command-line arguments
for specifying the paths to the reference and query images. These images are then read
using OpenCV’s ‘imread‘ function. To detect faces in the images, the script employs
a DNN-based face detector model, specified in ONNX format. The detected faces are
visualized using a ‘visualize‘ function, which draws bounding boxes and facial landmarks
on the images for better illustration.
The core functionality of face recognition is implemented using another ONNX model.
The script aligns and crops the detected faces, extracts their features, and compares them
to determine if they belong to the same person. This is achieved by calculating similar￾ity scores, including cosine and L2 distances, and comparing them against predefined
thresholds to ascertain identity matches. The results, including similarity scores and vi￾sualization of detected faces, are displayed to provide a comprehensive evaluation of the
models’ performance.
Overall, the script integrates argument parsing, image loading, face detection, visual￾ization, and face recognition into a cohesive workflow, leveraging advanced DNN model
and OpenCV’s capabilities to achieve high-accuracy face detection and recognition.
