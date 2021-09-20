# raspberry camera object recognition

 Object recognition performed with raspberry pi 3 B+ / pi 4 B , using a raspberry camera and a mobilenet .h5 model.

 This python script takes as input the .h5 file and loads the Mobilenet model exploiting the keras API. The camera preview is shown on the screen, and each acquired frame is fed to the model. Then the output probabilities and the inference time are displayed on the preview through an overlay.

 This script was tested on raspberry pi 3 B+ and raspberry pi 4 model B, using a Raspberry Pi Official Camera Module V2 8Mp.

 You can download the .h5 file from this repo:  "https://github.com/fchollet/deep-learning-models/releases/tag/v0.6".

 If you download "mobilenet_1_0_224_tf.h5" from this link, you can use it without any change in the code of object_recognition.py .

 Before you start the script, you have to install pip, opencv, tensorflow 2.2 or higher, and keras.

 ---------------------------------------------------------------------------------

Usage example:

1) go to the directory in which you have this script and the .h5 file

2) Type this on the terminal:

python3 object_recognition.py mobilenet_1_0_224_tf.h5

