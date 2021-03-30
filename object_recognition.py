import sys
if len(sys.argv) < 2:
    print('Usage: '+sys.argv[0]+' <h5file>')
    exit(0)
    
from keras.applications.mobilenet import MobileNet
from keras.preprocessing import image
from keras.applications.mobilenet import preprocess_input
from keras.applications.mobilenet import decode_predictions
from keras.models import Model
from keras.optimizers import SGD
import numpy as np
import cv2

import io
import re
import time

import picamera
from picamera import Color
from PIL import Image,ImageDraw, ImageFont

#camera resolution

CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480



weights_path = sys.argv[1]  #the .h5 file

#load the model from keras

model = MobileNet(input_shape=None, alpha=1.0, depth_multiplier=1, dropout=1e-3, include_top=True, weights=weights_path, input_tensor=None, pooling=None, classes=1000)

#start the preview with specified resolution and framerate
  
with picamera.PiCamera(resolution=(CAMERA_WIDTH, CAMERA_HEIGHT), framerate=30) as camera:
     camera.start_preview()
    
    
     overlay_renderer = None   #the overlay in which we write the predition and the inference time
     stream = io.BytesIO()   #the buffer in which we put the frames
      
     for _ in camera.capture_continuous(
            stream, format='jpeg', use_video_port=True):  #start to capture frames from the preview
            stream.seek(0)      #begin from the first position of the buffer
            img_ov = Image.new("RGBA", (CAMERA_WIDTH,CAMERA_HEIGHT)) #create the overlay using the same resolution of the camera
            draw = ImageDraw.Draw(img_ov)  #we are going to write on this overlay
            draw.font = ImageFont.truetype("/usr/share/fonts/truetype/freefont/FreeSans.ttf", 20)
            draw.rectangle([5,5,CAMERA_WIDTH-5,CAMERA_HEIGHT-5], fill=None, outline="red") #a purely decorative choice
            img = Image.open(stream).convert('RGB').resize((224, 224))  #we modify the frame to give it as input to the model
        
        #we then give the frame to the model and calculate the inference time
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0) 
            x = preprocess_input(x)
            start_time = time.monotonic()
            preds = model.predict(x)
            elapsed_ms = (time.monotonic() - start_time) * 1000
            preds= decode_predictions(preds)
            
        # we write the first two output probabilities onto the overlay
            draw.multiline_text((8,40), str(preds[0][0][1])+"\n"+str('%.1f' % ((preds[0][0][2])*100))+
                             " % \n", fill=(255,0,0,255), font=None, anchor=None, spacing=5, align="left")
            draw.multiline_text((8,85), str(preds[0][1][1])+"\n"+str('%.1f' % ((preds[0][1][2])*100))+
                             " % \n", fill=(255,0,0,255), font=None, anchor=None, spacing=5, align="left")
            draw.text((8,8),str('%.1f ms' % (elapsed_ms)), fill="red", font=None, anchor=None)
    
            if not overlay_renderer:
                """
                If overlay layer is not created yet, get a new one. Layer
                parameter must have 3 or higher number because the original
                preview layer has a # of 2 and a layer with smaller number will
                be obscured.
                """
                overlay_renderer = camera.add_overlay(img_ov.tobytes(),
                                                      layer=3,
                                                      size=img_ov.size,
                                                      alpha=0);
            else:
                overlay_renderer.update(img_ov.tobytes())
                
            print(preds[0][0], "inference time: ", elapsed_ms, "ms")

            stream.seek(0)
            stream.truncate()

  


