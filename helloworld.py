from naoqi import ALProxy
import numpy as np
import cv2
import matplotlib.pyplot as plot



# tts = ALProxy("ALTextToSpeech", "nao2.local", 9559)
# tts = ALProxy("ALTextToSpeech", "10.0.7.16", 9559)
# tts.say("Click")
vision = ALProxy('RobocupVision', '10.0.7.15', 9559)

cameraId = 0 # 0 for topcam
data = vision.getBGR24Image(cameraId)
image = np.fromstring(data, dtype=np.uint8).reshape((480, 640, 3))
rgb_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
plot.imshow(rgb_img)
print "done"