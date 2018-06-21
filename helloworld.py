from naoqi import ALProxy
# tts = ALProxy("ALTextToSpeech", "nao2.local", 9559)
tts = ALProxy("ALTextToSpeech", "10.0.7.12", 9559)
pic = ALProxy("RobocupVision", "10.0.7.13", 9559)
pic.takePicture("./", "test.png")
tts.say("Click")
#tts.say("Good day, sire!")
