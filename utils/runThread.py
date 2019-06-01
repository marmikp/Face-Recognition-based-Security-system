import threading
import face_recognition1 as fr
import cv2
class myThread (threading.Thread):
	def __init__(self, image, name):
		threading.Thread.__init__(self)
		self.image = image
		self.name = name
	def run(self):

    	#cv2.imshow('threAD', cv2.resize(self.image, (800,600)))		
		print("Point 1")
		recog_name = fr.testing_run(self.image)
		print("Point 2")
		print(str(recog_name))
		print("Point 3")