
import cv2
#import os module for reading training data directories and paths
import os
#import numpy to convert python lists to numpy arrays as 
#it is needed by OpenCV face recognizers
import numpy as np
import face_recognition
cv2.startWindowThread()


# ### Training Data

# In[2]:


#function to detect face using OpenCV
def detect_face(img):
    #convert the test image to gray image as opencv face detector expects gray images
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    #load OpenCV face detector, I am using LBP which is fast
    #there is also a more accurate but slow Haar classifier
    face_locations = face_recognition.face_locations(gray)
        
    if len(face_locations) ==0 :
        return None
    for face_location in face_locations:
        top, right, bottom, left = face_location
        # You can access the actual face itself like this:
        face_image = gray[top:bottom, left:right]
    #let's detect multiscale (some images may be closer to camera than others) images
    #result is a list of faces
    
    #return only the face part of the image
    return face_image


# In[3]:


label_text =[]
def prepare_traindata(path):
    global label_text
    label_text =[]
    faces=[]
    
    labels=[]
    c=0
    for i in os.listdir(path):
        label_text.append(i)
        for f in os.listdir(os.path.join(path, i)):
            ext = os.path.splitext(f)[1]
            if ext == '.jpg' or ext == '.jpeg':
                image_path = path + "/" + i+ "/" +f
                image = cv2.imread(image_path)
                 #detect face
                face= detect_face(image)

                if face is not None:
                    #add face to list of faces
                    faces.append(face)
                    #add label for this face
                    labels.append(c)

        c+=1
    return faces, labels
                
        



faces, labels = prepare_traindata("images")


face_recognizer = cv2.face.createLBPHFaceRecognizer()


face_recognizer.train(faces, np.array(labels))
print("#LBPH algo trained.")



def predict(test_img):
    img = test_img.copy()

    label= face_recognizer.predict(img)
    print(label_text[label[0]])
    
    return img



print("Predicting images...")

test_img1 = cv2.imread("test-data/test1.jpg")
tface=detect_face(test_img1)
predict(tface)

    
        

