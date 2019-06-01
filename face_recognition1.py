

import face1 as face1
import sys
import h5py
from PIL import Image
import numpy as np
import tensorflow as tf
import face_recognition
from sklearn.manifold import TSNE
import object_de_2 as od
import threading

print("Do You Want Train Data? (Y/N) : ")
ans_user = input()
mode = ""
if ans_user=="y" or ans_user == "Y":
    face1.create_embedded()
    print("Data are Trained...")


filename = 'tempFiles/embedded1.hdf5'
f = h5py.File(filename, 'r')
dic = {}
for raw in f:
    for j in f[raw]:
        dic[str(raw),int(j)] = f[raw][j][:] 
        
f.close()



embedded = np.load("tempFiles/embedding1.npy")
embedded2 = np.load("tempFiles/embedding2.npy")


filename = 'tempFiles/embedded2.hdf5'
f = h5py.File(filename, 'r')
mic = {}
for raw in f:
    for j in f[raw]:
        mic[str(raw),int(j)] = f[raw][j][:] 
        
f.close()
    


from sklearn.metrics import f1_score, accuracy_score

distances = [] # squared L2 distance between pairs
identical = [] # 1 if same identity, 0 otherwise
metadata = face1.metadata
num = len(metadata)



from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn import svm

targets = np.array([m.name for m in metadata])

encoder = LabelEncoder()
encoder.fit(targets)

# Numerical encoding of identities
y = encoder.transform(targets)

train_idx = np.arange(metadata.shape[0]) % 2 != 0
test_idx = np.arange(metadata.shape[0]) % 2 == 0

# 50 train examples of 10 identities (5 examples each)
X_train = embedded[train_idx]
# 50 test examples of 10 identities (5 examples each)
X_test = embedded[test_idx]

y_train = y[train_idx]
y_test = y[test_idx]

knn = KNeighborsClassifier(n_neighbors=1, metric='euclidean')

svc = svm.SVC(kernel='rbf', gamma=1.0, C=1.0)

knn.fit(X_train, y_train)
svc.fit(X_train, y_train)

acc_knn = accuracy_score(y_test, knn.predict(X_test))
acc_svc = accuracy_score(y_test, svc.predict(X_test))

print(f'KNN accuracy = {acc_knn}, SVM accuracy = {acc_svc}')



targets2 = np.array([m.name for m in metadata])

encoder1 = LabelEncoder()
encoder1.fit(targets2)

y = encoder1.transform(targets2)

train_idx = np.arange(metadata.shape[0]) % 2 != 0
test_idx = np.arange(metadata.shape[0]) % 2 == 0

# 50 train examples of 10 identities (5 examples each)
X_train = embedded2[train_idx]
# 50 test examples of 10 identities (5 examples each)
X_test = embedded2[test_idx]

y_train = y[train_idx]
y_test = y[test_idx]

knn1 = KNeighborsClassifier(n_neighbors=1, metric='euclidean')

svc1 = svm.SVC(kernel='rbf', gamma=1, C=1.0, probability=True)

knn1.fit(X_train, y_train)
svc1.fit(X_train, y_train)

acc_knn1 = accuracy_score(y_test, knn1.predict(X_test))
acc_svc1 = accuracy_score(y_test, svc1.predict(X_test))
print(f'KNN accuracy = {acc_knn1}, SVM accuracy = {acc_svc1}')



import warnings
import cv2
# Suppress LabelEncoder warning
warnings.filterwarnings('ignore')
from PIL import Image



def return_emd(img):
    img = face1.align_image(img)
    # scale RGB values to interval [0,1]
    img = (img / 255.).astype(np.float32)
    # obtain embedding vector for image
    myemd = face1.nn4_small2_pretrained.predict(np.expand_dims(img, axis=0))[0]
    
    return myemd

#cap.release()
def testing_run(frame):
    
    try:
        img= frame
        #face_locations = face_recognition.face_locations(img)
        face_locations = face_recognition.face_locations(image, number_of_times_to_upsample=0, model="cnn")

        #print("I found {} face(s) in this photograph.".format(len(face_locations)))

        for face_location in face_locations:

    # Print the location of each face in this image
            top, right, bottom, left = face_location
            #print("A face is located at pixel location Top: {}, Left: {}, Bottom: {}, Right: {}".format(top, left, bottom, right))

            # You can access the actual face itself like this:
            face_image = img[top:bottom, left:right]
            pil_image = Image.fromarray(face_image)
            
            
            example_idx = 20

            example_idx = 20
            example_image = np.asarray(pil_image)

            #example_image =cv2.cvtColor(example_image, cv2.COLOR_BGR2GRAY)
                    
            # obtain embedding vector for image
            myemd = return_emd(example_image)
            
            example_prediction = svc.predict([myemd])
            #imf = cv2.cvtColor(face_recognition.load_image_file("myimg.jpg"), cv2.COLOR_BGR2GRAY)
            fd = face1.face_recognition.face_encodings(example_image)[0]
            example_prediction1 = svc1.predict([fd])

            #print("svc1 pred: "+str(np.amax(svc1.predict_proba([fd]))))
            
            example_identity = encoder.inverse_transform(example_prediction)[0]
            example_identity1 = encoder1.inverse_transform(example_prediction1)[0]
            #print(str(example_identity)+" -- "+str(example_identity1))
            thresh =0.25
            #print(face1.distance(mic[(example_identity1,0)],fd))
            #print(face1.distance(mic[(example_identity1,1)],fd))
            
            if face1.distance(mic[(example_identity1,0)],fd) < thresh and face1.distance(mic[(example_identity1,1)],fd) < thresh :
                print(example_identity1)
                return example_identity1

                
            else :
                print("unknown")
                return "unknown"
            
        #plt.imshow(example_image)
        #plt.title(f'Recognized as {example_identity}');
    except Exception as e:
        print(e)
import time
#cam1.release()
class myThread1(threading.Thread):
    def __init__(self, image):
        threading.Thread.__init__(self)
        self.time = time
        self.image = image
    def run(self):
        name = ""
        global graph
        with graph.as_default():
            name = testing_run(self.image)
        print(name)

import pymysql

# Open database connection
db = pymysql.connect("localhost","root","","testing" )

# prepare a cursor object using cursor() method


# execute SQL query using execute() method.

# disconnect from server



class change_mode(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
    def run(self):
        cursor = db.cursor()
        cursor.execute("SELECT * from test")

        # Fetch a single row using fetchone() method.
        
        data = cursor.fetchone()
        global mode
        mode = str(data[0])
        print("Thread: "+mode)
        time.sleep(100)
            
c=0
try:
    if cam1.isOpened():
        cam1.release()
except:
    pass



cam1 = cv2.VideoCapture(0)


while True:
    thread_for_mode = change_mode()
    thread_for_mode.start()
    graph = tf.get_default_graph()
    _,img = cam1.read()
    print("Main Thread: "+mode)
    #time = str(datetime.now())
    name = od.detect_object(img)
    if name == "person":
    #testing_run(img)
        th1 = myThread1(img)
        th1.start()

db.close()