import sys
import h5py
import face1 as face1
from PIL import Image, ImageDraw
from PIL import Image
import tensorflow as tf
import numpy as np
import face_recognition
from sklearn.manifold import TSNE
import threading
import os
import time
import six.moves.urllib as urllib
import tarfile
import zipfile
from collections import defaultdict
from io import StringIO
import cv2
from handle import *
sys.path.append("..")
import warnings
# Suppress LabelEncoder warning
warnings.filterwarnings('ignore')

from utils import label_map_util

from utils import visualization_utils as vis_util

from db_conn import *



# print("Do You Want Train Data? (Y/N) : ")
# ans_user = input()
# if ans_user=="y" or ans_user == "Y":
#     face1.create_embedded()
#     print("Data are Trained...")



#############

# filename = 'tempFiles/embedded1.hdf5'
# f = h5py.File(filename, 'r')
# dic = {}
# for raw in f:
#     for j in f[raw]:
#         dic[str(raw),int(j)] = f[raw][j][:] 
        
# f.close()

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
# embedded = np.empty(num)
embedded2 = np.empty(num)

# embedded = np.load("tempFiles/embedding1.npy")
embedded2 = np.load("tempFiles/embedding2.npy")



################





from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn import svm
train_in =np.ones((metadata.shape[0]), dtype=bool)
test_in =np.ones((metadata.shape[0]), dtype=bool)
n1 =None
n2 =None
name=None
for i,m in enumerate(metadata):
    if name==None:
        name=m.name
    if name!=m.name:
        n1=None
        #n2=None
        name=None
    if n1 ==None :
        n1=1
        train_in[i]=False
        test_in[i]=True
        continue
#     if n2 ==None :
#         n2=1
#         train_in[i]=False
#         continue
    train_in[i]=True
    test_in[i]=False



targets2 = np.array([m.name for m in metadata])

encoder1 = LabelEncoder()
encoder1.fit(targets2)

y = encoder1.transform(targets2)


# 50 train examples of 10 identities (5 examples each)
X_train = embedded2[train_in]
# 50 test examples of 10 identities (5 examples each)
X_test = embedded2[test_in]

y_train = y[train_in]
y_test = y[test_in]

knn1 = KNeighborsClassifier(n_neighbors=3, metric='euclidean')

svc1 = svm.SVC(kernel='rbf', gamma=1, C=1.0, probability=True)

knn1.fit(X_train, y_train)
svc1.fit(X_train, y_train)

acc_knn1 = accuracy_score(y_test, knn1.predict(X_test))
acc_svc1 = accuracy_score(y_test, svc1.predict(X_test))
print(f'KNN accuracy = {acc_knn1}, SVM accuracy = {acc_svc1}')


##############
past_time = None
def retrain():
        now = datetime.datetime.now()
        global past_time
        if past_time == None:
                past_time = now
        time_diff =int((now-past_time).total_seconds())/60
        if time_diff > 15:
                past_time = now
                th.main_c.execute("select * from add_person")
                data = th.main_c.fetchall()
                th.main_c.execute("truncate table add_person")
                for d in data:
                        th.local_c.execute('insert into user_detail(name,no_img,last_update_date) values(%s,%s,%s)',(d[2],1,now.strftime("%Y-%m-%d")))
                        try:
                            os.system("mkdir images/"+str(d[2]))
                            os.system("mv temp_image/"+str(d[1])+" images/"+str(d[2])+"/1.jpg")
                        except Exeption as e:
                            os.system("md images/"+str(d[2]))
                            os.system("move temp_image/"+str(d[1])+" images/"+str(d[2])+"/1.jpg")

                th.main_c.execute("select * from for_asking where flag = 1")
                data1 = th.main_c.fetchall()
                th.main_c.execute("delete from for_asking where flag=1")
                for d1 in data1:
                        th.local_c.execute("select no_img from user_detail where name = %s",(d1[1]))
                        no_img = th.local_c.fetchone()
                        no = no_img[0] + 1
                        sf = len(os.listdir(os.path.join(path, d1[1])))
                        s =int(sf)+1
                        th.local_c.execute("update user_detail set no_img = %s",(no))
                        os.system("mv temp_image/"+str(d1[2])+" images/"+str(d1[1])+"/"+s+".jpg")
                #retrain model
                


################


# def return_emd(img44):
#     im = face1.align_image(img44)
#     # scale RGB values to interval [0,1]
#     im = (im / 255.).astype(np.float32)
#     # obtain embedding vector for image
#     myemd = face1.nn4_small2_pretrained.predict(np.expand_dims(im, axis=0))[0]
    
#     return myemd


###############

def testing_run(frame):
    
    #try:
        img= frame
        face_locations = face_recognition.face_locations(img)
        #print("I found {} face(s) in this photograph.".format(len(face_locations)))
        if len(face_locations) ==0 :
            print("noface")
            #data_maintain("noface",img,np.array(np.random(1,128)))
        else:
            cv2.imwrite("a.jpg",img)
            unknown_image = face_recognition.load_image_file("a.jpg")
            pp = Image.fromarray(unknown_image)
            draw = ImageDraw.Draw(pp) 
            a = []   
            for face_location in face_locations:

                # Print the location of each face in this image
                top, right, bottom, left = face_location
                # You can access the actual face itself like this:
                face_image = img[top:bottom, left:right]
                pil_image = Image.fromarray(face_image)
                draw.rectangle(((left-5, top-5), (right-5, bottom+5)), outline=(200, 0, 0))

                
                example_image = np.asarray(face_image)

                fd = face1.face_recognition.face_encodings(example_image)[0]
                example_prediction1 = svc1.predict([fd])

                example_identity1 = encoder1.inverse_transform(example_prediction1)[0]
                # Draw a label with a name below the face
                            
            
                thresh =0.30
                thresh2 =0.40
                
                if face1.distance(mic[(example_identity1,0)],fd) <thresh and face1.distance(mic[(example_identity1,1)],fd) <thresh:
                    print(example_identity1)
                    text_width, text_height = draw.textsize(str(example_identity1))

                    draw.rectangle(((left-5, bottom - text_height - 10-5), (right-5, bottom+5)), fill=(200, 0, 0), outline=(200, 0, 0))
                    draw.text((left + 6, bottom - text_height - 5), str(example_identity1), fill=(255, 255, 255, 255))
                    a.append(example_identity1)
                    
                   
                # elif face1.distance(mic[(example_identity1,0)],fd) < thresh2 and face1.distance(mic[(example_identity1,1)],fd) < thresh2 :
                #     print(example_identity1)
                #     print("2")
                #     for_asking(example_identity1, face_image,fd)
                else:
                    print("unknown")
                    a.append("unknown")
                    text_width, text_height = draw.textsize("Unknown")
                    draw.rectangle(((left, bottom - text_height - 10), (right, bottom)), fill=(0, 0, 255), outline=(0, 0, 255))
                    draw.text((left + 6, bottom - text_height - 5), "Unknown", fill=(255, 255, 255, 255))
                    
            del draw
            dx = PyMySQL.connect("localhost","root","","attendence" )
            cr= dx.cursor()
            now = datetime.datetime.now()
            cr.execute("INSERT INTO data(datetim) values(%s)",(now.strftime('%H:%M %Y-%m-%d')))
            dx.commit()
            for x in a:
                aq= ("UPDATE data SET {}=1 where datetim='{}'").format(x,now.strftime('%H:%M %Y-%m-%d'))
                cr.execute(aq)
                

            
            dx.commit()
            cr.close()
            dx.close()
            pp.show()
            # cv2.imshow('ImageWindow', img)
            # cv2.waitKey()

            
   
    # except Exception as e:
    #     print(e)

###################################

MODEL_NAME = 'ssd_mobilenet_v1_coco_2017_11_17'
MODEL_FILE = MODEL_NAME + '.tar.gz'
# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')

NUM_CLASSES = 90





tar_file = tarfile.open(MODEL_FILE)
for file in tar_file.getmembers():
  file_name = os.path.basename(file.name)
  if 'frozen_inference_graph.pb' in file_name:
    tar_file.extract(file, os.getcwd())

# ## Load a (frozen) Tensorflow model into memory.

# In[6]:

detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')



# In[7]:

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)



def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)





class myThread1(threading.Thread):
    def __init__(self, image, per):
        threading.Thread.__init__(self)
        self.time = time
        self.image = image
        self.per = per
        self.db = PyMySQL.connect("localhost","root","","main_db" )
        self.db1=PyMySQL.connect("localhost","root","","local_db" )
        self.main_c = self.db.cursor()
        self.local_c=self.db1.cursor()
 
        
    def run(self):
        testing_run(self.image)
        self.db.commit()
        self.db1.commit()
        self.main_c.close()
        self.local_c.close()
        self.db.close()
        self.db1.close()

        
         
        
            






def detection(img):
    image_np = img
    image_np_expanded = np.expand_dims(image_np, axis=0)
    #print(type(image_np_expanded))
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
    scores = detection_graph.get_tensor_by_name('detection_scores:0')
    classes = detection_graph.get_tensor_by_name('detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')
    (boxes, scores, classes, num_detections) = sess.run([boxes, scores, classes, num_detections],feed_dict={image_tensor: image_np_expanded})
    image_np,name1 = vis_util.visualize_boxes_and_labels_on_image_array(
        image_np,
        np.squeeze(boxes),
        np.squeeze(classes).astype(np.int32),
        np.squeeze(scores),
        category_index,
        use_normalized_coordinates=True,
        line_thickness=8)
    return name1






import requests
from io import BytesIO





#cam1 = cv2.VideoCapture('rtsp://192.168.43.1:8090/1')
sess = tf.Session(graph=detection_graph)
while True:
    try:
        print("Do You Want it again? (Y/N) : ")
        ans_user = input()
        if ans_user=="y" or ans_user == "Y":
            response = requests.get('http://192.168.15.148:8080/photo.jpg')
            img3 = Image.open(BytesIO(response.content))
            img = np.array(img3)
            #per = detection(img.copy())
            cv2.imwrite('aa.jpg',img)
            th1 = myThread1(img,'1')
            th1.start()
        
    except Exception as e:
        print(e)
    




db1.close()
db.close()
