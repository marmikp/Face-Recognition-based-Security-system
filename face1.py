

import bz2
import os
import face_recognition
import json
import h5py
from urllib.request import urlopen
from array import *

import numpy as np
import os.path
from db_conn import *
import datetime
# disconnect from server




def download_landmarks(dst_file):
    url = 'http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2'
    decompressor = bz2.BZ2Decompressor()
    
    with urlopen(url) as src, open(dst_file, 'wb') as dst:
        data = src.read(1024)
        while len(data) > 0:
            dst.write(decompressor.decompress(data))
            data = src.read(1024)

dst_dir = 'models'
dst_file = os.path.join(dst_dir, 'landmarks.dat')

if not os.path.exists(dst_file):
    os.makedirs(dst_dir)
    download_landmarks(dst_file)


# ### CNN architecture and training
# 
# The CNN architecture used here is a variant of the inception architecture [[2]](https://arxiv.org/abs/1409.4842). More precisely, it is a variant of the NN4 architecture described in [[1]](https://arxiv.org/abs/1503.03832) and identified as [nn4.small2](https://cmusatyalab.github.io/openface/models-and-accuracies/#model-definitions) model in the OpenFace project. This notebook uses a Keras implementation of that model whose definition was taken from the [Keras-OpenFace](https://github.com/iwantooxxoox/Keras-OpenFace) project. The architecture details aren't too important here, it's only useful to know that there is a fully connected layer with 128 hidden units followed by an L2 normalization layer on top of the convolutional base. These two top layers are referred to as the *embedding layer* from which the 128-dimensional embedding vectors can be obtained. The complete model is defined in [model.py](model.py) and a graphical overview is given in [model.png](model.png). A Keras version of the nn4.small2 model can be created with `create_model()`.

# In[2]:


from model import create_model

nn4_small2 = create_model()


# Model training aims to learn an embedding $f(x)$ of image $x$ such that the squared L2 distance between all faces of the same identity is small and the distance between a pair of faces from different identities is large. This can be achieved with a *triplet loss* $L$ that is minimized when the distance between an anchor image $x^a_i$ and a positive image $x^p_i$ (same identity) in embedding space is smaller than the distance between that anchor image and a negative image $x^n_i$ (different identity) by at least a margin $\alpha$.
# 
# $$L = \sum^{m}_{i=1} \large[ \small {\mid \mid f(x_{i}^{a}) - f(x_{i}^{p})) \mid \mid_2^2} - {\mid \mid f(x_{i}^{a}) - f(x_{i}^{n})) \mid \mid_2^2} + \alpha \large ] \small_+$$
# 
# $[z]_+$ means $max(z,0)$ and $m$ is the number of triplets in the training set. The triplet loss in Keras is best implemented with a custom layer as the loss function doesn't follow the usual `loss(input, target)` pattern. This layer calls `self.add_loss` to install the triplet loss:

# In[3]:


from keras import backend as K
from keras.models import Model
from keras.layers import Input, Layer

# Input for anchor, positive and negative images
in_a = Input(shape=(96, 96, 3))
in_p = Input(shape=(96, 96, 3))
in_n = Input(shape=(96, 96, 3))

# Output for anchor, positive and negative embedding vectors
# The nn4_small model instance is shared (Siamese network)
emb_a = nn4_small2(in_a)
emb_p = nn4_small2(in_p)
emb_n = nn4_small2(in_n)

class TripletLossLayer(Layer):
    def __init__(self, alpha, **kwargs):
        self.alpha = alpha
        super(TripletLossLayer, self).__init__(**kwargs)
    
    def triplet_loss(self, inputs):
        a, p, n = inputsprint(example_prediction+" -- "+example_prediction1)
        p_dist = K.sum(K.square(a-p), axis=-1)
        n_dist = K.sum(K.square(a-n), axis=-1)
        return K.sum(K.maximum(p_dist - n_dist + self.alpha, 0), axis=0)
    
    def call(self, inputs):
        loss = self.triplet_loss(inputs)
        self.add_loss(loss)
        return loss

# Layer that computes the triplet loss from anchor, positive and negative embedding vectors
#triplet_loss_layer = TripletLossLayer(alpha=0.2, name='triplet_loss_layer')([emb_a, emb_p, emb_n])

# Model that can be trained with anchor, positive negative images
#nn4_small2_train = Model([in_a, in_p, in_n], triplet_loss_layer)


# During training, it is important to select triplets whose positive pairs $(x^a_i, x^p_i)$ and negative pairs $(x^a_i, x^n_i)$ are hard to discriminate i.e. their distance difference in embedding space should be less than margin $\alpha$, otherwise, the network is unable to learn a useful embedding. Therefore, each training iteration should select a new batch of triplets based on the embeddings learned in the previous iteration. Assuming that a generator returned from a `triplet_generator()` call can generate triplets under these constraints, the network can be trained with:

# In[4]:


from data import triplet_generator

# triplet_generator() creates a generator that continuously returns 
# ([a_batch, p_batch, n_batch], None) tuples where a_batch, p_batch 
# and n_batch are batches of anchor, positive and negative RGB images 
# each having a shape of (batch_size, 96, 96, 3).
generator = triplet_generator() 

#nn4_small2_train.compile(loss=None, optimizer='adam')
#nn4_small2_train.fit_generator(generator, epochs=10, steps_per_epoch=100)

# Please note that the current implementation of the generator only generates 
# random image data. The main goal of this code snippet is to demonstrate 
# the general setup for model training. In the following, we will anyway 
# use a pre-trained model so we don't need a generator here that operates 
# on real training data. I'll maybe provide a fully functional generator
# later.


# The above code snippet should merely demonstrate how to setup model training. But instead of actually training a model from scratch we will now use a pre-trained model as training from scratch is very expensive and requires huge datasets to achieve good generalization performance. For example, [[1]](https://arxiv.org/abs/1503.03832) uses a dataset of 200M images consisting of about 8M identities. 
# 
# The OpenFace project provides [pre-trained models](https://cmusatyalab.github.io/openface/models-and-accuracies/#pre-trained-models) that were trained with the public face recognition datasets [FaceScrub](http://vintage.winklerbros.net/facescrub.html) and [CASIA-WebFace](http://arxiv.org/abs/1411.7923). The Keras-OpenFace project converted the weights of the pre-trained nn4.small2.v1 model to [CSV files](https://github.com/iwantooxxoox/Keras-OpenFace/tree/master/weights) which were then [converted here](face-recognition-convert.ipynb) to a binary format that can be loaded by Keras with `load_weights`:

# In[5]:


# nn4_small2_pretrained = create_model()
# nn4_small2_pretrained.load_weights('weights/nn4.small2.v1.h5')


# ### Custom dataset

# To demonstrate face recognition on a custom dataset, a small subset of the [LFW](http://vis-www.cs.umass.edu/lfw/) dataset is used. It consists of 100 face images of [10 identities](images). The metadata for each image (file and identity name) are loaded into memory for later processing.

# In[6]:



class IdentityMetadata():
    def __init__(self, base, name, file):
        # dataset base directory
        self.base = base
        # identity name
        self.name = name
        # image file name
        self.file = file

    def __repr__(self):
        return self.image_path()

    def image_path(self):
        return os.path.join(self.base, self.name, self.file) 
    
def load_metadata(path):
    metadata = []

    for i in os.listdir(path):
        for f in os.listdir(os.path.join(path, i)):
            # Check file extension. Allow only jpg/jpeg' files.
            ext = os.path.splitext(f)[1]
            if ext == '.jpg' or ext == '.jpeg':
                metadata.append(IdentityMetadata(path, i, f))
                
        
    return np.array(metadata)

metadata = load_metadata('images')

# ### Face alignment

# The nn4.small2.v1 model was trained with aligned face images, therefore, the face images from the custom dataset must be aligned too. Here, we use [Dlib](http://dlib.net/) for face detection and [OpenCV](https://opencv.org/) for image transformation and cropping to produce aligned 96x96 RGB face images. By using the [AlignDlib](align.py) utility from the OpenFace project this is straightforward:

# In[7]:


import cv2
#import matplotlib.pyplot as plt
#import matplotlib.patches as patches

from align import AlignDlib

#get_ipython().run_line_magic('matplotlib', 'inline')

def load_image(path):
    img = cv2.imread(path, 1)
    # OpenCV loads images with color channels
    # in BGR order. So we need to reverse them
    return img[...,::-1]

# Initialize the OpenFace face alignment utility
alignment = AlignDlib('models/landmarks.dat')

# Load an image of Jacques Chirac
jc_orig = load_image(metadata[6].image_path())

# Detect face and return bounding box
bb = alignment.getLargestFaceBoundingBox(jc_orig)

# Transform image using specified face landmark indices and crop image to 96x96
jc_aligned = alignment.align(96, jc_orig, bb, landmarkIndices=AlignDlib.OUTER_EYES_AND_NOSE)

# Show original image
# plt.subplot(131)
# plt.imshow(jc_orig)

# # Show original image with bounding box
# plt.subplot(132)
# plt.imshow(jc_orig)
# plt.gca().add_patch(patches.Rectangle((bb.left(), bb.top()), bb.width(), bb.height(), fill=False, color='red'))

# # Show aligned image
# plt.subplot(133)
# plt.imshow(jc_aligned);


# As described in the OpenFace [pre-trained models](https://cmusatyalab.github.io/openface/models-and-accuracies/#pre-trained-models) section, landmark indices `OUTER_EYES_AND_NOSE` are required for model nn4.small2.v1. Let's implement face detection, transformation and cropping as `align_image` function for later reuse.

# In[8]:


def align_image(img):
    return alignment.align(96, img, alignment.getLargestFaceBoundingBox(img), 
                           landmarkIndices=AlignDlib.OUTER_EYES_AND_NOSE)


# ### Embedding vectors

# Embedding vectors can now be calculated by feeding the aligned and scaled images into the pre-trained network.

# In[39]:

embedded = np.zeros((metadata.shape[0], 128))
embedded2 = np.zeros((metadata.shape[0], 128))

def distance(emb1, emb2):
    return np.sum(np.square(emb1 - emb2))
    #return face_recognition.compare_faces(known_faces, unknown_face_encoding)

def show_pair(idx1, idx2):
    plt.figure(figsize=(8,3))
    plt.suptitle(f'Distance = {distance(embedded2[idx1], embedded2[idx2]):.2f}')
    plt.subplot(121)
    plt.imshow(load_image(metadata[idx1].image_path()))
    plt.subplot(122)
    plt.imshow(load_image(metadata[idx2].image_path()));

#Creating Embedding
def create_embedded():
    
    print("Trainining.......")
    dic = {}
    j=0
    dic1 = {}

    # f1 = h5py.File('tempFiles/embedded1.hdf5','w')
    # grp1 = f1
    # lstr1=""



    # for i, m in enumerate(metadata):
    #     img = load_image(m.image_path())
    #     img = align_image(img)
    #     # scale RGB values to interval [0,1]
    #     img = (img / 255.).astype(np.float32)
    #     # obtain embedding vector for image
    #     embedded[i] = nn4_small2_pretrained.predict(np.expand_dims(img, axis=0))[0]
        
    #     mstr = m.image_path().split("/")
    #     mstr1 = mstr[1]
        
        
    #     if lstr1 != mstr1:
    #         lstr1=mstr1
    #         grp1 = f1.create_group(lstr1)
    #         j=0
    #         grp1.create_dataset(str(j), data=embedded[i])
    #         j=j+1
            
    #     else:
    #         grp1.create_dataset(str(j), data=embedded[i])
    #         lstr1=mstr1
    #         j=j+1

    # f1.close()
        

    jj=0
    mic1={}

    f = h5py.File('tempFiles/embedded2.hdf5','w')
    grp = f
    lstr=""
    data = {}  
    data['people'] = []

    for i, m in enumerate(metadata):
        im = face_recognition.load_image_file(m.image_path())
        try:
            embedded2[i] = face_recognition.face_encodings(im)[0]
            mstr1 = m.name
            if lstr != mstr1:
                lstr=mstr1
                grp = f.create_group(lstr)
                jj=0
                grp.create_dataset(m.file, data=embedded2[i])
                jj=jj+1
            else :
                
                grp.create_dataset(m.file, data=embedded2[i])
                lstr=mstr1
                jj=jj+1
        except Exception as e:
            os.remove(m.image_path())
            print("deleted :"+m.image_path())        
        
        
    f.close()


    np.save("tempFiles/embedding2.npy",embedded2)
    main_c = db.cursor()
    main_c.execute("truncate table user_detail")
    now = datetime.datetime.now()
    dt =now.strftime("%Y-%m-%d")
    path='images'
    for i in os.listdir(path):
        n_img = len(os.listdir(os.path.join(path, i)))        
        main_c.execute("insert into user_detail(name,no_img,last_update_date) values(%s,%s,%s)",(i,n_img,dt))
    
    db.commit()
    main_c.close()
 
