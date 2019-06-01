from PIL import Image
import requests
from io import BytesIO
import numpy as np
response = requests.get('http://192.168.43.1:8080/photo.jpg')
img = Image.open(BytesIO(response.content))
img2 = np.array(img)
import cv2
cv2.imwrite('aaa.jpg',img2)