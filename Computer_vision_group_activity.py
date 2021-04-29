#Computer Vision Group Assignment
#Computer aided Melanoma skin cancer detection using Image Processing

import matplotlib.pyplot as plt
import cv2
import os 
import matplotlib.image as mpimg
import numpy as np
from random import shuffle
from tqdm import tqdm 
from google.colab.patches import cv2_imshow
from google.colab.patches import cv2_imshow
from skimage.io import imread
from skimage.transform import resize
from skimage.feature import hog
from skimage import exposure
from google.colab import drive

TRAIN_DIR = '/content/train_images'
TEST_DIR = '/content/test_images'
IMG_SIZE = 50
LR = 1e-3
MODEL_NAME = 'healthyvsunhealthy-{}-{}.model'.format(LR, '2conv-basic')

def label_img(img):
    word_label = img[0]
  
    if word_label == 'h': return [1,0,0,0]
    
    elif word_label == 'b': return [0,1,0,0]
    elif word_label == 'v': return [0,0,1,0]
    elif word_label == 'l': return [0,0,0,1]

def create_train_data():
    training_data = []
    for img in tqdm(os.listdir(TRAIN_DIR)):
        label = label_img(img)
        path = os.path.join(TRAIN_DIR,img)
        img = cv2.imread(path,cv2.IMREAD_COLOR)
        try:
           img = cv2.resize(img, (1400, 1000), interpolation=cv2.INTER_AREA)
           print(img.shape)
        except:
                break
        height, width , layers = img.shape
        size=(width,height)
        print(size)
    #img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
    training_data.append([np.array(img),np.array(label)])
    shuffle(training_data)
    np.save('train_data.npy', training_data)
    return training_data

def process_test_data():
    testing_data = []
    for img in tqdm(os.listdir(TEST_DIR)):
        path = os.path.join(TEST_DIR,img)
        img_num = img.split('.')[0]
        img = cv2.imread(path,cv2.IMREAD_COLOR)
        try:
           img = cv2.resize(img, (1400, 1000), interpolation=cv2.INTER_AREA)
           print(img.shape)
        except:
                break
        height, width , layers = img.shape
        size=(width,height)
        print(size)
        #img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
        testing_data.append([np.array(img), img_num])
    shuffle(testing_data)
    np.save('test_data.npy', testing_data)
    return testing_data

train_data = create_train_data()
# If you have already created the dataset:
#train_data = np.load('train_data.npy')

#image enhancement
img = cv2.imread('/content/train_data.npy',0)
img = np.array(img)
img = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2GRAY)
#######histogram equalization#######
equ = cv2.equalizeHist(np.array(img))
res = np.hstack((img,equ)) #stacking images side-by-side
cv2.imwrite('/content/sample','png',res)
cv2_imshow(res)
hist,bins = np.histogram(img.flatten(),256,[0,256])

#####ploting########
cdf = hist.cumsum()
cdf_normalized = cdf * hist.max()/ cdf.max()

plt.plot(cdf_normalized, color = 'b')
plt.hist(img.flatten(),256,[0,256], color = 'r')
plt.xlim([0,256])
plt.legend(('cdf','histogram'), loc = 'upper left')
plt.show()

######After equalization, to remove artifacts in tile borders, bilinear interpolation is applied. using CLAHE######
# create a CLAHE object (Arguments are optional).
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
cl1 = clahe.apply(res)

cv2.imwrite('/content/sample','jpeg',cl1)
cv2_imshow(cl1)

#######image segmentation##############

#canny
img_canny = cv2.Canny(cl1,100,200)

#sobel
img_sobelx = cv2.Sobel(cl1,cv2.CV_8U,1,0,ksize=5)
img_sobely = cv2.Sobel(cl1,cv2.CV_8U,0,1,ksize=5)
img_sobel = img_sobelx + img_sobely

#prewitt
kernelx = np.array([[1,1,1],[0,0,0],[-1,-1,-1]])
kernely = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
img_prewittx = cv2.filter2D(cl1, -1, kernelx)
img_prewitty = cv2.filter2D(cl1, -1, kernely)

#"Original Image"
cv2_imshow(cl1)
#"Canny", 
cv2_imshow(img_canny)
#"Sobel X", 
cv2_imshow(img_sobelx)
#"Sobel Y", 
cv2_imshow(img_sobely)
#"Sobel", 
cv2_imshow(img_sobel)


#"Prewitt X", 
cv2_imshow(img_prewittx)
#"Prewitt Y", 
cv2_imshow(img_prewitty)
#"Prewitt" 
prewitt=img_prewittx + img_prewitty
cv2_imshow(prewitt)

########colorful image(RGB) to HSV###########   
#prewitt is best in this case hence taking prewitt image for rgb to hsv
backtorgb = cv2.cvtColor(prewitt,cv2.COLOR_GRAY2RGB)

hsvImage = cv2.cvtColor(backtorgb, cv2.COLOR_BGR2HSV)
cv2_imshow(hsvImage)   
#cv2.waitKey(0)
#cv2.destroyAllWindows()

#passing hsv image to hog

#creating hog features
fd, hog_image = hog(hsvImage, orientations=9, pixels_per_cell=(8, 8),
                  cells_per_block=(2, 2), visualize=True, multichannel=True)
plt.axis("off")
plt.imshow(hog_image, cmap="gray")

img = cv2.imread('/content/train_data.npy')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray,50,150,apertureSize = 3)

lines = cv2.HoughLines(edges,1,np.pi/180,200)
for rho,theta in lines[0]:
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a*rho
    y0 = b*rho
    x1 = int(x0 + 1000*(-b))
    y1 = int(y0 + 1000*(a))
    x2 = int(x0 - 1000*(-b))
    y2 = int(y0 - 1000*(a))

    cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2)

cv2.imwrite('/content/sample','jpg',img)

pip install tflearn

import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
import tensorflow as tf
from tensorflow.python.framework import ops
ops.reset_default_graph()

convnet = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 3], name='input')

convnet = conv_2d(convnet, 32, 3, activation='relu')
convnet = max_pool_2d(convnet, 3)

convnet = conv_2d(convnet, 64, 3, activation='relu')
convnet = max_pool_2d(convnet, 3)

convnet = conv_2d(convnet, 128, 3, activation='relu')
convnet = max_pool_2d(convnet, 3)

convnet = conv_2d(convnet, 32, 3, activation='relu')
convnet = max_pool_2d(convnet, 3)

convnet = conv_2d(convnet, 64, 3, activation='relu')
convnet = max_pool_2d(convnet, 3)

convnet = fully_connected(convnet, 1024, activation='relu')
convnet = dropout(convnet, 0.8)

convnet = fully_connected(convnet, 4, activation='softmax')
convnet = regression(convnet, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')

model = tflearn.DNN(convnet, tensorboard_dir='log')

if os.path.exists('{}.meta'.format(MODEL_NAME)):
    model.load(MODEL_NAME)
    print('model loaded!')

train = train_data[:-500]
test = train_data[-500:]

X = np.array([i[0] for i in train]).reshape(-1,IMG_SIZE,IMG_SIZE,3)
Y = [i[1] for i in train]

test_x = np.array([i[0] for i in test]).reshape(-1,IMG_SIZE,IMG_SIZE,3)
test_y = [i[1] for i in test]

model.fit({'input': X}, {'targets': Y}, n_epoch=8, validation_set=({'input': test_x}, {'targets': test_y}),
    snapshot_step=40, show_metric=True, run_id=MODEL_NAME)

model.save(MODEL_NAME)
