import numpy as np
from keras.layers import GlobalAveragePooling2D
from keras.layers import Dense
from keras.models import Sequential
from keras.applications.resnet50 import ResNet50
from PIL import Image
import cv2  
from keras.preprocessing import image
from tqdm import tqdm
from extract_bottleneck_features import extract_Xception
from keras.applications.resnet50 import preprocess_input
from dog_names import dog_names
Xception_model = Sequential()
Xception_model.add(GlobalAveragePooling2D(input_shape=(7, 7, 2048)))
Xception_model.add(Dense(133, activation='softmax'))
Xception_model.load_weights('saved_models/weights.best.Xception.hdf5')
face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt.xml')
ResNet50_model = ResNet50(weights='imagenet')


def path_to_tensor(img_path):
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=(224, 224))
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)


def ResNet50_predict_labels(img_path):
    # returns prediction vector for image located at img_path
    img = preprocess_input(path_to_tensor(img_path))
    return np.argmax(ResNet50_model.predict(img))


def face_detector(img_path):
    # return true if face is detected on the image
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    return len(faces) > 0


def dog_detector(img_path):
    # return true if dog is detected on the image
    prediction = ResNet50_predict_labels(img_path)
    return ((prediction <= 268) & (prediction >= 151)) 


def Xception_dog_breed_prediction(image_path):
    # extract bottleneck features
    bottleneck_feature = extract_Xception(path_to_tensor(image_path))
    # obtain predicted vector
    prediction_value = Xception_model.predict(bottleneck_feature)
    # return dog breed name from the list
    return dog_names[np.argmax(prediction_value)]


def dog_or_human_detector(image_path):
    if dog_detector(image_path):
        dog_breed = Xception_dog_breed_prediction(image_path)
        message = '{}{}'.format('A dog was detected on the picture, its predicted breed is ', dog_breed)
    elif face_detector(image_path):
        breed = Xception_dog_breed_prediction(image_path)
        message = '{} {} {}'.format('A human was detected on the picture, this human looks close to ', breed, ' breed')
    else:
        message = 'Did not detect a dog or a human'
    return message