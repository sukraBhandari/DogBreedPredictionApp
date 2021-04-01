# Dog Breed Project
### Table of Contents
1. [Project Motivation](#motivation)
2. [Dataset](#dataset)
3. [File Structure](#file)
4. [Libraries and Dependencies](#libraries)
5. [Instruction for running application](#instructions)
6. [Results](#results)
7. [Screenshots](#screenshots)

## Project Motivation<a name="motivation"></a>

Dog Breed Prediction project is a final project of the Data Science NanoDegree program. The primary objective of this project is to use Convolutional Neural Networks (CNNs) tools to identify dog breed from images. I was fasinated by this project because of the recent hype in computer vision. Whether we know it or not, many things that we do everday have some sort of computer vision algorithm embeded to them. For instance unlocking our phone using faces, shopping at cashier less stores, virtual reality and so on.Here, I will first implement an algorithm that will identify whether images has human or dogs. If a dog is identified in the image, we will predict the dog's breed. For the sake of fun, if we identify a human, we will predict what dog breed that human most resembles. Finally, I will deployed this model to the flask web app where users can upload images to classify dog's breed. Following data science/computer vision techniques were used in this project:

1. Loading and preprocessing Data - In the first part, implement a function that loads dataset from the folder and in the second part, using TensorFlow as bakcend, convert the dataset in 4D array ie (number_of_samples, rows, columns, channels), that will be fed into the CNN model.
2. Human detector - Using Haar-feature-based-cascase-classifers, we will detect human faces in images. 
3. Dog detector - For dog detection,  we will use ResNet-50 model.
4. CNN from scratch - Build a cnn model from sratch, train it and evaluate the model accuracy. A typical CNN architecture stack with few convolutional layers along with relu activation layer and pooling layer technique is implemented.
5. CNN using Transfer Learning - In this section, we will use pretrained network as a fixed feature extractor, where the last convlutional output of the pretrain network is fed as an input to our CNN model. Finally, techniques such global average pooling layer and a fully connected layer equipped with softmax activation is used to identify dog's breed.
6. Web application - An interactive application that allows user to upload images to identify the dpg's breed.



## Dataset<a name="dataset"></a>

The dataset used in this project was provided by Udacity. It contains two diffrent image dataset:
1. Dog Images : 8351 dog images from 133 different breeds. 
2. Human Images: 13233 human images 
       
## Repository Structure<a name="file"></a>
   Repository contains two section
```
- dog-project
| - haarcascades
| - saved_models
| - dog_app.ipynb
| - extract_bottleneck_features.py

- flask app
| - saved_models
| - static folder 
| - templates
| - app.py
| - dog_names.py
| - extract_bottleneck_features.py
| - predictions.py
| - requirements.txt
| - utils.py

```

## Libraries and Dependencies <a name="libraries"></a>

To deploy the application, following libraries were used in this project.

 - python==3.7 !important
 - keras
 - tensorflow
 - numpy
 - flask
 - opencv
 - werkzeug
 - tqdm
 - Pillow
 - flask_uploads
 - flask_WTF
 - flask_bootstrap
 - gunicorn
 - Jinga

## Instruction for running application<a name="instructions"></a>

In order to run this application in localhost, please execute the following steps. It is recommended to use a virtual environment to run the app. 
   1. Set up a virtual environment using conda for the Anaconda and install required libraries
   ```
   conda create --name dogapp python=3.7
   conda activate dogapp
   pip install -r requirements.txt
   ```
   2. Run the app locally with following code 
   ```
   set FLASK_APP=app.py
   flask run
   ```   
  
   
 ## Results and Future Implementation<a name="results"></a>
CNNs model that we built from scratch gave us sligtly over 5% accuracy, which was better than the random guess which would give us accuracy of less than 1% given we have 133 categories of dog breeds. But the model that we built using ResNet50, a pretrained model on ImageNet, we were able to get over 80% accuracy. I think the ouput was much better than what I had expected given we had slight over 5% accuracy on the CNN model that we built from scratch. This is almost a "free lunch". We added two layers to the pretrained network and able to get over 80% accuracy is mind blowing. However, there is still plenty of room to improve our algorithm.
    - There is an evidence of overfitting in our model. We can collect more dataset and also use data agumentation techiques such as image rotate, blur, rescale and flip on current dataset to increase the size of our training data. 
    -  Train our model longer/ higher epochs to improve our validation accuracy. 
    -  We could also more layer to the model.
    -  Currently the model is only able to predict if there is only one dog on the image. We might also consider adding classification and localization to able to classify multiple dog breed if there were multiple dogs in a single image. Popular architecure such as YOLO (You Only Look Once) can be used for fast and accurate oject detection.
 
## Screenshots <a name="screenshots"></a>
