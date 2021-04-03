# Dog Breed Project
### Table of Contents
1. [Project Overview](#overview)
2. [Project Motivation](#motivation)
3. [Dataset](#dataset)
4. [Metrics](#metrics)
5. [Methodology and Implementation](#method)
6. [Model Evaluation and Results](#results)
7. [Conclusion](#conclusion)
8. [File Structure](#file)
9. [Libraries and Dependencies](#libraries)
10. [Instruction for running application](#instructions)
11. [Screenshots](#screenshots)

## Project Overview<a name="overview"></a>
Dog Breed Prediction project is a final project of the Data Science NanoDegree program. The primary objective of this project is to build an image classification pipeline. Once the image is provided, the pipeline will preprocess the image and feed into two different deep learning models to generate the final output. The first model is a dog detector model to detect a dog in an image. Given a dog is detected on the image, the second model will identify an estimate of the dog's breed. Finally, I will deploy this model to the flask web app to process real-world, user-supplied images to identify dogs breed.

## Project Motivation<a name="motivation"></a>
I was fascinated by this project because of the recent hype in computer vision. Whether we know it or not, many things that we do every day have some sort of computer vision algorithm embedded in them. For instance unlocking our phones using faces, shopping at cashier-less stores, virtual reality, self-driving cars and so on. Here, I will first implement an algorithm that will identify whether images has dogs, humans, or neither. If a dog is identified in the image, we will predict the dog's breed. For the sake of fun, if we identify a human, we will predict what dog breed that human most resemble. 

## Dataset<a name="dataset"></a>
The dataset used in this project was provided by Udacity. It contains two different image dataset:
1. Dog Images: There are 8351 dog images from 133 different breeds. We will split the dataset into a training set, a validation set and a test set. 
2. Human Images: 13233 human images. This dataset is used on a human face detector algorithm. 

## Metrics<a name="metrics"></a>
For this project, I will be using accuracy metrics to measure the performance of the model. Our dataset is somewhat fairly distributed among 133 categories, therefore, the number of correct predictions that model makes is sufficient. If there was a significant class imbalance, other metrics such as precision and recall might be better choices.

## Methodology and Implementation<a name="method"></a>
Following data science/computer vision methodologies were used in this project:

1. Loading and preprocessing Data - In the first part, implement a function that loads the dataset from the folder, and in the second part, using TensorFlow as backend, convert the dataset in 4D array ie (number_of_samples, rows, columns, channels), that will be fed into the CNN model.
2. Human detector - Using Haar-feature-based cascade-classifers, we will detect human faces in images. 
3. Dog detector - For dog detection,  we will use the ResNet-50 model.
4. CNN from scratch - Build a CNNs model from scratch, train it and evaluate the model accuracy. A typical CNNs architecture stack with few convolutional layers along with relu activation layer and pooling layer technique is implemented.
5. CNN using Transfer Learning - In this section, we will use a pre-trained network as a fixed feature extractor, where the last convolutional output of the a pre-trained network is fed as an input to our CNN model. Finally, techniques such global average pooling layer and a fully connected layer equipped with softmax activation are used to identify a dog's breed. Because of the time constrain, I will be using the Resent50 bottleneck feature to create a CNN model.


## Model Evaluation and Results<a name="results"></a>
The dog and human detector models were working as expected. Detector models took the image path and determine whether the image has a dog, human, or neither. The Next step was to test both of our CNN models. 

The CNNs model that we built from the scratch was able to generate an accuracy slightly above 5%. The goal was to get above 1% accuracy. Given we had 133 class of dog breeds, a random guess will produce an accuracy of less than 1%. It was not a bad start. Since our dataset was fairly small, the model was not able to generalize. In order to improve the accuracy of this model, we could use data augmentation techniques to increase the size of the dataset, which could lead to better accuracy. 

Now, it was time to test a CNNs model using transfer learning. We used transfer learning to create a CNN using the Resent50 model, pre-trained on ImageNet. To this model, we added a global average pooling layer where the input shape was based on the output of the base model. Then, a dense output layer with 133 multiclass output using softmax activation was added to the model. We complied the model and start training. This was our model with 272k trainable parameters. After training the model for just 20 epochs, our validation accuracy was 81.5% percent. 


## Conclusion<a name="conclusion"></a>
CNNs model that we built from scratch gave us slightly over 5% accuracy, which was better than the random guess which would give us an accuracy of less than 1% given we have 133 categories of dog breeds. But the model that we built using ResNet50, a pre-trained model on ImageNet, we were able to get over 80% accuracy. I think the output was much better than what I had expected given we had slightly over 5% accuracy on the CNN model that we built from scratch. This is almost a "free lunch". We added two layers to the pre-trained network and able to get over 81% accuracy is mind-blowing. However, there is still plenty of room to improve our algorithm.

* There is evidence of overfitting in our model. We can collect more images and also use data augmentation techniques such as image rotation, blur, rescale and flip on the current dataset to increase the size of our training data. 
* Train our model longer/ higher epochs to improve our validation accuracy. 
* We could also more layers to the model.
* Currently the model is only able to predict if there is only one dog on the image. We might also consider adding classification and localization to able to classify multiple dog breeds if there were multiple dogs in a single image. Popular architecture such as YOLO (You Only Look Once) can be used for fast and accurate object detection.

## Repository Structure<a name="file"></a>
   Repository contains two section
```
- dog-project
| - haarcascades
| - saved_models
| - test_images
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
## Screenshots <a name="screenshots"></a>
![screen1](https://user-images.githubusercontent.com/7229266/113328333-0420a300-92d1-11eb-9d5f-160aea8ebc88.PNG)

![screen2](https://user-images.githubusercontent.com/7229266/113328445-25818f00-92d1-11eb-8df3-b84a5a1cbd4d.PNG)
