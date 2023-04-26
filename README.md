# Fashion Item Recommender System Using Neural Networks and K-Nearest Neighbors

# Summary
This repository contains a fashion item recommender system, that suggests similar fashion items based on the user's choice of either a random product or a taken photo.

The [dataset](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small/code) used in this project is the Fashion Product Images Dataset available on Kaggle. The pre-trained neural network used for feature extraction is the [Inception-V3 model](https://keras.io/api/applications/inceptionv3/) trained on the ImageNet dataset. 

# Requirements
The system requires the following dependencies to run:

- Python 3
- Tensorflow
- Streamlit
- Pillow
- Joblib
- Numpy
- Pandas
- Scikit-learn
- Tqdm
- Matplotlib
- Wordcloud

# How it works
The system uses a pre-trained neural network to extract feature vectors from the images of fashion items. These feature vectors are then used to train a K-Nearest Neighbors (KNN) model. They are also used in the Logistic Regression Multinomial Classification in order to predict the master category of the item. 

# Scripts

1 - <b>The script EDA.py</b>:
This script performs Exploratory Data Analysis (EDA). The information below can be drawn from this analysis:
  - the structure of the dataset (size of the dataframe, data types of each column, number of unique values in each column)
  - number of missing values in each column which are then visialised in a bar chart using Plotly
  - the most frequently mentioned words in the 'productDisplayName' column are identified using WordCloud
  - checking for missing images in the image folder that correspond to the 'id's in the dataset
  - finding the images with no metadata
  - determining the black and white images

<img src="https://github.com/tnzmnjm/fashion-item-recommender-system/blob/master/column%20missing%20values.png">
<img src="https://github.com/tnzmnjm/fashion-item-recommender-system/blob/master/wordcloud.png" width="500" height="500"> 


2 - <b>The script NN.py</b>: 
performs batch photo vectorization using the InceptionV3 model from the Keras library. 
  - images are loaded from a specified directory and resized to 299x299 pixels. 
  - these images are then processed using the InceptionV3 preprocessing function and predicts the output vector using the InceptionV3 model.
  - the script also performs error handling for files that cannot be found or have an incorrect number of channels.
  - After vectorizing the images, it normalizes the resulting matrix and saves it to a file.
  - The main function calls other functions in the script and saves the results to files named "random_ids.npy" and "Vectorised Matrix.npy".

3 - <b>The script KNN-main.py</b>:
This code is for creating a product recommendation system using the K-Nearest Neighbors (KNN) algorithm. 
  - The program uses a preprocessed dataset of product images to compute image features using the Inception V3 model. 
  - The NearestNeighbors class from scikit-learn is now trained on the vectorised_matrix array, with n_neighbors=5 and algorithm='auto'. This means that for any query point, the model will find the 5 nearest neighbors based on Euclidean distance.
  - The substitution_product_recommendation function takes a product ID as an input and returns a dataframe of recommended substitute products based on image similarity computed by the KNN model. The program saves the trained KNN model to a file named "knn_model.joblib" using the joblib library.

4 - <b>The script knn_streamlit.py</b>:
  - Streamlit web application is used for demonstration of the result. After running the streamlit file, there is an option for either choosing a product at random or taking a photo
  - based on the user preference, the image vector of that image wil be found/calculated and will be passed to the function substitution_product_recommendation(image_vector)which will then pass the vectors to our knn model and provides the closest neighbours(recommendations).

# Example


![image](https://user-images.githubusercontent.com/22201551/232073396-6fd55bde-9169-4704-9c34-599af43f48ed.png)

Choosing to see a product at random, an image of that produnct along with the metadata will be displayed:

![image](https://user-images.githubusercontent.com/22201551/232072089-48aaaf57-97fc-4c8a-a951-83f49e507662.png)

Please note in order to do the identity check, I have left the image of the randomly selected item to be present.
