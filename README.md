# Fashion Item Recommender System Using Neural Networks and K-Nearest Neighbors

# Summary
This repository contains a fashion item recommender system, that suggests similar fashion items based on the user's choice of either a random product or a taken photo.

The dataset used in this project is the Fashion Product Images Dataset available on Kaggle. The pre-trained neural network used for feature extraction is the Inception-V3 model trained on the ImageNet dataset. 

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
The system uses a pre-trained neural network to extract feature vectors from the images of fashion items. These feature vectors are then used to train a K-Nearest Neighbors (KNN) model. They are also used in the logistic regression multinomial classification in order to predict the master category of the item. 

# Scripts

- EDA.py
This script performs Exploratory Data Analysis (EDA) on a dataset, which is loaded as a dataframe (df). The dataset is related to apparel products and consists of 10 columns (id, gender, masterCategory, subCategory, articleType, baseColour, season, year, usage, productDisplayName).
The first part of the code examines the structure of the dataset, including the size of the dataframe and the data types of each column. The number of unique values in each column is also displayed, along with the number of missing values. The missing values are visualized in a bar chart using Plotly.
Next, the most frequently mentioned words in the productDisplayName column are identified using WordCloud. The resulting image is saved to a file.
The code then checks for missing images in the image folder that correspond to the ids in the dataset. Similarly, it identifies the images with no corresponding metadata. The photos with no metadata are removed from the vectorized matrix.
Finally, the code determines which images in the dataset are black and white and saves the id of the black and white photo in a variable.

- EDA.py
This script performs Exploratory Data Analysis (EDA). The information below can be drawn from this analysis:
  - the structure of the dataset (size of the dataframe, data types of each column, number of unique values in each column)
  - Number of missing values in each column which are then visialised in a bar chart using Plotly
  - the most frequently mentioned words in the productDisplayName column are identified using WordCloud
  - checking for missing images in the image folder that correspond to the ids in the dataset
  - finding the images with no metadata
  - determining the black and white images

<img src="https://github.com/tnzmnjm/fashion-item-recommender-system/blob/master/column%20missing%20values.png">



![image](https://user-images.githubusercontent.com/22201551/234316792-5325c63e-5366-43db-90bb-46f2d82b9382.png)


![image](https://user-images.githubusercontent.com/22201551/234317254-5cc0d86f-c951-423c-91d2-eb0c73569af7.png)


- The script NN.py: performs batch photo vectorization using the InceptionV3 model from the Keras library. The script loads images from a specified directory and resizes them to 299x299 pixels. Then, it preprocesses the images using the InceptionV3 preprocessing function and predicts the output vector using the InceptionV3 model.
The script also performs error handling for files that cannot be found or have an incorrect number of channels.
After vectorizing the images, it normalizes the resulting matrix and saves it to a file.
The main function calls other functions in the script and saves the results to files named "random_ids.npy" and "Vectorised Matrix.npy".

- The KNN-main.py:
This code is for creating a product recommendation system using the K-Nearest Neighbors (KNN) algorithm. The program uses a preprocessed dataset of product images to compute image features using the Inception V3 model. 
The NearestNeighbors class from scikit-learn is now trained on the vectorised_matrix array, with n_neighbors=5 and algorithm='auto'. This means that for any query point, the model will find the 5 nearest neighbors based on Euclidean distance.The substitution_product_recommendation function takes a product ID as an input and returns a dataframe of recommended substitute products based on image similarity computed by the KNN model. The program saves the trained KNN model to a file named "knn_model.joblib" using the joblib library.


# Example
Streamlit web application is used for demonstration of the result. After running the streamlit file, there is an option for either choosing a product at random or taking a photo :

![image](https://user-images.githubusercontent.com/22201551/232073396-6fd55bde-9169-4704-9c34-599af43f48ed.png)

Choosing to see a product at random, an image of that produnct along with the metadata will be displayed:

![image](https://user-images.githubusercontent.com/22201551/232072089-48aaaf57-97fc-4c8a-a951-83f49e507662.png)

Please note in order to do the identity check, I have left the image of the randomly selected item to be present.
