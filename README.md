# Fashion Item Recommender System Using Neural Networks and K-Nearest Neighbors

# Summary
This repository contains a fashion item recommender system, that suggests similar fashion items based on the user's choice of either a random product or a taken photo.

The dataset used in this project is the Fashion Product Images Dataset available on Kaggle. The pre-trained neural network used for feature extraction is the Inception-V3 model trained on the ImageNet dataset. The KNN model is trained using the feature vectors extracted from the images.


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
The system uses a pre-trained neural network to extract feature vectors from the images of fashion items. These feature vectors are then used to train a K-Nearest Neighbors (KNN) model.

When the user selects a product or takes a photo, the feature vector for the selected product or the taken photo is extracted using the pre-trained neural network. The KNN model is then used to find the five most similar items to the selected product or the taken photo.


The NearestNeighbors class from scikit-learn is now trained on the vectorised_matrix array, with n_neighbors=5 and algorithm='auto'. This means that for any query point, the model will find the 5 nearest neighbors based on Euclidean distance.

# Example
After running the streamlit file, there is an option for either choosing a product at random or taking a photo :

![image](https://user-images.githubusercontent.com/22201551/232073396-6fd55bde-9169-4704-9c34-599af43f48ed.png)

Choosing to see a product at random, an image of that produnct along with the metadata will be displayed:

![image](https://user-images.githubusercontent.com/22201551/232072089-48aaaf57-97fc-4c8a-a951-83f49e507662.png)

Please note in order to do the identity check, I have left the image of the randomly selected item to be present but have removed it from the final result and instead the metadata of the second closest item is returned.
