from os.path import join

import streamlit as st
import tensorflow as tf
from PIL import Image
from joblib import load
import numpy as np
import pandas as pd

import knn_neural_network


def substitution_product_from_photo(vectorised_photo):
    distance, nbr_indices_in_vectorised_matrix = clf.kneighbors(vectorised_photo)
    nbrs_product_ids = []
    for i in range(len(nbr_indices_in_vectorised_matrix[0])):
        nbrs_product_ids.append(random_ids[nbr_indices_in_vectorised_matrix[0][i]])
    selected_df = df_random.loc[df_random.id.isin(nbrs_product_ids)]
    return selected_df


def add_vector_to_vectorised_matrix(vectorised_matrix, vectorised_image):
    final_vectorised_matrix = np.vstack((vectorised_matrix, vectorised_image))
    return final_vectorised_matrix


def add_new_product_id_to_random_ids(random_ids, product_id):
    random_ids = np.append(random_ids, np.int64(product_id))


df = pd.read_csv('/Users/tannazmnjm/Downloads/archive/styles.csv', sep=',',
                 engine='python', on_bad_lines='skip')
clf = load('knn_model.joblib')
image_dir = '/Users/tannazmnjm/Downloads/archive/images/'
random_ids = np.load('ids_full.npy')
vectorised_matrix = np.load('features_fullx2048.npy')
df_random = df.loc[df['id'].isin(random_ids)]
model = knn_neural_network.get_model()

del df
# UI Section

st.title('Fashion items Dataset')
st.text(f'Select from available products or take a picture: {len(df_random)}')

img_file_buffer = st.camera_input("Take a picture")


if img_file_buffer is not None:
    # need to prepare the image before calling the model:
    image_pillowed = tf.keras.preprocessing.image.load_img(img_file_buffer, target_size=(299, 299))
    imag_numpied = knn_neural_network.pillow_image_to_numpy(image_pillowed)
    image_preprocesed = knn_neural_network.numpy_image_nn_preprocessing(imag_numpied)
    vectorised_image = model.predict(image_preprocesed[tf.newaxis, ...])

    # add this vector to the vectorised_matrix
    add_vector_to_vectorised_matrix(vectorised_matrix, vectorised_image)

    substitution_df = substitution_product_from_photo(vectorised_image)

    filenames = [join(image_dir, f"{id}.jpg") for id in substitution_df.id.values]
    images = [Image.open(img_file) for img_file in filenames]
    st.image(images)
    #
    # use_gender = st.checkbox('Gender')
    # use_subCategory = st.checkbox('Sub Category')
    # use_articleType = st.checkbox('Article Type')
    # use_baseColour = st.checkbox('Colour')
    #
    # if use_gender:
    #     selected_id_gender = df_random.loc[df_random.id == selected_id, "gender"].values[0]
    #     if selected_id_gender != 'Unisex':
    #         substitution_df = substitution_df[substitution_df.gender == selected_id_gender]
    #
    # if use_subCategory:
    #     subCategory = df_random.loc[df_random.id == selected_id, "subCategory"].values[0]
    #     substitution_df = substitution_df[substitution_df.subCategory == subCategory]
    #
    # if use_articleType:
    #     articleType = df_random.loc[df_random.id == selected_id, "articleType"].values[0]
    #     substitution_df = substitution_df[substitution_df.articleType == articleType]
    #
    # if use_baseColour:
    #     colour = df_random.loc[df_random.id == selected_id, "baseColour"].values[0]
    #     substitution_df = substitution_df.loc[substitution_df.baseColour == colour]

    st.dataframe(substitution_df)

else:
    st.write('Please enter a valid product ID')
