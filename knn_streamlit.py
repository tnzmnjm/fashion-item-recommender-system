from os.path import join

import tensorflow as tf
import streamlit as st
from PIL import Image
from joblib import load
import numpy as np
import pandas as pd

import knn_neural_network


def substitution_product_recommendation(image_vector):
    distance, nbr_indices = clf.kneighbors(image_vector.reshape(1, -1))

    nbrs_product_ids = []
    for i in range(len(nbr_indices[0])):
        nbrs_product_ids.append(random_ids[nbr_indices[0][i]])
    neighbours_df = df_random.loc[df_random.id.isin(nbrs_product_ids)]

    st.subheader('Please choose your required filters')




    st.dataframe(neighbours_df)
    filenames = [join(image_dir, f"{id}.jpg") for id in neighbours_df.id.values]
    images = [Image.open(img_file) for img_file in filenames]
    st.image(images)

    return neighbours_df


df = pd.read_csv('/Users/tannazmnjm/Downloads/archive/styles.csv',
                 sep=',',
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
st.text(f'Total number of products: {len(df_random)}')

st.write('Please choose to either see a product at random or take a '
         'photo with your camera :')

select_random_product = st.button('Random product  🎲')
if select_random_product:
    selected_product_id = np.random.choice(random_ids)
    # selected_product = 52417
    selected_id_image_path = join(image_dir, f'{selected_product_id}.jpg')
    st.dataframe(df_random[df_random.id == selected_product_id])
    st.image(Image.open(selected_id_image_path))

    substitution_product_recommendation(
        vectorised_matrix[np.where(random_ids == selected_product_id)])
# # # TODO: need to remove the item itself from the results


input_camera_photo = st.camera_input("Take a picture")
if input_camera_photo:
    image_pillowed = \
        tf.keras.preprocessing.image.load_img(
            input_camera_photo, target_size=(299, 299))
    imag_numpied = knn_neural_network.pillow_image_to_numpy(image_pillowed)
    image_preprocesed = knn_neural_network.numpy_image_nn_preprocessing(imag_numpied)
    vectorised_image = model.predict(image_preprocesed[tf.newaxis, ...])

    substitution_product_recommendation(vectorised_image)



 # use_gender = st.checkbox('Gender')
    # use_subcategory = st.checkbox('Sub Category')
    # use_articletype = st.checkbox('Article Type')
    # use_basecolour = st.checkbox('Colour')

    # if use_gender:
    #     selected_id_gender = \
    #         df_random.loc[df_random.id == selected_product_id, "gender"].values[0]
    #     if selected_id_gender != 'Unisex':
    #             neighbours_df = \
    #                 neighbours_df[neighbours_df.gender == selected_id_gender]
    #
    # if use_subcategory:
    #     subcategory = \
    #         df_random.loc[df_random.id ==
#         selected_product_id, "subCategory"].values[0]
    #     substitution_df = neighbours_df[neighbours_df.subCategory == subcategory]
    #
    # if use_articletype:
    #     articletype = \
    #         df_random.loc[df_random.id ==
#         selected_product_id , "articleType"].values[0]
    #     substitution_df = neighbours_df[neighbours_df.articleType == articletype]
    #
    # if use_basecolour:
    #     colour = df_random.loc[df_random.id ==
#     selected_product_id, "baseColour"].values[0]
    #     substitution_df = neighbours_df.loc[neighbours_df.baseColour == colour]










