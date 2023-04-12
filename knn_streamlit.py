from os.path import join

import streamlit as st
from PIL import Image
from joblib import load
import numpy as np
import pandas as pd

import knn_neural_network


def substitution_product_recommendation(product_id):
    print(f"Searching for product id: {product_id} in random_ids")
    index_in_random_id = np.where(random_ids == product_id)[0][0]
    index_in_random_vectorised_matrix = index_in_random_id
    distance, nbr_indices_in_vectorised_matrix = \
        clf.kneighbors(vectorised_matrix[index_in_random_vectorised_matrix]
                       .reshape(1, -1))
    nbrs_product_ids = []
    for i in range(len(nbr_indices_in_vectorised_matrix[0])):
        nbrs_product_ids.append\
            (random_ids[nbr_indices_in_vectorised_matrix[0][i]])
    selected_df = df_random.loc[df_random.id.isin(nbrs_product_ids)]

    return selected_df


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
st.text(f'Select from available products: {len(df_random)}')


selected_product = st.selectbox(label='Please choose/search for a product :',
                                options=df_random.productDisplayName.values)
selected_id = \
    df_random[df_random.productDisplayName == selected_product]["id"].values[0]


if selected_product is not None:

    if selected_id > 0:
        st.write(f'Selected Product ID: {selected_id}')

        selected_id_image_path = join(image_dir, f'{selected_id}.jpg')
        st.image(Image.open(selected_id_image_path))
        st.dataframe(df_random[df_random.id == selected_id])

        substitution_df = substitution_product_recommendation(selected_id)
        # dropping the selected id from the result
        substitution_df = substitution_df[substitution_df["id"] != selected_id]

        # recommended product will have the same gender unless it's unisex which
        # will be everything

        st.subheader('Substitutable Products')
        st.subheader('Please choose your required filters')

        use_gender = st.checkbox('Gender')
        use_subCategory = st.checkbox('Sub Category')
        use_articleType = st.checkbox('Article Type')
        use_baseColour = st.checkbox('Colour')

        if use_gender:
            selected_id_gender = \
                df_random.loc[df_random.id == selected_id, "gender"].values[0]
            if selected_id_gender != 'Unisex':
                substitution_df = \
                    substitution_df[substitution_df.gender == selected_id_gender]

        if use_subCategory:
            subCategory = \
                df_random.loc[df_random.id == selected_id, "subCategory"].values[0]
            substitution_df = substitution_df[substitution_df.subCategory == subCategory]

        if use_articleType:
            articleType = \
                df_random.loc[df_random.id == selected_id , "articleType"].values[0]
            substitution_df = substitution_df[substitution_df.articleType == articleType]

        if use_baseColour:
            colour = df_random.loc[df_random.id == selected_id, "baseColour"].values[0]
            substitution_df = substitution_df.loc[substitution_df.baseColour == colour]

        st.dataframe(substitution_df)

        filenames = [join(image_dir, f"{id}.jpg") for id in substitution_df.id.values]
        images = [Image.open(img_file) for img_file in filenames]
        st.image(images)

    else:
        st.write('Please enter a valid product ID')
    #
