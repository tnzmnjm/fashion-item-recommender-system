import subprocess
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from joblib import dump


def substitution_product_recommendation(product_id):
    print(f"Searching for product id: {product_id} in random_ids")
    index_in_random_id = np.where(random_ids == product_id)[0][0]
    index_in_random_vectorised_matrix = index_in_random_id
    distance, nbr_indices_in_vectorised_matrix = \
        clf.kneighbors(vectorised_matrix[index_in_random_vectorised_matrix].reshape(1, -1))
    nbrs_product_ids = []
    for i in range(len(nbr_indices_in_vectorised_matrix[0])):
        nbrs_product_ids.append(random_ids[nbr_indices_in_vectorised_matrix[0][i]])
    selected_df = df_random.loc[df_random.id.isin(nbrs_product_ids)]

    return selected_df


random_ids = np.load('ids_full.npy')

vectorised_matrix = np.load('features_fullx2048.npy')


df = pd.read_csv('/Users/tannazmnjm/Downloads/archive/styles.csv',
                 sep=',',
                 on_bad_lines='skip')

df_random = df.loc[df['id'].isin(random_ids)]

# new_photo_dict = {'id': [101010],
#                   'gender': 'Woman',
#                   'masterCategory': 'Footwear',
#                   'subCategory': 'Shoes',
#                   'articleType': 'Flats',
#                   'baseColour': 'Red',
#                   'season': 'Summer',
#                   'year': [2022],
#                   'usage': 'Casual',
#                   'productDisplayName': 'Tannaz Woman Red Sandal'}

# new_photo_df = pd.DataFrame(new_photo_dict)
# concatenate_df = pd.concat([df_random, new_photo_df])
# # np.save('concatenate_df', concatenate_df)
# concatenate_df.to_csv("tannaz_random_df.csv", index=False)

clf = NearestNeighbors(n_neighbors=5,
                       algorithm='auto').fit(vectorised_matrix)

# vectorised_photo = np.load('vectorised_photo_101010.npy')
# distance, indices = clf.kneighbors(vectorised_photo.reshape(1, -1))

# np.save("tannaz_matrix.npy", np.vstack((vectorised_matrix, vectorised_photo)))
dump(clf, 'knn_model.joblib')
# print(substitution_product_recommendation(3954))
# distance, indices = clf.kneighbors(vectorised_matrix[30].reshape(1, -1))


# image_list = ['open']
# for i in range(len(indices[0])):
    # print(f'random_id[indices[0][{i}] is : {random_id[indices[0][i]]}')
    # print(df_random.loc[df_random.id == random_id[indices[0][i]]].to_dict())
#     image_list.append(f'/Users/tannazmnjm/Downloads/archive/images/{random_id[indices[0][i]]}.jpg')
#
# subprocess.run(image_list)





