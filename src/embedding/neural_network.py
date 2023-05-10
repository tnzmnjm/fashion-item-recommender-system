import tensorflow as tf
import numpy as np
import pandas as pd
from tqdm import tqdm


# The function below opens an image and return an image object
def load_image_from_disk(image_id):
    image_path = 'data/images' + str(image_id) + '.jpg'
    return tf.keras.preprocessing.image.load_img(image_path, target_size=(299, 299))


# numpy representation of the image:
def pillow_image_to_numpy(pillow_image):
    return np.asarray(pillow_image)


# preprocessing a given input image for use with the InceptionV3 model
def numpy_image_nn_preprocessing(numpy_image_array):
    return tf.keras.applications.inception_v3.preprocess_input(numpy_image_array)


# vectorised representation of the photos (each row is a photo of size 1 x 2048)
def batch_photo_vectorization(list_of_random_ids):
    vectorized_photo_matrix = np.zeros((len(list_of_random_ids), 2048))
    processed_random_ids = []
    for i in tqdm(range(len(list_of_random_ids)), desc='encoding images'):
        try:
            loaded_image = load_image_from_disk(list_of_random_ids[i])
        except FileNotFoundError:
            print(f"Error: Could not find image file {list_of_random_ids[i]}"
                  f" - Skipping .")
            continue
        numpied_image = pillow_image_to_numpy(loaded_image)

        if len(numpied_image.shape) != 3:
            print(f"Error: Image with channels {len(numpied_image.shape)} "
                  f"found - Skipping .")
            continue

        preprocessed_image = numpy_image_nn_preprocessing(numpied_image)
        vectorized_photo = model.predict(preprocessed_image[tf.newaxis, ...])

        vectorized_photo_matrix[i] = vectorized_photo
        processed_random_ids.append(list_of_random_ids[i])

    return vectorized_photo_matrix, processed_random_ids


def get_model():
    return tf.keras.applications.inception_v3.InceptionV3(
        include_top=False,
        weights="imagenet",
        pooling="avg"
    )


def main():

    print("Initialising the model")
    # Now we initialise the model and download its weights
    model = get_model()

    print("Loading the features (styles.csv)")
    df = pd.read_csv('/data/styles.csv', sep=',',
                     on_bad_lines='skip')

    df_random = df.sample(n=1100)
    random_ids = df_random.id.values
    np.save("models/random_ids.npy", random_ids)

    df_random = df.loc[df['id'].isin(random_ids)]

    vectorised_matrix, processed_ids = batch_photo_vectorization(random_ids)

    # np.linalg.norm is used to compute the L2 norm of each row of a matrix -
    # the L2 norm of a row is the square root of the sum of the squares of
    # its elements.
    raw_norms = np.linalg.norm(vectorised_matrix, axis=1, ord=2)
    normalised_matrix = vectorised_matrix / raw_norms[:, np.newaxis]

    np.save('models/vectorised_matrix.npy', normalised_matrix)


if __name__ == '__main__':
    main()

