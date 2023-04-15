import tensorflow as tf
import numpy as np
import pandas as pd
from tqdm import tqdm


# The function below opens an image and return an image object
def load_image_from_disk(image_id):
    image_path = '/Users/tannazmnjm/Downloads/archive/images/' + str(image_id) + '.jpg'
    return tf.keras.preprocessing.image.load_img(image_path, target_size=(299, 299))


# I need to get the pillow image object and return the numpy representation of it
def pillow_image_to_numpy(pillow_image):
    return np.asarray(pillow_image)


# get a numpy array representation of the image and returns Preprocessed
# numpy.array or a tf.Tensor with type float32 which can be passed to

def numpy_image_nn_preprocessing(numpy_image_array):
    return tf.keras.applications.inception_v3.preprocess_input(numpy_image_array)


# given a list/np array of the random ids, it wil give me the vector
# representation of those photos (each row is a photo of size 1 x 2048)
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


def batch_image_encoding(ids: list[int], batch_size: int = 32) ->\
        (np.ndarray, list[int]):
    vectors = np.array([])
    buffer, valid_ids = [], []
    for i in tqdm(range(len(ids)), desc='encoding images'):
        try:
            loaded_image = load_image_from_disk(ids[i])
        except FileNotFoundError:
            print(f"Error: Could not find image file {ids[i]} - Skipping .")
            continue
        numpied_image = pillow_image_to_numpy(loaded_image)

        if len(numpied_image.shape) != 3:
            continue

        preprocessed_image = numpy_image_nn_preprocessing(numpied_image)
        valid_ids.append(ids[i])
        if len(buffer) == batch_size:
            embeddings = model.predict(np.asarray(buffer))
            if len(vectors) == 0:
                # First batch
                vectors = embeddings
            else:
                vectors = np.vstack((vectors, embeddings))
            # vectors.append(model.predict(np.asarray(buffer)))
            buffer = []
        else:
            buffer.append(preprocessed_image)

    if buffer:
        buffer = np.asarray(buffer)
        vectors = np.vstack((vectors, model.predict(buffer)))

    return vectors, valid_ids


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

    print("Loading styles.csv")
    df = pd.read_csv('/Users/tannazmnjm/Downloads/archive/styles.csv', sep=',',
                     on_bad_lines='skip')

    df_random = df.sample(n=1100)
    random_ids = df_random.id.values
    np.save("random_ids.npy", random_ids)

    df_random = df.loc[df['id'].isin(random_ids)]

    vectorised_matrix, processed_ids = batch_photo_vectorization(random_ids)

    raw_norms = np.linalg.norm(vectorised_matrix, axis=1, ord=2)
    normalised_matrix = vectorised_matrix / raw_norms[:, np.newaxis]


    np.save('Vectorised Matrix.npy', normalised_matrix)


if __name__ == '__main__':
    main()

