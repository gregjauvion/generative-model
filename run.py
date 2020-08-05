
import pickle
import numpy as np
import tensorflow as tf
import PIL.Image
import os


# The models used here come from Karras 2018 ICLR (https://github.com/tkarras/progressive_growing_of_gans)
# https://drive.google.com/drive/folders/15hvzxt_XxuokSmj0uO4xxMTMWVc0cIMU
TYPE_MODEL = {
    'face': 'karras2018iclr-celebahq-1024x1024.pkl',
    'living_room': 'karras2018iclr-lsun-livingroom-256x256.pkl'
}


def load_model(type_):

    with open(f'models/{TYPE_MODEL[type_]}', 'rb') as f:
        _, _, model = pickle.load(file)

    return model


def generate(model, number, path):
    """
    Generate {number} images with {model}.
    Save in {path} a dict with:
    - vectors: The latent vectors
    - images: The images as arrays
    """

    # Generate random latent vectors
    vectors = np.random.RandomState().randn(number, model.input_shapes[0][1:])

    # Generate the images
    images = model.run(vectors, _)

    # Convert images to PIL-compatible format.
    images = np.clip(np.rint((images + 1.0) / 2.0 * 255.0), 0.0, 255.0).astype(np.uint8) # [-1,1] => [0,255]
    images = images.transpose(0, 2, 3, 1) # NCHW => NHWC

    # Dump
    result = {'vectors': vectors, 'images': images}
    os.makedirs(path, exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(result, f)

    return


def write_pngs(path_in, path_out):
    """
    Read pickle written by generate() and write the pngs in the directory {path_out}
    """

    with open(path_in, 'rb') as f:
        dump = pickle.load(f)

    os.makedirs(path_out, exist_ok=True)

    # Write all pngs
    images = dump['images']
    for i in range(images.shape[0]):
        PIL.Image.fromarray(images[i], 'RGB').save(f'{path_out}/{i}.png')


if __name__=='__main__':

    # Set TF session
    tf.InteractiveSession()

    # Generate some images
    model = load_model('face')
    generate(model, 20, 'outputs/face/images.pkl')

    # Save as png
    write_pngs('outputs/face/images.pkl', 'outputs/face')
