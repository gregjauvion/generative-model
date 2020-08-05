
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

# Images are generated per batch to avoid OOM errors
BATCH_SIZE = 10



def load_model(type_):

    with open(f'models/{TYPE_MODEL[type_]}', 'rb') as f:
        _, discriminator, generator = pickle.load(f)

    return generator, discriminator


def generate(generator, discriminator, number, path_out, vectors=None):
    """
    Generate {number} images with {model}.
    Save in {path} a dict with:
    - vectors: The latent vectors
    - images: The images as arrays
    """

    # Generate random latent vectors if not specified
    if vectors is None:
        vectors = np.random.rand(number, *generator.input_shapes[0][1:])

    # Generate dummy labels (not used by those networks but they need to be here)
    labels = np.zeros([vectors.shape[0]] + generator.input_shapes[1][1:])

    # Generate the images per batch of BATCH_SIZE
    images = np.concatenate([generator.run(vectors[i*BATCH_SIZE:(i + 1)*BATCH_SIZE], labels[i*BATCH_SIZE:(i + 1)*BATCH_SIZE]) for i in range(number//BATCH_SIZE)])
    probabilities = discriminator.run(images)[0].reshape(-1)

    # Convert images to PIL-compatible format.
    images = np.clip(np.rint((images + 1.0) / 2.0 * 255.0), 0.0, 255.0).astype(np.uint8) # [-1,1] => [0,255]
    images = images.transpose(0, 2, 3, 1) # NCHW => NHWC

    # Dump
    result = {'vectors': vectors, 'images': images, 'probabilities': probabilities}
    os.makedirs(path_out, exist_ok=True)
    with open(f'{path_out}/images.pkl', 'wb') as f:
        pickle.dump(result, f)

    return


def write_pngs(path_in):
    """
    Read pickle written by generate() and write the pngs
    """

    with open(f'{path_in}/images.pkl', 'rb') as f:
        dump = pickle.load(f)

    os.makedirs(f'{path_in}/images', exist_ok=True)

    # Write all pngs
    images = dump['images']
    for i in range(images.shape[0]):
        PIL.Image.fromarray(images[i], 'RGB').save(f'{path_in}/images/{i}.png')


if __name__=='__main__':

    # Set TF session
    tf.InteractiveSession()

    # Load model
    type_ = 'face'
    generator, discriminator = load_model(type_)

    number = 50


    ###
    # Random images generation
    ###

    generate(generator, discriminator, number, f'outputs/{type_}/random')
    write_pngs(f'outputs/{type_}/random')


    ###
    # Generate all images between two random vectors
    ###

    vs = np.random.rand(2, *generator.input_shapes[0][1:])
    v1, v2 = vs[0,:], vs[1,:]

    # Interpolate between both vectors    
    vectors = np.concatenate([[(1 - i / number) * v1 + (i / number) * v2] for i in range(number)])
    generate(generator, discriminator, number, f'outputs/{type_}/interpolation', vectors=vectors)
    write_pngs(f'outputs/{type_}/interpolation')
