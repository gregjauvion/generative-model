
import pickle
import dnnlib
import dnnlib.tflib as tflib
import numpy as np
import os
import PIL
import tensorflow as tf


BATCH_SIZE = 5

TYPE_MODEL = {
    'church': 'stylegan2-church-config-f.pkl',
    'horse': 'stylegan2-horse-config-f.pkl',
    'face': 'stylegan2-ffhq-config-f.pkl'
}


def load_model(type_):

    with open(f'models/{TYPE_MODEL[type_]}', 'rb') as f:
        _, discriminator, generator = pickle.load(f, encoding='latin1')

    return generator, discriminator


def generate_images(generator, discriminator, number, path_out, vectors=None):

    rnd = np.random.RandomState(1234321)

    # Generate random latent vectors if not specified
    if vectors is None:
        vectors = rnd.randn(number, *generator.input_shape[1:])

    # Generate images one by one
    images, probabilities = None, None
    noise_vars = [var for name, var in generator.components.synthesis.vars.items() if name.startswith('noise')]
    for i in range(number // BATCH_SIZE):
        print(f'Batch {i}...')
        tflib.set_vars({var: rnd.randn(*var.shape.as_list()) for var in noise_vars}) # [height, width]

        imgs = generator.run(vectors[i*BATCH_SIZE:(i + 1)*BATCH_SIZE], None)
        probas = discriminator.run(imgs, None).reshape(-1)
        if images is None:
            images, probabilities = imgs, probas
        else:
            images = np.concatenate((images, imgs))
            probabilities = np.concatenate((probabilities, probas))

    # Convert images to PIL-compatible format
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
        print(f'Building image {i}...')
        PIL.Image.fromarray(images[i], 'RGB').save(f'{path_in}/images/{i}.png')


if __name__=='__main__':

    # Set TF session
    tflib.init_tf()
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

    # Load model
    type_ = 'face'
    generator, discriminator = load_model('face')

    number = 20


    ###
    # Random images generation
    ###

    generate(generator, discriminator, number, f'outputs/{type_}/random')
    write_pngs(f'outputs/{type_}/random')


    ###
    # Generate all images between two random vectors
    ###

    vs = np.random.RandomState(123454321).randn(2, *generator.input_shape[1:])
    v1, v2 = vs[0,:], vs[1,:]

    # Interpolate between both vectors    
    vectors = np.concatenate([[(1 - i / number) * v1 + (i / number) * v2] for i in range(number)])
    generate(generator, discriminator, number, f'outputs/{type_}/interpolation', vectors=vectors)
    write_pngs(f'outputs/{type_}/interpolation')
