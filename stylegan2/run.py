
import pickle
import dnnlib
import dnnlib.tflib as tflib
import numpy as np


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
    noise_vars = [var for name, var in g.components.synthesis.vars.items() if name.startswith('noise')]
    for e in range(number):
        tflib.set_vars({var: rnd.randn(*var.shape.as_list()) for var in noise_vars}) # [height, width]
        images = np.concatenate((images, g.run(latents, None)))
        # TODO compute probabilities

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
        PIL.Image.fromarray(images[i], 'RGB').save(f'{path_in}/images/{i}.png')


if __name__=='__main__':

    # Set TF session
    tflib.init_tf()

    # Load model
    type_ = 'face'
    generator, discriminator = load_model('face')


    ###
    # Random images generation
    ###

    generate(generator, discriminator, number, f'outputs/{type_}/random')
    write_pngs(f'outputs/{type_}/random')
