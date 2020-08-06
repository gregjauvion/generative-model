
import pickle
import dnnlib


TYPE_MODEL = {
    'church': 'stylegan2-church-config-f.pkl',
    'horse': 'stylegan2-horse-config-f.pkl',
    'face': 'stylegan2-ffhq-config-f.pkl'
}


def load_model(type_):

    with open(f'models/{TYPE_MODEL[type_]}', 'rb') as f:
        _, discriminator, generator = pickle.load(f)

    return generator, discriminator


dnnlib.tflib.init_tf()

g, d = load_model('church')


