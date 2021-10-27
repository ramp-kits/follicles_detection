from PIL import Image


def load_image(path):
    Image.MAX_IMAGE_PIXELS = None
    return Image.open(path)
