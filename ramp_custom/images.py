from PIL import Image


def load_image(path):
    """Load an image.

    This is usefull because a submission cannot contain
    the keywork 'open'

    Parameters
    ----------
    path : str
        absolute path to image

    Returns
    -------
    image : PIL.Image

    """
    Image.MAX_IMAGE_PIXELS = None
    return Image.open(path)
