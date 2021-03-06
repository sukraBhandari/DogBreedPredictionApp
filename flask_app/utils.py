import os
import uuid
from PIL import Image
from flask import url_for
from flask_uploads import UploadSet, extension

# accept only following set of image extensions.
image_set = UploadSet('images', ('png', 'jpg', 'jpeg', 'PNG', 'JPG', 'JPEG'))


def get_url(image):
    """
    function to return image static url
    """
    return url_for('static', filename='images/dogs/{}'.format(image), _external=True)


def save_image(image):
    """
    function to store images on the server
    """
    temp_image = '{}.{}'.format(uuid.uuid4(), extension(image.filename))

    image_set.save(image, folder='dogs', name=temp_image)
    image_path = image_set.path(filename=temp_image, folder='dogs')
    image = Image.open(image_path)
    if image.mode != "RGB":
        image = image.convert("RGB")
    if max(image.width, image.height) > 500:
        image.thumbnail((500, 500))
    thumbnail_filename = '{}.jpg'.format(uuid.uuid4())
    thumbnail_path = image_set.path(filename=thumbnail_filename,
                                    folder='dogs')
    image.save(thumbnail_path, optimize=True, quality=95)
    os.remove(image_path)

    return thumbnail_path, get_url(thumbnail_filename)
