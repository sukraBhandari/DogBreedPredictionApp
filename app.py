import os
from flask import Flask, render_template
from flask_bootstrap import Bootstrap
from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileAllowed
from wtforms import SubmitField
from flask_uploads import configure_uploads, patch_request_class
from utils import (image_set, save_image, get_url)
from predictions import dog_or_human_detector


app = Flask(__name__)
basedir = os.path.abspath(os.path.dirname(__file__))
app.config['SECRET_KEY'] = '!!!!Who let the dog out!!!!'
app.config['UPLOADED_IMAGES_DEST'] = os.path.join("static", "images")
bootstrap = Bootstrap(app)
configure_uploads(app, (image_set))
patch_request_class(app, 5 * 1024 * 1024)


class ImageForm(FlaskForm):
    """
    Form for image upload
    """
    file = FileField('Try your dog breed by uploading new image',
                     validators=[FileAllowed(['jpg', 'jpeg', 'png'])])
    submit = SubmitField('Upload')


@app.route('/', methods=['GET', 'POST'])
def index():
    form = ImageForm()
    url = get_url('default-dog.jpg')
    image_path = image_set.path(filename='default-dog.jpg', folder='dogs')
    message = dog_or_human_detector(image_path)
    if form.validate_on_submit() and form.file.data:
        image_path, url = save_image(image=form.file.data)
        message = dog_or_human_detector(image_path)
    return render_template('index.html',
                           image_path=url,
                           prediction=message,
                           form=form)
