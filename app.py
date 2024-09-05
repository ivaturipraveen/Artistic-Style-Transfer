from flask import Flask, request, render_template, send_from_directory, redirect, url_for
import os
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import PIL.Image

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['CONTENT_FOLDER'] = os.path.join(app.config['UPLOAD_FOLDER'], 'content')
app.config['STYLE_FOLDER'] = os.path.join(app.config['UPLOAD_FOLDER'], 'style')
app.config['RESULT_FOLDER'] = 'static/images/'

os.makedirs(app.config['CONTENT_FOLDER'], exist_ok=True)
os.makedirs(app.config['STYLE_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)

def load_img(path_to_img):
    max_dim = 512
    img = tf.io.read_file(path_to_img)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    shape = tf.cast(tf.shape(img)[:-1], tf.float32)
    long_dim = max(shape)
    scale = max_dim / long_dim
    new_shape = tf.cast(shape * scale, tf.int32)
    img = tf.image.resize(img, new_shape)
    img = img[tf.newaxis, :]
    return img

def tensor_to_image(tensor):
    tensor = tensor * 255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor) > 3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return PIL.Image.fromarray(tensor)

def perform_style_transfer(content_path, style_path, result_path):
    content_image = load_img(content_path)
    style_image = load_img(style_path)
    
    hub_model = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')
    stylized_image = hub_model(tf.constant(content_image), tf.constant(style_image))[0]
    
    result_image = tensor_to_image(stylized_image)
    result_image.save(result_path)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        content_file = request.files['content_image']
        style_file = request.files['style_image']
        
        if content_file and style_file:
            content_path = os.path.join(app.config['CONTENT_FOLDER'], content_file.filename)
            style_path = os.path.join(app.config['STYLE_FOLDER'], style_file.filename)
            
            content_file.save(content_path)
            style_file.save(style_path)
            
            result_path = os.path.join(app.config['RESULT_FOLDER'], 'stylized_image.jpg')
            perform_style_transfer(content_path, style_path, result_path)
            
            return redirect(url_for('result'))
    
    return render_template('index.html')

@app.route('/result')
def result():
    return render_template('index.html', image_url=url_for('static', filename='images/stylized_image.jpg'))

@app.route('/download')
def download():
    return send_from_directory(app.config['RESULT_FOLDER'], 'stylized_image.jpg', as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
