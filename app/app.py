from flask import Flask,render_template,request
import numpy as np
import tensorflow as tf
import base64
import os

model_path = os.path.join(os.path.dirname(__file__), 'static', 'quantized_model.tflite')
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()


class_names=['Healthy','Multiple Diseases','Rust','Scab']

def predict_model(encoded):

    image_data = base64.b64decode(encoded)

    image = tf.io.decode_jpeg(image_data, channels=3)
    
    # Convert image to float32 and normalize pixel values
    image = tf.cast(image, tf.float32) / 255.0
    
    # Resize the image to the required dimensions
    image = tf.image.resize(image, (256, 256))
    
    # Add a batch dimension to the image
    image = tf.expand_dims(image, axis=0)

    # Perform prediction with the model
    interpreter.set_tensor(input_details[0]['index'], image)
    interpreter.invoke()
    output_details = interpreter.get_output_details()
    predictions = interpreter.get_tensor(output_details[0]['index'])

    return [class_names[np.argmax(predictions)],np.round(np.max(predictions)*100,3)]
    
    

app=Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html',title='Home')

@app.route('/about')
def about():
    return render_template('about.html',title='About')

@app.route('/',methods=['POST'])
def predict():

    img = request.files['imagefile']
    image_data = img.read()
    encoded_image = base64.b64encode(image_data).decode('utf-8')

    classification=predict_model(encoded_image)

    
    return render_template('result.html',result=classification[0],percentage=classification[1],image_data=encoded_image)




if __name__=='__main__':
    app.run()