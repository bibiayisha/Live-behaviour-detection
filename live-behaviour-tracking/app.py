def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import cv2
import numpy as np
import silence_tensorflow.auto
from flask import Flask, render_template, Response, jsonify, current_app
from backend.preprocessing import preprocessing
from backend.load_model import load_model

TAG = '[app]'
app = Flask(__name__)
model = load_model()
camera = None
prediction = ''
start_video_flag = False
IMAGE_HEIGHT, IMAGE_WIDTH = 256, 256
CLASSES_LIST = [
    'Argue',
    'Eating In Class',
    'Explainig The Subject',
    'Hand Raise',
    'Holding Book',
    'Holding Mobile',
    'Reading Book',
    'Sitting On Desk',
    'Writing On Board',
    'Writing On Textbook',
    'Unknown',
]
label_map = {
    'Argue': 'abnormal',
    'Eating In Class': 'abnormal',
    'Explainig The Subject': 'normal',
    'Hand Raise': 'normal',
    'Holding Book': 'normal',
    'Holding Mobile': 'abnormal',
    'Reading Book': 'normal',
    'Sitting On Desk': 'abnormal',
    'Writing On Board': 'abnormal',
    'Writing On Textbook': 'normal',
    'Unknown': 'Unknown',
}

def read_frame():
    global camera
    success, camera_img = camera.read()
    # this loop will make sure that successful frame is read from camera
    while not success:
        success, camera_img = camera.read()
        print('[read_frame][stuck in while loop]', camera, success, type(camera_img))
    camera_img = cv2.flip(camera_img, 1)
    record = preprocessing(camera_img)
    return success, camera_img, record

def generate_frames():
    # overall algorithm:
    # - initially read 15 frames
    # - as soon as start button is pressed, read the 16th frame
    # - inference model to get output and calculate prediction string
    # - return random frame if stop button is pressed or start button is not pressed (initially)
    global prediction, start_video_flag, camera
    features = []
    camera = cv2.VideoCapture(0)
    # read the first 15 camera frames, this will execute only once initially
    while len(features) < 15:
        success, camera_img, record = read_frame()
        features.append(record)
    camera.release()
    camera = None

    while True:
        # print('[camera]', camera)
        if camera is not None and start_video_flag:
            # if the start button is pressed on front-end then read 16th frame
            success, camera_img, record = read_frame()
            features.append(record)
            X = np.asarray(features)
            # discard the most previous frame (0th index frame)
            features = features[1:]

            if not success:
                break
            else:
                if X is not None:
                    X = np.reshape(X, (1, 16, 128, 128, 3))
                    output = model.predict(X)
                    predicted_class_index = np.argmax(output)
                    predicted_class = CLASSES_LIST[predicted_class_index]
                    category_pred = label_map[predicted_class]
                    prediction = f"{category_pred}: {predicted_class}"
                    ret, buffer = cv2.imencode('.jpg', camera_img)
                    frame = buffer.tobytes()
                else:
                    prediction = "<testing>"
        else:
            frame = np.random.uniform(0, 256, (128, 128, 3))
            frame = cv2.resize(frame, dsize=(128, 128))
            ret, frame = cv2.imencode('.jpg', frame)
            frame = frame.tobytes()
            prediction = "Camera is OFF"
        yield(b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    # print('[index][render_template]\n', render_template('index.html'))
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/result', methods=['POST'])
def result():
    global prediction
    # prediction = np.random.randint(0, 256)
    # print('[result][prediction]', prediction, type(prediction))
    return jsonify({'prediction': str(prediction)})

@app.route('/start_video')
def start_video():
    global start_video_flag, camera
    start_video_flag = True
    if camera is None:
        camera = cv2.VideoCapture(0)
    return ('', 204)

@app.route('/stop_video')
def stop_video():
    global start_video_flag, camera
    start_video_flag = False
    if camera is not None:
        camera.release()
        camera = None
    return ('', 204)

if __name__ == '__main__':
    app.run(debug=True)
