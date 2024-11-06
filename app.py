import os
import tempfile
import cv2
from flask import Flask, jsonify, render_template, request, redirect
import csv, importlib, torch, time
from demo_utils import *
from queue import Queue
from threading import Thread

app = Flask(__name__)

cfg = CFG()
device = cfg.device

class_map = {}
reader = csv.reader(open(cfg.class_map_path))
header = next(reader)
for row in reader:
    class_map[int(row[0])] = row[1]

pose_model = Model()
module = importlib.import_module('models.'+cfg.model_type)
model = getattr(module, 'Model')(*cfg.model_params.get_model_params())
model.load_state_dict(torch.load(cfg.save_model_path, map_location=device)['model_state_dict'])
model.to(device)
model.eval()

# Default settings
top_k = 5
record_duration = 5
is_processing = False
frame_queue = Queue(100)
results = []

def recognise(data):
    vid_feat = torch.tensor(cfg.test_transform(data), dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        output_list = torch.argsort(model(vid_feat.to(device)).squeeze(0)).cpu().tolist()
    res = []
    for i in range(1, top_k+1):
        res.append(class_map[output_list[-i]])
    return res

def get_frame_feature():
    global is_processing, frame_queue, results
    features = []
    vid_height, vid_width = 0, 0
    while True:
        if not frame_queue.empty():
            frame = frame_queue.get()
            vid_height, vid_width = frame.shape[:2]
            features.append(pose_model(frame)[0])

        if is_processing and frame_queue.empty():
            data = {
                'feat': np.array(features),
                'num_frames': len(features),
                'vid_name': 'none',
                'vid_width': vid_width,
                'vid_height': vid_height
                }
            results = recognise(data)
            print(results)
            features = []
            is_processing = False

@app.route('/')
def index():
    global top_k, record_duration, results
    return render_template('index.html', top_k=top_k, record_duration=record_duration, results=results)

@app.route('/settings', methods=['POST'])
def settings():
    global top_k, record_duration
    top_k = int(request.form['top_k'])
    record_duration = int(request.form['record_duration'])
    return redirect('/')

@app.route('/record', methods=['POST'])
def receive_frame():
    global frame_queue, is_processing
    video_data = request.get_data()
    
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_file:
        temp_file.write(video_data)
        temp_file_path = temp_file.name

        cap = cv2.VideoCapture(temp_file_path)

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_queue.put(frame)

        cap.release()
    os.remove(temp_file_path)
    is_processing = True
    return 'Frame received successfully'

@app.route('/result', methods=['GET'])
def send_results():
    global results, top_k
    results = []
    while len(results) < top_k:
        time.sleep(0.1)
    return jsonify(results)

if __name__ == '__main__':
    feature_process = Thread(target=get_frame_feature)
    feature_process.daemon = 1
    feature_process.start()
    app.run(debug=True, host='0.0.0.0', port=5000)