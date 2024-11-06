from flask import Flask, request, jsonify
from flask_cors import CORS
import os, torch, importlib, csv
from demo_utils import *

cfg = CFG()
device = cfg.device

module = importlib.import_module('models.'+cfg.model_type)
model = getattr(module, 'Model')(*cfg.model_params.get_model_params())
model.load_state_dict(torch.load(cfg.save_model_path, map_location=device)['model_state_dict'])
model.to(device)
model.eval()


class_map = {}
reader = csv.reader(open(cfg.class_map_path))
header = next(reader)
for row in reader:
    class_map[int(row[0])] = row[1]

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['OUTPUT_LEN'] = 5
app.config['FILE_NAME'] = 'tmp.mp4'
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024  # 5 MB max size limit

# Ensure the upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'video' not in request.files:
        return jsonify({'message': 'No video file part'}), 400

    file = request.files['video']
    
    if file.filename == '':
        return jsonify({'message': 'No selected file'}), 400

    if file:
        # filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], app.config['FILE_NAME'])
        file.save(file_path)

        # recognise video
        vid_feat = torch.tensor(cfg.test_transform(get_video_data(file_path)), dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            output_list = torch.argsort(model(vid_feat.to(device)).squeeze(0)).cpu().tolist()
        results = {}
        for i in range(1, app.config['OUTPUT_LEN']+1):
            results[f'word{i}'] = class_map[output_list[-i]]
        print(results)
        
        return jsonify(results), 200

    return jsonify({'message': 'File upload failed'}), 500

if __name__ == '__main__':
    app.run(debug=True)
