import os, pickle, torch, importlib, csv
from pose_modules.Mediapipe import Model
from decord import VideoReader, cpu
import numpy as np
from data_transform import *
from models.model_params import SignAttention_v6Params

class CFG():
    def __init__(self):
        cuda_id = 0
        self.mode = 'test'
        self.src_len = 192
        self.feature_type = 'keypoints'
        self.input_dim = 2

        self.model_type = 'SignAttention_v6'

        self.device = torch.device(
            f"cuda:{cuda_id}" if torch.cuda.is_available() else "cpu")

        print("Running on device = ", self.device)

        self.model_params = SignAttention_v6Params({'num_class' : 2002, 'src_len' : 192}, self.input_dim, self.device)

        self.batch_size = 1

        self.origin_idx = 0 #nose point
        self.anchor_points = [3, 4] #shoulder points
        self.left_slice = [9, 19, 7] #left hand slice
        self.right_slice = [19, 29, 8] #right hand slice

        kps = [0, 2, 5, 11, 12, 13, 14, 15, 16] + [0+33+468, 4+33+468, 5+33+468, 8+33+468, 9+33+468, 12+33+468, 13+33+468, 16+33+468, 17+33+468, 20+33+468,
             0+21+33+468, 4+21+33+468, 5+21+33+468, 8+21+33+468, 9+21+33+468, 12+21+33+468, 13+21+33+468, 16+21+33+468,
               17+21+33+468, 20+21+33+468]

        self.test_transform = self.val_transform = Compose([MediapipeDataProcess(),
                                                            PoseSelect(kps, [0, 1]),
                                                            HandCorrection(self.left_slice, self.right_slice),
                                                            NormalizeKeypoints(self.origin_idx, self.anchor_points),
                                                            TemporalSample(self.src_len),
                                                            WindowCreate(self.src_len),
                                                            ])
        
        self.class_map_path = 'input/FDMSE/class_map_FDMSE.csv'
        self.save_model_path = 'output/FDMSE/SignAttention_v6_240402_1556/model_best_loss.pt'

def get_video_features(vid_name) -> list:
    pose_model = Model()
    kp_shape = (543, 4)

    if type(vid_name) is str:
        cap = VideoReader(vid_name, ctx=cpu(0))
    else:
        cap = VideoReader(vid_name, ctx=cpu(0))

    num_frames = len(cap)
    vid_height, vid_width = cap[0].shape[:2]

    features = np.zeros((num_frames, *kp_shape))

    i_th_frame = 0

    for image in cap:
        # saving the i-th frame feature
        features[i_th_frame] = pose_model(image.asnumpy())[0]
        i_th_frame += 1

    return vid_name, features, i_th_frame, vid_width, vid_height

def get_video_data(video):
    vid_name, features, num_frames, vid_width, vid_height = get_video_features(video)
    data = {
            'feat': features,
            'num_frames': num_frames,
            'vid_name': vid_name,
            'vid_width': vid_width,
            'vid_height': vid_height
            }
    return data
