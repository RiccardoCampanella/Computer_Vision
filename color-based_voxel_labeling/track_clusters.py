from reconstructor_utils import *
from calibration_utils import *
from tracker_utils import *
import yaml
import numpy as np
from tqdm import tqdm
import pickle


def main():
    with open('reconstructor_config.yaml') as f:
        reconstructor_config = yaml.load(f, Loader=yaml.FullLoader)
    reconstructor_config['x_shift'] = -int(1/6*reconstructor_config['voxel_grid'][0])
    reconstructor_config['y_shift'] = -int(1/4*reconstructor_config['voxel_grid'][1])

    tracker_config = {
        'debug': False,
        'track_grid_top_down': True,
        'show_fbf': False,
        'gmm_clusters': 3,
        'waist_cutoff': 900,
        'head_cutoff': 1450,
        'reinit_camera_frequency': 5,
        'clusters_allignment': 'hungarian',
    }


    k = 2726
    frames_step = 10
    # gather first k frames from each camera
    support_frames_seq = [[None for _ in range(4)] for _ in range(k)]
    for i in range(1, 5):
        cap = cv.VideoCapture(f'data/cam{i}/video.avi')
        for j in range(k):
            ret, frame = cap.read()
            support_frames_seq[j][i-1] = frame
        cap.release()
    
    reconstructor = Reconstructor(reconstructor_config)
    tracker = Tracker(reconstructor, tracker_config, support_frames_seq[0])
    print('Reconstructor and Tracker initialized')

    support_frames_seq = [support_frames_seq[i] for i in range(1, k, frames_step)]
    for i in tqdm(range(1, len(support_frames_seq))):
        tracker.track(support_frames_seq[i])
    

    # save list of dicts with voxel colors for each frame
    with open('voxel_colors_frame_by_frame.pkl', 'wb') as f:
        pickle.dump(tracker.colored_clusters_seq, f)

    with open('trajectory.pkl', 'wb') as f:
        pickle.dump(tracker.colored_trajectory_seq, f)


if __name__ == "__main__":
    main()
