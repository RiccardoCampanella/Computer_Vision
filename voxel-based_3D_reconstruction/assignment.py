import glm
import random
import numpy as np
import cv2 as cv
from reconstructor_utils import Reconstructor
import yaml

block_size = 1.0


def generate_grid(width, depth):
    print('generating grid called')
    # Generates the floor grid locations
    # You don't need to edit this function
    data, colors = [], []
    for x in range(width):
        for z in range(depth):
            data.append([x*block_size - width/2, -block_size, z*block_size - depth/2])
            colors.append([1.0, 1.0, 1.0] if (x+z) % 2 == 0 else [0, 0, 0])
    return data, colors


def set_voxel_positions(width, height, depth):
    # Generates random voxel locations
    print('set_voxel_positions called')
    with open("reconstructor_config.yaml") as f:
        reconstructor_config = yaml.load(f, Loader=yaml.FullLoader)
    reconstructor = Reconstructor(reconstructor_config)
    print('reconstructor initialized')
    support_frames = []
    for i in range(4):
        cap = cv.VideoCapture(f'data/cam{i+1}/video.avi')
        ret, frame = cap.read()
        support_frames.append(frame)
        cap.release()
    print('support frames loaded')
    print('reconstructing...')
    reconstruction, voxel_colors = reconstructor.reconstruct(support_frames)
    print('reconstruction done')
    data, colors = [], []
    for x in range(width):
        for y in range(height):
            for z in range(depth):
                # Check if there is a voxel at the (x, y, z) position in the reconstruction matrix
                if reconstruction[x, y, z] == 1:
                    # Calculate the position of the voxel
                    data.append([x*block_size, z*block_size,  y*block_size])
                    
                    '''bgr_color = voxel_colors[x, y, z]
                    rgb_color = [bgr_color[2], bgr_color[1], bgr_color[0]]  # Convert BGR to RGB
                    print(rgb_color)'''
                    colors.append([150, 150, 150])
    return data, colors


def get_cam_positions():
    print('get_cam_positions called')
    cam_positions = []
    for i in range(4):
        cv_file = cv.FileStorage(f'data/cam{i+1}/calibration_params.xml', cv.FILE_STORAGE_READ)
        rvec = cv_file.getNode("Rvec").mat()
        tvec = cv_file.getNode("Tvec").mat()
        cv_file.release()
        rotation_matrix = cv.Rodrigues(rvec)[0]
        position_vector = -np.matrix(rotation_matrix).T * np.matrix(tvec) / 20
        cam_positions.append([position_vector[0][0], -position_vector[2][0], position_vector[1][0]])
    return cam_positions, \
        [[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0], [1.0, 1.0, 0]]


def get_cam_rotation_matrices():
    cam_rotations = []
    for i in range(4):
        # Load rotation vector from calibration data
        cv_file = cv.FileStorage(f'data/cam{i+1}/calibration_params.xml', cv.FILE_STORAGE_READ)
        rvec = cv_file.getNode("Rvec").mat()
        cv_file.release()

        rotation_matrix, _ = cv.Rodrigues(rvec)

        # Initialize a 4x4 identity matrix in glm
        glm_rotation_matrix = glm.mat4(1)

        # Manually set the 3x3 rotation part of the glm matrix
        for row in range(3):
            for col in range(3):
                glm_rotation_matrix[row][col] = rotation_matrix[row, col]

        cam_rotations.append(glm_rotation_matrix)

    return cam_rotations
