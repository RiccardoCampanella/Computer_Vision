import glm
import random
import numpy as np
import pickle

block_size = 1.0


def generate_grid(width, depth):
    # Generates the floor grid locations
    # You don't need to edit this function
    data, colors = [], []
    for x in range(width):
        for z in range(depth):
            data.append([x*block_size - width/2, -block_size, z*block_size - depth/2])
            colors.append([1.0, 1.0, 1.0] if (x+z) % 2 == 0 else [0, 0, 0])
    return data, colors


def set_voxel_positions(width, height, depth, iterator=0):
    # Generates random voxel locations
    # load voxel_colors_frame_by_frame.pkl
    with open("voxel_colors_frame_by_frame.pkl", "rb") as f:
        voxel_colors_frame_by_frame = pickle.load(f)
    with open("trajectory.pkl", "rb") as f:
        trajectory = pickle.load(f)
    
    draw_clusters = True
    if iterator >= len(voxel_colors_frame_by_frame):
        iterator = len(voxel_colors_frame_by_frame) - 1
        draw_clusters = False
    
    cur_reconstruction = voxel_colors_frame_by_frame[iterator]
    cur_trajectory = [trajectory[i] for i in range(iterator)]
    
    data, colors = [], []
    if draw_clusters:
        for x, y, z in cur_reconstruction.keys():
            data.append([(x-19)*block_size, z*block_size,  (y-25)*block_size])
            bgr_color = cur_reconstruction[(x, y, z)]
            color = [bgr_color[2], bgr_color[1], bgr_color[0]]  # Convert BGR to RGB
            colors.append(color)
    
    for points in cur_trajectory:
        for x, y in points.keys():
            data.append([(x-19)*block_size, 1,  (y-25)*block_size])
            bgr_color = points[(x, y)]
            color = [bgr_color[2], bgr_color[1], bgr_color[0]]  # Convert BGR to RGB
            colors.append(color)
    return data, colors


def get_cam_positions():
    # Generates dummy camera locations at the 4 corners of the room
    # TODO: You need to input the estimated locations of the 4 cameras in the world coordinates.
    return [[-64 * block_size, 64 * block_size, 63 * block_size],
            [63 * block_size, 64 * block_size, 63 * block_size],
            [63 * block_size, 64 * block_size, -64 * block_size],
            [-64 * block_size, 64 * block_size, -64 * block_size]], \
        [[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0], [1.0, 1.0, 0]]


def get_cam_rotation_matrices():
    # Generates dummy camera rotation matrices, looking down 45 degrees towards the center of the room
    # TODO: You need to input the estimated camera rotation matrices (4x4) of the 4 cameras in the world coordinates.
    cam_angles = [[0, 45, -45], [0, 135, -45], [0, 225, -45], [0, 315, -45]]
    cam_rotations = [glm.mat4(1), glm.mat4(1), glm.mat4(1), glm.mat4(1)]
    for c in range(len(cam_rotations)):
        cam_rotations[c] = glm.rotate(cam_rotations[c], cam_angles[c][0] * np.pi / 180, [1, 0, 0])
        cam_rotations[c] = glm.rotate(cam_rotations[c], cam_angles[c][1] * np.pi / 180, [0, 1, 0])
        cam_rotations[c] = glm.rotate(cam_rotations[c], cam_angles[c][2] * np.pi / 180, [0, 0, 1])
    return cam_rotations
