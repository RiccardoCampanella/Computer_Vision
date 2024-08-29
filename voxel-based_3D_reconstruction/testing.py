import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import cv2 as cv
from reconstructor_utils import *
import yaml


def draw_reconstruction(voxel_reconstruction, colors):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Get the indices of non-zero elements in the voxel_reconstruction array
    on_voxels = np.argwhere(voxel_reconstruction)
    
    # Prepare colors array: convert BGR to RGB and normalize to [0, 1]
    voxel_colors = np.array([colors[tuple(v)] for v in on_voxels])
    voxel_colors = voxel_colors[:, [2, 1, 0]] / 255.0  # Convert BGR to RGB and normalize
    
    # Scatter plot: unpack the indices to x, y, z and apply colors
    ax.scatter(on_voxels[:, 0], on_voxels[:, 1], on_voxels[:, 2], c=voxel_colors, marker='s', s=1)
    
    # Set the labels for axes
    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Z Axis')
    
    # Set equal scaling by scaling the axes limits
    max_range = np.array([voxel_reconstruction.shape[0], voxel_reconstruction.shape[1], voxel_reconstruction.shape[2]]).max() / 2.0
    mid_x = voxel_reconstruction.shape[0] / 2
    mid_y = voxel_reconstruction.shape[1] / 2
    mid_z = voxel_reconstruction.shape[2] / 2
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    ax.set_aspect('auto')

    # Show the plot
    plt.show()


def main():
    with open("reconstructor_config.yaml") as f:
        reconstructor_config = yaml.load(f, Loader=yaml.FullLoader)
    
    support_frames = []
    for i in range(4):
        cap = cv.VideoCapture(f'data/cam{i+1}/video.avi')
        ret, frame = cap.read()
        support_frames.append(frame)
        cap.release()

    support_frames_hsv = [cv.cvtColor(frame, cv.COLOR_BGR2HSV) for frame in support_frames]

    reconstructor = Reconstructor(reconstructor_config)
    fr_masks = reconstructor.get_foreground_masks(support_frames_hsv)
    reconstruction, clrs = reconstructor.reconstruct(support_frames)

    print(reconstruction.shape)
    draw_reconstruction(reconstruction, clrs)

if __name__ == "__main__":
    main()
