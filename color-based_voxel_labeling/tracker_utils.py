import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math

    
class Tracker:
    def __init__(self, reconstructor, config, init_frames):
        self.reconstructor = reconstructor
        self.config = config
        self.voxel_cutoff_bottom = config['waist_cutoff'] // reconstructor.voxel_size[2]+1
        self.voxel_cutoff_top = config['head_cutoff'] // reconstructor.voxel_size[2]+1

        rec, clrs = self.reconstructor.reconstruct(init_frames)
        active_voxels = np.array([k for k in clrs.keys()])
        xy_set = np.array([np.array([v[0], v[1]], dtype=np.float32) for v in active_voxels])
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        flags = cv.KMEANS_RANDOM_CENTERS
        compactness, labels, centers = cv.kmeans(xy_set, 4, None, criteria, 10, flags)

        self.offline_clusters = dict()
        color_trackers = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0)]
        self.top_down_seq = []
        self.center_seq = []
        self.colored_clusters_seq = []
        self.colored_trajectory_seq = []

        colored_clusters = dict()
        colored_trajectory = dict()
        top_down_frame = [None for _ in range(4)]
        for i, clr in enumerate(color_trackers):
            cluster = dict()
            cluster['center'] = centers[i]
            cluster['color'] = clr
            voxels = active_voxels[labels.ravel() == i]
            color_model = cv.ml.EM_create()
            color_model.setClustersNumber(self.config['gmm_clusters'])
            color_model.setCovarianceMatrixType(cv.ml.EM_COV_MAT_GENERIC)
            color_model.setTermCriteria((cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.1))
            color_set = np.float32([clrs[tuple(v)] for v in voxels if v[2] >= self.voxel_cutoff_bottom and v[2] <= self.voxel_cutoff_top])

            if self.config['debug']:
                rec_dict = {tuple(v): clrs[tuple(v)] for v in voxels if v[2] >= self.voxel_cutoff_bottom and v[2] <= self.voxel_cutoff_top}
                draw_reconstruction(rec_dict)
                create_color_image(color_set)
                draw_mask(self.reconstructor, rec_dict, init_frames)
                print(len(color_set))
            
            color_model.trainEM(color_set)
            cluster['color_model'] = color_model
            self.offline_clusters[i] = cluster
            for v in voxels:
                colored_clusters[tuple(v)] = clr
            
            colored_trajectory[tuple(centers[i])] = clr

            # save top down frame view for quick sequence display and debugging
            if self.config['track_grid_top_down']:
                top_down_frame[i] = np.unique([[v[0], v[1]] for v in voxels], axis=0)
            
        if self.config['track_grid_top_down']:
            self.top_down_seq.append(top_down_frame)
            if self.config['show_fbf']:
                self.show_top_down_frame(top_down_frame)
        
        self.colored_clusters_seq.append(colored_clusters)
        self.colored_trajectory_seq.append(colored_trajectory)
        self.center_seq.append(centers)
        self.track_ctr = 1

    
    def track(self, support_frames, num_clusters=4):
        rec, clrs = self.reconstructor.reconstruct(support_frames)
        active_voxels = np.array([k for k in clrs.keys()])
        xy_set = np.array([np.array([v[0], v[1]], dtype=np.float32) for v in active_voxels])
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        flags = cv.KMEANS_RANDOM_CENTERS
        try:
            compactness, labels, centers = cv.kmeans(xy_set, num_clusters, None, criteria, 10, flags)
        except:
            print('KMeans failed')
            return


        if num_clusters > 4:
            labels, centers, self.refine_clusters(active_voxels, labels, centers, num_clusters)

        cur_centers = [None for _ in range(num_clusters)]
        top_down_frame = [None for _ in range(4)]
        colored_clusters = dict()
        colored_trajectory = dict()

        cost_matrix = np.zeros((4, 4))
        for i in range(num_clusters):
            voxels = active_voxels[labels.ravel() == i]
            color_set = np.float32([clrs[tuple(v)] for v in voxels if v[2] >= self.voxel_cutoff_bottom and v[2] <= self.voxel_cutoff_top])
            probs = []
            for j in range(4):
                mean_log_proba = np.mean([self.offline_clusters[j]['color_model'].predict2(sample)[0][0] for sample in color_set])
                probs.append(mean_log_proba)
                cost_matrix[i, j] = mean_log_proba
            
            if self.config['debug']:
                rec_dict = {tuple(v): clrs[tuple(v)] for v in voxels if v[2] >= self.voxel_cutoff_bottom and v[2] <= self.voxel_cutoff_top}
                draw_reconstruction(rec_dict)
                create_color_image(color_set)
                draw_mask(self.reconstructor, rec_dict, support_frames)
                print(len(color_set), np.mean(color_set, axis=0))
                print(probs)
            
            if self.config['clusters_allignment'] == 'argmax':
                cluster_idx = np.argmax(probs)
                color = self.offline_clusters[cluster_idx]['color']
                center = centers[i]
                cur_centers[cluster_idx] = center
                colored_trajectory[tuple(center)] = color
                if self.config['track_grid_top_down']:
                    top_down_frame[cluster_idx] = np.unique([[v[0], v[1]] for v in voxels], axis=0)
                
                for v in voxels:
                    colored_clusters[tuple(v)] = color
        
        if self.config['clusters_allignment'] == 'hungarian':
            row_ind, col_ind = linear_sum_assignment(-cost_matrix)
            for i, j in zip(row_ind, col_ind):
                voxels = active_voxels[labels.ravel() == i]
                cluster_idx = j
                color = self.offline_clusters[cluster_idx]['color']
                center = centers[i]
                cur_centers[cluster_idx] = center
                colored_trajectory[tuple(center)] = color
                if self.config['track_grid_top_down']:
                    top_down_frame[cluster_idx] = np.unique([[v[0], v[1]] for v in voxels], axis=0)
                for v in voxels:
                    colored_clusters[tuple(v)] = color

            
        self.center_seq.append(cur_centers)
        self.colored_clusters_seq.append(colored_clusters)
        self.colored_trajectory_seq.append(colored_trajectory)
        if self.config['track_grid_top_down']:
            self.top_down_seq.append(top_down_frame)
            if self.config['show_fbf']:
                self.show_top_down_frame(top_down_frame)
        
        self.track_ctr += 1
        if self.track_ctr == self.config['reinit_camera_frequency']:
            self.reconstructor.reinitalize_cameras()
            self.track_ctr = 0

    

    def refine_clusters(self, active_voxels, labels, centers, num_clusters):
        pass

    def show_top_down_frame(self, top_down_frame):
        # draw grid of size config['voxel_grid'] and color tiles with coordinates from top_down_frame according to self.offline_clusters[i]['color']
        grid_size = self.reconstructor.config['voxel_grid']
        fig, ax = plt.subplots()

        # Create a white grid to start with
        ax.set_xlim(0, grid_size[1])
        ax.set_ylim(0, grid_size[0])
        ax.set_aspect('equal')
        ax.set_facecolor('white')

        # Iterate over the clusters and their tiles
        for i, cluster_tiles in enumerate(top_down_frame):
            if type(cluster_tiles) == type(None):
                continue
            color = tuple(c/255 for c in self.offline_clusters[i]['color'])
            for tile in cluster_tiles:
                # Draw each tile with the corresponding cluster color
                rect = patches.Rectangle((tile[1], tile[0]), 1, 1, linewidth=1, edgecolor=color, facecolor=color)
                ax.add_patch(rect)

        # Hide the axes, grid lines, and ticks
        ax.axis('off')

        # Display the grid
        plt.show()


def project_cube_corners(img, reconstructor, camera_num):
    frame = img.copy()
    pt1 = [0, 0, 0]
    pt3 = [0, 100, 0]
    pt4 = [115, 0, 0]
    pt6 = [115, 100, 0]
    pts = np.array([pt1, pt3, pt4, pt6])

    for pt in pts:
        x, y, z = pt
        voxel_center_world = np.array([(x + reconstructor.config['x_shift'] + 0.5) * reconstructor.voxel_size[0], 
                                                    (y + reconstructor.config['y_shift'] + 0.5) * reconstructor.voxel_size[1], 
                                                    -(z + reconstructor.config['z_shift'] - 0.5) * reconstructor.voxel_size[2]], dtype=np.float32)
        
        pt_2d = reconstructor.cameras[camera_num].project_voxel_to_image(voxel_center_world)
        frame = cv.circle(frame, tuple(pt_2d), 10, (0, 255, 0), -1)
    
    plt.imshow(frame)


def draw_reconstruction(clrs):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Get the indices of non-zero elements in the voxel_reconstruction array
    on_voxels = np.array([x for x in clrs.keys()])
    
    # Prepare colors array: convert BGR to RGB and normalize to [0, 1]
    voxel_colors = np.array([clrs[tuple(v)] for v in on_voxels])
    voxel_colors = voxel_colors[:, [2, 1, 0]] / 255.0  # Convert BGR to RGB and normalize
    
    # Scatter plot: unpack the indices to x, y, z and apply colors
    ax.scatter(on_voxels[:, 0], on_voxels[:, 1], on_voxels[:, 2], c=voxel_colors, marker='s', s=1)
    
    # Set the labels for axes
    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Z Axis')
    
    # Set equal scaling by scaling the axes limits
    max_range = np.array([100, 200, 100]).max() / 2.0
    mid_x = 100 / 2
    mid_y = 200 / 2
    mid_z = 100 / 2
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    ax.set_aspect('auto')

    # Show the plot
    plt.show()

def create_color_image(cluster_colors):
    """
    Creates and displays an image where each pixel represents a color from the cluster.

    Parameters:
    - cluster_colors: A list of color values in the cluster.
    """
    # Calculate dimensions to create a square (or nearly square) image
    num_colors = len(cluster_colors)
    side_length = math.ceil(math.sqrt(num_colors))
    
    # Initialize a blank (black) image
    image = np.zeros((side_length, side_length, 3), dtype=np.uint8)
    
    # Fill the image with cluster colors
    for i, color in enumerate(cluster_colors):
        # Convert color to RGB (if needed) since OpenCV uses BGR
        # If your colors are already in RGB, you can skip this line
        color_rgb = color[::-1]
        
        # Calculate position in the image
        row = i // side_length
        col = i % side_length
        
        # Assign the color
        image[row, col] = color_rgb
    
    # Display the image
    plt.imshow(image)
    plt.axis('off')  # Turn off axis numbers and ticks
    plt.show()

def draw_mask(reconstructor, clrs, support_frames):
    main_camera = reconstructor.config['color_camera']-1
    frame = support_frames[main_camera]
    mask = np.zeros(frame.shape, dtype=np.uint8)
    for k, v in clrs.items():
        x, y, z = k
        voxel_center_world = np.array([(x + reconstructor.config['x_shift'] + 0.5) * reconstructor.voxel_size[0], 
                                                   (y + reconstructor.config['y_shift'] + 0.5) * reconstructor.voxel_size[1], 
                                                   -(z + reconstructor.config['z_shift'] - 0.5) * reconstructor.voxel_size[2]], dtype=np.float32)
        pt_2d = reconstructor.cameras[main_camera].project_voxel_to_image(voxel_center_world)
        b, g, r = clrs[k]
        mask[pt_2d[1], pt_2d[0], :] = [r, g, b]
    plt.imshow(mask, cmap='gray')
    plt.show()
