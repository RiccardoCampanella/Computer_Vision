import cv2 as cv
import numpy as np

def process_frames_MOG(video_path, backSub, update_model=True, display_result=False):
    cap = cv.VideoCapture(video_path)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to HSV color space
        hsv_frame = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        
        # Apply the background subtractor to get the foreground mask
        fgMask = backSub.apply(hsv_frame, learningRate=-1 if update_model else 0)

        if display_result:
            cv.imshow('Frame', frame)
            cv.imshow('FG Mask', fgMask)
            
            keyboard = cv.waitKey(30)
            if keyboard == 'q' or keyboard == 27:
                break

    cap.release()
    cv.destroyAllWindows()


def refine_mask_with_hsv_thresholding(hsv_frame, fgMask, config):
    # Define thresholds for Hue, Saturation, and Value channels
    hue_threshold = config['hue_threshold'] # Adjust these values based on your needs
    sat_threshold = config['saturation_threshold']  # Adjust these values based on your needs
    val_threshold = config['value_threshold'] # Adjust these values based on your needs

    # Threshold the HSV image to get only the colors in the specified ranges
    hue_mask = cv.inRange(hsv_frame[:, :, 0], hue_threshold[0], hue_threshold[1])
    sat_mask = cv.inRange(hsv_frame[:, :, 1], sat_threshold[0], sat_threshold[1])
    val_mask = cv.inRange(hsv_frame[:, :, 2], val_threshold[0], val_threshold[1])

    # Combine the HSV masks with the initial foreground mask using logical AND
    combined_mask = cv.bitwise_and(fgMask, fgMask, mask=hue_mask)
    combined_mask = cv.bitwise_and(combined_mask, combined_mask, mask=sat_mask)
    combined_mask = cv.bitwise_and(combined_mask, combined_mask, mask=val_mask)

    if 'erosion_dilation' in config:
        kernel = np.ones(config['kernel'], np.uint8)
        for layer in config['erosion_dilation']:
            if layer == 'erosion':
                combined_mask = cv.erode(combined_mask, kernel, iterations=config['erosion_iterations'])
            elif layer == 'dilation':
                combined_mask = cv.dilate(combined_mask, kernel, iterations=config['dilation_iterations'])
            else:
                raise ValueError(f"Unknown layer type {layer}")

    return combined_mask


class Camera:
    def __init__(self, config, path, backSub):
        # read camera parameters from xml file
        self.config = config
        cv_file = cv.FileStorage(path+'calibration_params.xml', cv.FILE_STORAGE_READ)
        self.mtx = cv_file.getNode("CameraMatrix").mat()
        self.dist = cv_file.getNode("DistortionCoeffs").mat()
        self.rvec = cv_file.getNode("Rvec").mat()
        self.tvec = cv_file.getNode("Tvec").mat()
        cv_file.release()
        self.backSub = backSub
        process_frames_MOG(path+'background.avi', backSub, update_model=True, display_result=False)
        self.video_path = path+'video.avi'
    
    def get_foreground_mask(self, hsv_frame):
        fgMask = self.backSub.apply(hsv_frame, 0.000001)
        refined_mask = refine_mask_with_hsv_thresholding(hsv_frame, fgMask, self.config)
        return refined_mask
    
    def project_voxel_to_image(self, voxel_center):
        """
        Project a single 3D voxel center to a 2D point in the camera image plane.
        """
        voxel_center_reshaped = np.array(voxel_center, dtype=np.float32).reshape(-1, 1, 3)
        projected_points, _ = cv.projectPoints(voxel_center_reshaped, self.rvec, self.tvec, self.mtx, self.dist)
        return tuple(projected_points[0, 0, :].astype(int))
    

    def calculate_depth(self, point_world):
        # Convert rotation vector to rotation matrix
        R, _ = cv.Rodrigues(self.rvec)
        
        # Transform the point from world coordinates to camera coordinates
        point_camera = np.dot(R, point_world) + self.tvec.reshape(-1)
        
        # Depth is the Z component in the camera coordinates
        depth = point_camera[2]
        
        return depth
    

        

class Reconstructor:
    def __init__(self, config):
        self.config = config
        self.voxel_size = [int(config['space_size'][i] / config['voxel_grid'][i]) for i in range(3)]

        self.cameras = dict()
        for cam in range(self.config['num_cams']):
            backSub = cv.bgsegm.createBackgroundSubtractorMOG()
            self.cameras[cam] = Camera(self.config['camera_config'], f'data/cam{cam+1}/', backSub)

        self.lookup_table = dict()
        for x in range(self.config['voxel_grid'][0]):
            for y in range(self.config['voxel_grid'][1]):
                for z in range(self.config['voxel_grid'][2]):
                    # Calculate the world coordinates of the voxel center
                    voxel_center_world = np.array([(x + self.config['x_shift'] + 0.5) * self.voxel_size[0], 
                                                   (y + self.config['y_shift'] + 0.5) * self.voxel_size[1], 
                                                   -(z + self.config['z_shift'] - 0.5) * self.voxel_size[2]], dtype=np.float32)
                    for i in range(self.config['num_cams']):
                        cam = self.cameras[i]
                        projected_point = cam.project_voxel_to_image(voxel_center_world)
                        depth = cam.calculate_depth(voxel_center_world)
                        self.lookup_table[(x, y, z, i)] = (projected_point[0], projected_point[1], depth, i)
    

    def reinitialize_background_model(self):
        for i in range(self.config['num_cams']):
            backSub = cv.bgsegm.createBackgroundSubtractorMOG()
            self.cameras[i].backSub = backSub
            process_frames_MOG(self.cameras[i].video_path, backSub, update_model=True, display_result=False)
    

    def reinitalize_cameras(self):
        self.cameras = dict()
        for cam in range(self.config['num_cams']):
            backSub = cv.bgsegm.createBackgroundSubtractorMOG()
            self.cameras[cam] = Camera(self.config['camera_config'], f'data/cam{cam+1}/', backSub)


        
    def get_foreground_masks(self, support_frames):
        masks = []
        for cam in self.cameras:
            masks.append(self.cameras[cam].get_foreground_mask(support_frames[cam]))
        return masks
    
    def reconstruct(self, support_frames, camera_list = [0, 1, 2, 3]):
        bgr_frames = support_frames.copy()
        support_frames = [cv.cvtColor(frame, cv.COLOR_BGR2HSV) for frame in support_frames]
        foreground_masks = self.get_foreground_masks(support_frames)
        if 'reduce_noise' in self.config and self.config['reduce_noise']:
            foreground_masks = [self.reduce_noise(fg) for fg in foreground_masks]
        if self.config['show_masks']:
            img1 = self.cameras[0].get_foreground_mask(support_frames[0])
            img2 = self.cameras[1].get_foreground_mask(support_frames[1])
            img3 = self.cameras[2].get_foreground_mask(support_frames[2])
            img4 = self.cameras[3].get_foreground_mask(support_frames[3])
            # Concatenate images horizontally
            img_hor1 = np.hstack((img1, img2))
            img_hor2 = np.hstack((img3, img4))
            cv.imshow('Foreground Masks', np.vstack((img_hor1, img_hor2)))
            cv.waitKey(0)
            cv.destroyAllWindows()

        # Initialize a 3D array to store the voxel reconstruction results
        voxel_reconstruction = np.zeros(self.config['voxel_grid'], dtype=np.uint8)
        voxel_colors = dict()

        # Iterate over each voxel in the 3D space
        for x in range(self.config['voxel_grid'][0]):
            for y in range(self.config['voxel_grid'][1]):
                for z in range(self.config['voxel_grid'][2]):
                    # Check if the voxel is visible in the foreground of all views
                    visible_in_all_views = True
                    colors = []
                    depths = []
                    for i in camera_list:
                        mask = foreground_masks[i]
                        p0, p1, depth, _ = self.lookup_table[(x, y, z, i)]
                        if 0 <= p0 < mask.shape[1] and 0 <= p1 < mask.shape[0]:
                            if mask[p1, p0] == 0:
                                visible_in_all_views = False
                                break
                            colors.append(bgr_frames[i][p1, p0])
                            depths.append(depth)
                        else:
                            visible_in_all_views = False
                            break
                    if visible_in_all_views:
                        voxel_reconstruction[x, y, z] = 1
                        voxel_colors[(x, y, z)] = colors[np.argmin(depths)]
                        if self.config['color_camera']:
                            voxel_colors[(x, y, z)] = colors[self.config['color_camera']-1]

        return voxel_reconstruction, voxel_colors
    
    def reduce_noise(self, fg_objs):
        morph_ops = []
        processed = []
        params = cv.SimpleBlobDetector_Params()
        params.filterByArea = True
        params.filterByColor = True
        params.blobColor = 0  
        num_iter = 4

        for curr_iter in range(num_iter):
            print("iteration n", curr_iter)
            if curr_iter == 0:
                processed = fg_objs
                params.minArea = 60
                detector = cv.SimpleBlobDetector_create(params)
                kernel_size = 7
                kernel = np.ones((kernel_size, kernel_size), np.uint8)
            else:
                processed = morph_ops.copy()
                morph_ops.clear()
                kerle_size = kernel_size-2
                kernel = np.ones((kernel_size, kernel_size), np.uint8)
                params.minArea = params.minArea // 3
                detector = cv.SimpleBlobDetector_create(params)
                

            print("Number of processed images:", len(processed))
            for image in processed:
                # Closing
                close = cv.morphologyEx(image, cv.MORPH_CLOSE, kernel)

                # Use blob detection to fill holes in the closed foreground object
                keypoints = detector.detect(close)
                if keypoints:
                    print(f"Number of keypoints in img: {len(keypoints)}")
                    filled_contour_image = np.zeros_like(image)
                    
                    for keypoint in keypoints:
                        cv.circle(filled_contour_image, (int(keypoint.pt[0]), int(keypoint.pt[1])), int(keypoint.size / 2), (255), thickness=cv.FILLED)
                    
                    filled_contour_image =  cv.bitwise_or(filled_contour_image, close)
                    
                    #store result
                    morph_ops.append(filled_contour_image)
                    
                # Otherwise just apply closing to the preprocessed image
                else:
                    morph_ops.append(close)
                
            print("images in morph:", len(morph_ops))
        
        print("images returned in morph:", len(morph_ops))
        return morph_ops