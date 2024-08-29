import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import time

class Calibrator:
    def __init__(self, config):
        self.calibration_images = []
        self.config = config
        for fname in os.listdir(config['data_dir']):
            img = cv.imread(config['data_dir']+'/'+fname)
            self.calibration_images.append(img)
        
        if config['include_images']:
            for fname in config['include_images']:
                img = cv.imread(config['data_dir']+'/'+fname)
                self.calibration_images.append(img)

    def resize_image(self, img):
    # resize to display the image correctly on the laptop
        resize_factor = 0.25  # You can adjust this factor as needed
        new_width = int(img.shape[1] * resize_factor)
        new_height = int(img.shape[0] * resize_factor)
        return cv.resize(img, (new_width, new_height))
    
    def set_window_with_scaling(self, img):
        scaling_factor = 0.25
        window_width = int(img.shape[1] * scaling_factor)
        window_height = int(img.shape[0] * scaling_factor)
        cv.namedWindow('image', cv.WINDOW_NORMAL)
        cv.resizeWindow('image', window_width, window_height)
 
    def click_event(self, event, x, y, flags, params):
        if event == cv.EVENT_LBUTTONDOWN:
            # If the left mouse button is clicked, record the position
            self.corners.append((x, y))

            # Draw a circle at the position and display the coordinates
            cv.circle(self.cur_img, (x, y), 5, (0, 255, 0), -1)
            self.set_window_with_scaling(self.cur_img)
            cv.imshow('image', self.cur_img)
    
    def get_corners(self, img):
        self.corners = []
        self.cur_img = img
        # Create a window and set the callback function
        self.set_window_with_scaling(img)
        cv.imshow('image', img)
        cv.setMouseCallback('image', self.click_event)

        # Wait for the user to click on four corners
        while len(self.corners) < 4:
            if cv.waitKey(1) & 0xFF == ord('q'):
                break

        # Close the window
        cv.destroyAllWindows()

        # Return the corners
        return self.corners
    
    def linear_interpolation(self, corners):
        # corners must be ordered [top-left, top-right, bottom-right, bottom-left]!

        tl, tr, br, bl = [np.array(x) for x in corners]

        # Number of inner corners
        nx, ny = self.config['grid']

        # Prepare an array to hold the interpolated inner corner positions
        interpolated_corners = np.zeros((ny * nx, 2), dtype=np.float32)

        for y in range(ny):
            for x in range(nx):
                # Calculate the interpolation weights
                fx = x / (nx - 1)
                fy = y / (ny - 1)

                # Interpolate along the top and bottom edges
                top_edge = tl + (tr - tl) * fx
                bottom_edge = bl + (br - bl) * fx

                # Interpolate between the top and bottom edges to get the final position
                interpolated_point = top_edge + (bottom_edge - top_edge) * fy

                # Assign the interpolated point to the array
                interpolated_corners[y * nx + x] = interpolated_point

        return interpolated_corners
    

    def perspective_interpolation(self, corners):
        # corners must be ordered [top-left, top-right, bottom-right, bottom-left]!

        # Define the dimensions of the rectified chessboard image
        scale_factor = self.config['scale_factor']
        # Adjust width and height to account for the "extension" by one square on each side since user clicks inner corners
        width = (self.config['grid'][0] + 1) * scale_factor
        height = (self.config['grid'][1] + 1) * scale_factor

        # Source points from the manually clicked outermost inner corners
        src = np.array(corners, dtype="float32")

        # Adjust destination points to "extend" the chessboard by one square in each direction
        dst = np.array([
            [scale_factor, scale_factor],  # Top-left
            [width - scale_factor - 1, scale_factor],  # Top-right
            [width - scale_factor - 1, height - scale_factor - 1],  # Bottom-right
            [scale_factor, height - scale_factor - 1]],  # Bottom-left
            dtype="float32")

        # Calculate the perspective transform matrix
        M = cv.getPerspectiveTransform(src, dst)

        # Interpolate the positions of all inner corners in the rectified image
        interpolated_corners = []
        for y in range(1, self.config['grid'][1] + 1):
            for x in range(1, self.config['grid'][0] + 1):
                point = (x * scale_factor, y * scale_factor)
                interpolated_corners.append(point)

        # Map these points back to the original image's perspective
        M_inv = cv.invert(M)[1]  # Inverse of the transformation matrix
        interpolated_corners = np.array(interpolated_corners, dtype="float32").reshape(-1, 1, 2)
        original_perspective_points = cv.perspectiveTransform(interpolated_corners, M_inv)

        # Convert points back to the original shape
        original_perspective_points = original_perspective_points.reshape(-1, 2)

        return original_perspective_points


    def calibrate(self):
        objpoints = []  # 3d point in real world space
        imgpoints = []  # 2d points in image plane.

        # Assuming square_size is known and is the same for all squares
        square_size = self.config['square_size']  # replace with your actual square size in consistent units (e.g., meters)
        objp = np.zeros((self.config['grid'][0] * self.config['grid'][1], 3), np.float32)
        objp[:,:2] = np.mgrid[0:self.config['grid'][0], 0:self.config['grid'][1]].T.reshape(-1, 2) * square_size
        channels = None

        print('Calibrating...')
        for img in tqdm(self.calibration_images):
            print('Processing image', img.shape)
            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            channels = gray.shape[::-1]
            ret, corners = cv.findChessboardCorners(gray, self.config['grid'], None)
            if ret:
                print('Chessboard Found!')
            if not ret:
                print('Chessboard Not found, select corners maunally:')
                outer_corners = self.get_corners(img)
                if self.config['interpolation'] == 'perspective':
                    corners = self.perspective_interpolation(outer_corners)
                elif self.config['interpolation'] == 'linear':
                    corners = self.linear_interpolation(outer_corners)
            
            criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.01)
            corners_subpix = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            objpoints.append(objp)
            imgpoints.append(corners_subpix)

            if self.config['display_calibration']:
                # Draw and display the corners
                cv.drawChessboardCorners(img, self.config['grid'], corners_subpix, ret)
                self.set_window_with_scaling(img)
                cv.imshow('image', img)
                if self.config['display_timer']:
                    cv.waitKey(self.config['display_timer'])
                else:
                    while True:
                        if cv.waitKey(1) & 0xFF == 13:
                            break
                cv.destroyAllWindows()

        self.ret, self.mtx, self.dist, self.rvecs, self.tvecs = cv.calibrateCamera(objpoints, imgpoints, channels, None, None)
        self.objp = objp
    

    def online_phase(self, img):
        # take a new image and draw the world 3D axes (XYZ) with the origin at the center of the world coordinates, using the estimated camera parameters
        axis = np.float32([[3,0,0], [0,3,0], [0,0,-3]]).reshape(-1,3)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        ret, corners = cv.findChessboardCorners(gray, self.config['grid'], None)
        if ret:
            print('Chessboard Found!')
        if not ret:
            print('Chessboard Not found, select corners maunally:')
            outer_corners = self.get_corners(img)
            if self.config['interpolation'] == 'perspective':
                corners = self.perspective_interpolation(outer_corners)
            elif self.config['interpolation'] == 'linear':
                corners = self.linear_interpolation(outer_corners)
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.01)
        corners_subpix = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

        retval, rvec, tvec = cv.solvePnP(self.objp, corners_subpix, self.mtx, self.dist)
        cube_objpoints = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [1, 1, 0],
            [0, 1, 0],
            [0, 0, -1],
            [1, 0, -1],
            [1, 1, -1],
            [0, 1, -1]
        ], dtype=np.float32)*self.config['square_size']
        cube_imgpoints = cv.projectPoints(cube_objpoints, rvec, tvec, self.mtx, self.dist)[0]
        print(type(cube_imgpoints[0].ravel()))
        cube_imgpoints = np.int32([p.ravel() for p in cube_imgpoints])

        # Draw bottom face
        for i, j in zip(range(4), range(1, 5)):
            cv.line(img, tuple(cube_imgpoints[i]), tuple(cube_imgpoints[j % 4]), (0, 255, 0), 3)

        # Draw top face
        for i, j in zip(range(4, 8), range(5, 9)):
            cv.line(img, tuple(cube_imgpoints[i]), tuple(cube_imgpoints[j % 4 + 4]), (0, 255, 0), 3)

        # Draw sides
        for i, j in zip(range(4), range(4, 8)):
            cv.line(img, tuple(cube_imgpoints[i]), tuple(cube_imgpoints[j]), (0, 255, 0), 3)

        # Show the image
        self.set_window_with_scaling(img)
        cv.imshow('image', img)
        while True:
            if cv.waitKey(1) & 0xFF == 13:
                break
        cv.destroyAllWindows()


    def test_interpolation(self, img):
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        outer_corners = self.get_corners(img)
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.01)
        if self.config['interpolation'] == 'perspective':
            corners = self.perspective_interpolation(outer_corners)
        elif self.config['interpolation'] == 'linear':
            corners = self.linear_interpolation(outer_corners)
        corners_subpix = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        print('outer corners:', outer_corners)
        print('inner corners:', corners_subpix)

        cv.drawChessboardCorners(img, self.config['grid'], corners_subpix, True)
        self.set_window_with_scaling(img)
        cv.imshow('image', img)
        while True:
            if cv.waitKey(1) & 0xFF == 13:
                break
        cv.destroyAllWindows()
        

    def calibrate_frame(self, objp, objpoints, imgpoints, criteria, project, frame):
        ret, corners = cv.findChessboardCorners(frame, self.config['grid'], None)

        if ret:
            #corners = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            objpoints.append(objp)
            imgpoints.append(corners)
            project = True
            #print(objp, corners)
        if not ret:
            #print("Chessboard not found")
            pass

        cv.drawChessboardCorners(frame, self.config['grid'], corners, ret)
        cv.imshow('Webcam Chessboard', frame)
        
        return objpoints, imgpoints, corners, project


    def online_phase_frame(self, objp, cube_objpoints, criteria, frame):
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        ret, corners = cv.findChessboardCorners(gray, self.config['grid'], None)
        #corners_subpix = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

        retval, rvec, tvec = cv.solvePnP(self.objp, corners, self.mtx, self.dist)

        cube_imgpoints = cv.projectPoints(cube_objpoints, rvec, tvec, self.mtx, self.dist)[0]
        cube_imgpoints = np.int32([p.ravel() for p in cube_imgpoints])

        # Draw bottom face
        for i, j in zip(range(4), range(1, 5)):
            cv.line(frame, tuple(cube_imgpoints[i]), tuple(cube_imgpoints[j % 4]), (0, 255, 0), 3)

        # Draw top face
        for i, j in zip(range(4, 8), range(5, 9)):
            cv.line(frame, tuple(cube_imgpoints[i]), tuple(cube_imgpoints[j % 4 + 4]), (0, 255, 0), 3)

        # Draw sides
        for i, j in zip(range(4), range(4, 8)):
            cv.line(frame, tuple(cube_imgpoints[i]), tuple(cube_imgpoints[j]), (0, 255, 0), 3)

        # Show the image
        cv.imshow('Webcam Cube Projection', frame)
            

    def online_phase_in_real_time(self, offline_phase_in_real_time=False):
        cap = cv.VideoCapture(0) 
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.01)
        frame_count = 0

        # Free choice implementation for the offline phase in real time
        if offline_phase_in_real_time == True:
            print("OFFLINE PHASE")
            objpoints = [] 
            imgpoints = []  

            square_size = self.config['square_size']  
            objp = np.zeros((self.config['grid'][0] * self.config['grid'][1], 3), np.float32)
            objp[:,:2] = np.mgrid[0:self.config['grid'][0], 0:self.config['grid'][1]].T.reshape(-1, 2) * square_size
            
            corners = None
            channels = None
            project = False

            task_start_time = start_time = time.time()
            elapsed_time = 0
            duration = 20  # Set the duration in seconds

            while True:
                # Read a frame from the webcam
                ret, frame = cap.read()

                if ret:
                    skip_frames = 10
                    frame_count += 1
                    if frame_count % skip_frames == 0:
                        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
                        channels = gray.shape[::-1]
                        objpoints, imgpoints, corners, project = self.calibrate_frame(objp, objpoints, imgpoints, criteria, project, frame)
                    
                    cv.waitKey(1)

                    # Calculate progress percentage
                    elapsed_time = time.time() - start_time
                    progress_percentage = (elapsed_time / duration) * 100

                else: 
                    print("Invalid frame!")
                    print("Can't receive frame. Retrying ...")
                    cap.release()
                    cap = cv.VideoCapture(0)                                                                              

                # Update progress display
                print(f"Progress: {progress_percentage:.2f}%")

                if  elapsed_time > duration or cv.waitKey(1) & 0xFF == ord('q'):
                    cap.release()
                    cv.destroyAllWindows()
                    break

            cap.release()
            cv.destroyAllWindows()

            print("Camera Calibration")
            self.ret, self.mtx, self.dist, self.rvecs, self.tvecs = cv.calibrateCamera(objpoints, imgpoints, channels, None, None)
            self.objp = objp
            print(f"Offline ask (real-time) completed in {time.time() - task_start_time:.2f} seconds.")

        # Choice Task implementation for the online phase in real time
        print("ONLINE PHASE")

        # projection of the cube is allowed when: 
        # 1) offline phase is not in real time 2) offline phase is in real-time and corners are found at least for one frame 
        if offline_phase_in_real_time == False:
            project = True 

        axis = np.float32([[3,0,0], [0,3,0], [0,0,-3]]).reshape(-1,3)
        cube_objpoints = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [1, 1, 0],
            [0, 1, 0],
            [0, 0, -1],
            [1, 0, -1],
            [1, 1, -1],
            [0, 1, -1]
        ], dtype=np.float32)*self.config['square_size']
    
        print("Projection")
        while project:
            ret, frame = cap.read()

            if ret:
                skip_frames = 10
                frame_count += 1
                if frame_count % skip_frames == 0:
                    pass
                self.online_phase_frame(objp, cube_objpoints, criteria, frame)
            else: 
                print("Invalid frame!")
                print("Can't receive frame. Retrying ...")
                cap.release()
                cap = cv.VideoCapture(0)   

            cv.waitKey(1)

            if  cv.waitKey(1) & 0xFF == ord('q'):
                cap.release()
                cv.destroyAllWindows()
                break

        cap.release()
        cv.destroyAllWindows()


if __name__ == '__main__':
    config = {
        'data_dir': 'Run1',
        'grid': (6, 9),
        'square_size': 3.4,
        'img_limit': 25,
        'include_images': None,
        'display_calibration': True,
        'display_timer': 1000,
        'online_image': 'test.jpeg',
        'interpolation': 'perspective',
        'scale_factor': 25
    }
    
    calibrator = Calibrator(config)
    
    calibrator.calibrate()

    config['data_dir']='Run2'
    calibrator.calibration_images.clear()
    calibrator.calibrate()

    config['data_dir']='Run3'
    calibrator.calibration_images.clear()
    calibrator.calibrate()

    test_img = cv.imread(config['online_image'])
    calibrator.test_interpolation(test_img)

    config['online_image'] = 'online.jpg'
    calibrator.online_phase(test_img)

    calibrator.online_phase_in_real_time()
    
    calibrator.online_phase_in_real_time(offline_phase_in_real_time=True)
