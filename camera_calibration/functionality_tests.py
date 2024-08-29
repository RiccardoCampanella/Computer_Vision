from calibration_utils import Calibrator
import yaml
import cv2 as cv
import sys
import argparse

def main():
    # scirpt to test the functionality of the calibration_utils: interpolation, manual corners refinement, and camera positions in 3D space ы
    parser = argparse.ArgumentParser(description="Extract command line arguments")

    parser.add_argument("--run", type=str, help="Name of the run")
    parser.add_argument("--cameras", action='store_true', help="Test cameras or not")
    parser.add_argument("--interpolation", action='store_true', help="Test interpolation or not")
    parser.add_argument("--refinement", action='store_true', help="Test refinement or not")

    args = parser.parse_args()
    with open('config.yaml') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    if args.run:
        config['data_dir'] = args.run
    calibrator = Calibrator(config)

    test_img = cv.imread(config['online_image'])
    if args.interpolation:
        print('Testing interpolation...')
        calibrator.test_interpolation(test_img.copy())
    if args.refinement:
        print('Testing refinement...')
        calibrator.test_corner_refinement(test_img.copy())
    
    if args.cameras:
        calibrator.calibrate()
        calibrator.online_phase(test_img)
        print('Testing cameras...')
        calibrator.display_camera_positions()

        test_img = cv.imread('right.jpeg')
        calibrator.online_phase(test_img)
        print('Testing cameras...')
        calibrator.display_camera_positions()

        test_img = cv.imread('right_fromtop.jpg')
        calibrator.online_phase(test_img)
        print('Testing cameras...')
        calibrator.display_camera_positions()
if __name__ == "__main__":
    main()