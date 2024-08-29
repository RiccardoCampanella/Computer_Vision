from calibration_utils import Calibrator
import yaml
import cv2 as cv
import sys
import argparse

def main():
    # run name: run1, run2 or run3 can be passed as an argument, else all 3 runs will be calibrated
    parser = argparse.ArgumentParser(description="Extract command line arguments")
    parser.add_argument("--run", type=str, help="Name of the run")
    args = parser.parse_args()

    with open('config.yaml') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    runs = ['run1', 'run2', 'run3']
    if args.run:
        runs = [args.run]
    
    for run_name in runs:
        print(f'Calibrating {run_name}...')
        config['data_dir'] = run_name
        calibrator = Calibrator(config)
        calibrator.calibrate()
        print(f'Calibration for {run_name} complete.')
        online_img = cv.imread(config['online_image'])
        print(f'Online phase for {run_name}...')
        calibrator.online_phase(online_img)

if __name__ == "__main__":
    main()
        