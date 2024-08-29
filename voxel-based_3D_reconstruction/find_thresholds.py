import cv2 as cv
import numpy as np
from tqdm import tqdm
from reconstructor_utils import process_frames_MOG, refine_mask_with_hsv_thresholding

STEP = 20


def calculate_iou(mask1, mask2):
    intersection = np.logical_and(mask1, mask2)
    union = np.logical_or(mask1, mask2)
    iou_score = np.sum(intersection) / np.sum(union)
    return iou_score

def optimize_thresholds(frames, ground_truth_masks, fgMasks, hue_range=(0, 180), sat_range=(0, 255), val_range=(0, 255), hue_window=120, sat_window=120, val_window=120, step=STEP):
    best_iou = 0
    best_thresholds = {'hue': None, 'saturation': None, 'value': None}
    best_masks = [None, None, None, None]

    for h_start in range(hue_range[0], hue_range[1], step):
        h_end = min(h_start + hue_window, hue_range[1])
        for s_start in range(sat_range[0], sat_range[1], step):
            s_end = min(s_start + sat_window, sat_range[1])
            for v_start in range(val_range[0], val_range[1], step):
                v_end = min(v_start + val_window, val_range[1])

                thresholds = {
                    'hue_threshold': (h_start, h_end),
                    'saturation_threshold': (s_start, s_end),
                    'value_threshold': (v_start, v_end)
                }

                thresholds = {
                    'hue_threshold': (h_start, h_end),
                    'saturation_threshold': (s_start, s_end),
                    'value_threshold': (v_start, v_end),
                    'kernel': (5, 5),
                    'erosion_dilation': ['dilation', 'erosion', 'dilation', 'erosion'],
                    'erosion_iterations': 1,
                    'dilation_iterations': 1
                }
                iou = 0.0
                automated_masks = []
                for i, frame in enumerate(frames):
                    automated_mask = refine_mask_with_hsv_thresholding(frame, fgMasks[i], thresholds)
                    automated_masks.append(automated_mask)
                    iou += calculate_iou(automated_mask, ground_truth_masks[i])
                iou /= len(frames)

                if iou > best_iou:
                    best_iou = iou
                    best_thresholds = thresholds
                    best_masks = automated_masks

    return best_masks, best_thresholds, best_iou

def main():
    backSubs = []
    for i in range(4):
        backSub = cv.bgsegm.createBackgroundSubtractorMOG()
        process_frames_MOG(f'data/cam{i+1}/background.avi', backSub, update_model=True, display_result=False)
        backSubs.append(backSub)

    mask_calibration_frames = [cv.imread(f'data/unpainted_frame{i+1}.jpg') for i in range(4)]
    mask_calibration_frames_painted = [cv.imread(f'data/painted_frame{i+1}.png') for i in range(4)]
    lower_blue = np.array([250, 0, 0])
    upper_blue = np.array([255, 0, 0])
    calibration_masks = [cv.inRange(mask_calibration_frames_painted[i], lower_blue, upper_blue) for i in range(4)]
    hsv_frames = [cv.cvtColor(mask_calibration_frames[i], cv.COLOR_BGR2HSV) for i in range(4)]
    fgMasks = [backSubs[i].apply(hsv_frames[i], 0) for i in range(4)]

    window_sizes = [60, 80, 100, 120, 140, 160, 180, 200, 220, 240, 255]

    best_iou = 0

    pbar = tqdm(total=7*11*11)
    for hue_window in window_sizes:
        for sat_window in window_sizes:
            for val_window in window_sizes:
                masks, thr, iou = optimize_thresholds(
                    hsv_frames, calibration_masks, fgMasks,
                    hue_window=hue_window, sat_window=sat_window, val_window=val_window, step = STEP
                )
                pbar.update(1)
                if iou > best_iou:
                    best_iou = iou
                    best_parameters = (hue_window, sat_window, val_window)
                    best_masks = masks
                    best_thresholds = thr
    pbar.close()

    print(f'Best IOU: {best_iou}')
    print(f'Best Parameters: {best_parameters}')
    print(f'Best Thresholds: {best_thresholds}')

    for best_mask in best_masks:
        cv.imshow("Best Automated Mask", best_mask)
        cv.waitKey(0)

        cv.destroyAllWindows()

if __name__ == "__main__":
    main()