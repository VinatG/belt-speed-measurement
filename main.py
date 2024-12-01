"""
Python script to determine the conveyor belt's speed using motion tracking of the rocks.

CLI Arguements:
    1. --number-of-features : Number of features to be used for the detector
    2. --fps : FPS of the input video
    3. --const : Lowe's ratio test threshold
    4. --visualize : If set, the results are visualized

Usage:

python main.py --number-of-features 500 --fps 60 --const 0.7 --visualize --output-csv-file-path speeds.csv
"""
import os
import cv2
import argparse
import numpy as np
import pyzed.sl as sl
import pandas as pd

PATH = os.path.dirname(os.path.abspath(__file__))
sdk_version = sl.Camera().get_sdk_version() 
if sdk_version.split(".")[0] != "4":
    print("This sample is meant to be used with the SDK v4.x, Aborting.")
    exit(1)

# Left view region of interest belt polygon 
LEFT_BELT_POLYGON = np.array([[415, 720.00], [462, 0.00], [753, 0.00], [890, 720.00]], np.int32)


# Kalman Filter Class
class KalmanFilter:
    def __init__(self, process_var = 1e-2, measurement_var = 1e-1, initial_value = 0):
        self.process_var = process_var
        self.measurement_var = measurement_var
        self.estimate = initial_value
        self.error_cov = 1

    def update(self, measurement):
        predicted_estimate = self.estimate
        predicted_error_cov = self.error_cov + self.process_var
        kalman_gain = predicted_error_cov / (predicted_error_cov + self.measurement_var)
        self.estimate = predicted_estimate + kalman_gain * (measurement - predicted_estimate)
        self.error_cov = (1 - kalman_gain) * predicted_error_cov
        return self.estimate

# Function to extract the region of interest from the frame using the polygon
def mask_polygon(frame, polygon):
    mask = np.zeros_like(frame, dtype = np.uint8)
    cv2.fillPoly(mask, [polygon], 255)
    return cv2.bitwise_and(frame, mask)

# Function to perform feature detection and matching on the previous and current frame
def find_features(args, feature_extractor, prev_gray_frame, curr_gray_frame, polygon):
    # Apply polygon mask on the previous and current frame to extract region of interests
    prev_gray_frame = mask_polygon(prev_gray_frame, polygon)
    curr_gray_frame = mask_polygon(curr_gray_frame, polygon)

    # Detect features using the SIFT feature extractor
    keypoints_prev, descriptors_prev = feature_extractor.detectAndCompute(prev_gray_frame, None)
    keypoints_curr, descriptors_curr = feature_extractor.detectAndCompute(curr_gray_frame, None)

    # Using FLANN for feature matching
    index_params = dict(algorithm = 0, trees = 5)
    search_params = dict(checks = 50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(descriptors_prev, descriptors_curr, k = 2)

    # Applying Lowe's ratio test
    good_matches = [m for m, n in matches if m.distance < args.const * n.distance]
    return keypoints_prev, keypoints_curr, good_matches

# Map Image Coordinates to Real-World Measurements
def image_to_world(x, y, z, fx, fy, cx, cy):
    X = z * (x - cx) / fx
    Y = z * (y - cy) / fy
    return X, Y, z

# Calculate Speed given the coordinates using euclidean distance
def calculate_speed(world_coordinates, time_delta):
    total_distance = 0
    count = 0
    for coord in world_coordinates:
        p1, p2 = coord
        distance = np.linalg.norm(np.array(p1) - np.array(p2))
        total_distance += distance
        count += 1
    avg_distance = total_distance / count if count > 0 else 0
    return avg_distance / time_delta if time_delta > 0 else 0

# Visualize the frame using CV2
def visualize(frame, polygon, keypoints, speed_left = None, direction = None):
    vis_frame = cv2.polylines(frame.copy(), [polygon], isClosed = True, color = (0, 255, 0), thickness = 3)
    for kp in keypoints:
        x, y = kp.pt
        cv2.circle(vis_frame, (int(x), int(y)), radius = 3, color = (255, 0, 0), thickness = -1)
    y_offset = 30
    cv2.putText(vis_frame, f"Speed: {speed_left:.2f} mm/s", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    cv2.putText(vis_frame, f"Direction: {direction}", (10, y_offset + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    cv2.imshow("Visualization", vis_frame)
    cv2.waitKey(1)

# Function to detect the direction of the belt using feature flow patterns
def detect_belt_direction(keypoints_prev, keypoints_curr, good_matches):
    flow_vectors = []

    for match in good_matches:
        idx1 = match.queryIdx
        idx2 = match.trainIdx
        p1 = keypoints_prev[idx1].pt
        p2 = keypoints_curr[idx2].pt
        flow_vectors.append((p2[0] - p1[0], p2[1] - p1[1]))

    avg_flow_x = np.mean([vec[0] for vec in flow_vectors])
    avg_flow_y = np.mean([vec[1] for vec in flow_vectors])

    if abs(avg_flow_x) > abs(avg_flow_y):
        direction = "Right" if avg_flow_x > 0 else "Left"
    else:
        direction = "Down" if avg_flow_y > 0 else "Up"
    return direction

def main(args):
    FPS = args.fps
    
    cam = sl.Camera()
    input_type = sl.InputType()
    input_type.set_from_svo_file(os.path.join(PATH, "data", "belt_sample.svo"))
    init = sl.InitParameters(input_t=input_type, svo_real_time_mode = False)
    init.depth_mode = sl.DEPTH_MODE.PERFORMANCE
    status = cam.open(init)
    if status != sl.ERROR_CODE.SUCCESS: #Ensure the camera opened succesfully 
        print("Camera Open", status, "Exit program.")
        exit(1)

    runtime_params = sl.RuntimeParameters()
    left_image = sl.Mat()
    left_depth = sl.Mat()

    # Retrieve camera intrinsic parameters
    camera_params = cam.get_camera_information().camera_configuration.calibration_parameters.left_cam
    fx = camera_params.fx
    fy = camera_params.fy
    cx = camera_params.cx
    cy = camera_params.cy

    feature_extractor = cv2.SIFT_create(nfeatures = args.number_of_features) # Initializing the the SIFT feature detector
    kalman_filter = KalmanFilter()
    prev_left_frame = None

    # Pandas data frame to log the speeds
    speeds_data = []

    while cam.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:
        cam.retrieve_image(left_image, sl.VIEW.LEFT)
        cam.retrieve_measure(left_depth, sl.MEASURE.DEPTH)

        curr_left_frame = left_image.get_data()
        curr_left_gray = cv2.cvtColor(curr_left_frame, cv2.COLOR_BGR2GRAY)

        timestamp = cam.get_timestamp(sl.TIME_REFERENCE.IMAGE).get_milliseconds()

        if prev_left_frame is not None:
            # Match features in the left frame
            keypoints_prev, keypoints_curr, good_matches = find_features(args, feature_extractor, prev_left_frame, curr_left_gray, LEFT_BELT_POLYGON)

            actual_coordinates = [] # List to store the actual coordinates(real world coordinates)
            for match in good_matches:
                idx1 = match.queryIdx
                idx2 = match.trainIdx
                p1 = keypoints_prev[idx1].pt
                p2 = keypoints_curr[idx2].pt
                z1 = left_depth.get_value(int(p1[1]), int(p1[0]))[1]
                z2 = left_depth.get_value(int(p2[1]), int(p2[0]))[1]
                if np.isfinite(z1) and np.isfinite(z2):
                    world_p1 = image_to_world(p1[0], p1[1], z1, fx, fy, cx, cy)
                    world_p2 = image_to_world(p2[0], p2[1], z2, fx, fy, cx, cy)
                    actual_coordinates.append((world_p1, world_p2))

            time_delta = 1 / FPS  # Use FPS for consistent time delta
            direction = detect_belt_direction(keypoints_prev, keypoints_curr, good_matches)

            speed_left = calculate_speed(actual_coordinates, time_delta)
            smoothed_speed = kalman_filter.update(speed_left)

            print(f"Timestamp: {timestamp} ms, Smoothed Speed: {smoothed_speed:.2f} mm/s, Direction: {direction}")
            speeds_data.append({"Timestamp (ms)": timestamp, "Smoothed Speed (mm/s)": smoothed_speed})
            if args.visualize:
                visualize(curr_left_frame, LEFT_BELT_POLYGON, keypoints_curr, smoothed_speed, direction)

        prev_left_frame = curr_left_gray

    # Save speeds data to a CSV
    speeds_df = pd.DataFrame(speeds_data)
    speeds_df.to_csv(args.output_csv_file_path, index = False)
    print("Speeds Log saved successfully")

    cam.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--number-of-features", type = int, default = 500, help = "Number of features to be used for the feature detector")
    parser.add_argument("--fps", type = int, default = 60, help = "FPS of the input video")
    parser.add_argument("--const", type = float, default = 0.7, help = "Lowe's ratio test threshold")
    parser.add_argument("--visualize", action = "store_true", help = "If set, the results are visualized")
    parser.add_argument("--output-csv-file-path", type = str, required = True, help = "Path to the output csv file path containing the speed logs")

    args = parser.parse_args()
    main(args)
