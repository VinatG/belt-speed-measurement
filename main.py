import os, sys
PATH = os.path.dirname(os.path.abspath(__file__))
# ZED SDK 4.x
import pyzed.sl as sl
sdk_version = sl.Camera().get_sdk_version() 
if sdk_version.split(".")[0] != "4":
    print("This sample is meant to be used with the SDK v4.x, Aborting.")
    exit(1)

# import other libraries as needed
import cv2

def main():
    cam = sl.Camera()
    input_type = sl.InputType()
    input_type.set_from_svo_file(os.path.join(PATH, "data", "belt_sample.svo"))
    init = sl.InitParameters(input_t=input_type, svo_real_time_mode=False)
    init.depth_mode = sl.DEPTH_MODE.PERFORMANCE 
    status = cam.open(init)
    if status != sl.ERROR_CODE.SUCCESS: #Ensure the camera opened succesfully 
        print("Camera Open", status, "Exit program.")
        exit(1)
    
    runtime = sl.RuntimeParameters()
    image = sl.Mat()
    depth = sl.Mat()
    timestamp = sl.Timestamp()
    
    # You may need this to get pixel coordinates from depth coordinates
    camera_parameters = cam.get_camera_information().camera_configuration.calibration_parameters.left_cam
    print("Camera Parameters: ", camera_parameters.fx, camera_parameters.fy, camera_parameters.cx, camera_parameters.cy)

    key = ''
    fhand = open('results.csv', 'w')
    while key != 113:
        err = cam.grab(runtime)
        if err == sl.ERROR_CODE.END_OF_SVOFILE_REACHED:
            print("End of SVO reached")
            break
        elif err != sl.ERROR_CODE.SUCCESS:
            print("Error grabbing frame: ", err)
            break
        cam.retrieve_image(image, sl.VIEW.LEFT)
        cam.retrieve_measure(depth, sl.MEASURE.DEPTH)
        timestamp = cam.get_timestamp(sl.TIME_REFERENCE.IMAGE)

        # TODO implement your code here
        speed = 0 # millimeters per second
        print("Timestamp: ", timestamp.get_milliseconds(), "Speed: ", speed)
        fhand.write(str(timestamp.get_milliseconds()) + "," + str(speed) + "\n")
        pass

        # Optional: you can display the image with below code
        # cv2.imshow("Image", image.get_data())
        # key = cv2.waitKey(5)

    cam.close()

if __name__ == "__main__":
    main()

            