import cv2
import cv2.aruco as aruco
import numpy as np
import pyrealsense2 as rs

# Defines the path to save the calibration file and the dictonary used
name = "realsense_d435.npz"
dictionary = aruco.DICT_6X6_250

# Initialize communication with intel realsense
pipeline = rs.pipeline()
realsense_cfg = rs.config()
realsense_cfg.enable_stream(rs.stream.color, 1280, 720, rs.format.rgb8, 6)
pipeline.start(realsense_cfg)

# Check communication
print("Test dara source...")
try:
	np.asanyarray(pipeline.wait_for_frames().get_color_frame().get_data())
except:
	raise Exception("Can't get rgb frame from data source")

# Define what the calibration board looks like (same as the pdf)
board = cv2.aruco.CharucoBoard_create(4,4, .025, .0125, aruco.Dictionary_get(dictionary))
record_count = 0
# Create two arrays to store the recorded corners and ids
all_corners = []
all_ids = []

print("Start recording [1/4]")
print("1. Move the grid from calibration directory a little bit in front of the camera and press [r] to make a record (if auto record is not set to True)")
print("2. Finish this task and start calculation press [c]")
print("3. Interrupt application [ESC]")
while True:
	# Get frame from realsense and convert to grayscale image
	frames = pipeline.wait_for_frames()
	img_rgb = np.asanyarray(frames.get_color_frame().get_data())
	img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
	
	# Detect markers on the gray image
	res = aruco.detectMarkers(img_gray, aruco.getPredefinedDictionary(aruco.DICT_6X6_250))
	# Draw the detected markers
	aruco.drawDetectedMarkers(img_rgb, res[0], res[1])
	# Display the result
	cv2.imshow("AR-Example", cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
	
	key = cv2.waitKey(10)
	# If eight markers have been found, allow recording
	if len(res[0]) == 8:
		# Interpolate the corners of the markers
		res2 = aruco.interpolateCornersCharuco(res[0], res[1], img_gray, board)
		# Add the detected interpolated corners and the marker ids to the arrays if the user press [r] and the interpolation is valid
		if key == ord('r') and res2[1] is not None and res2[2] is not None and len(res2[1]) > 8:
			all_corners.append(res2[1])
			all_ids.append(res2[2])
			record_count += 1
			print("Record: " + str(record_count))
	# If [c] pressed, start the calculation
	if key == ord('c'):
		# Close all cv2 windows
		cv2.destroyAllWindows()
		# Check if recordings have been made
		if(record_count != 0):
			print("Calculate calibration [2/4] --> Use "+str(record_count)+" records"),
			# Calculate the camera calibration
			ret, mtx, dist, rvecs, tvecs = aruco.calibrateCameraCharuco(all_corners, all_ids, board, img_gray.shape, None, None)
			print("Save calibration [3/4]")
			# Save the calibration information into a file
			np.savez_compressed(name, ret=ret, mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs)
			print("Done [4/4]")
		else:
			print("Interrupted since there are no records...")
		break
	# If [ESC] pressed, close the application
	if key == 27:
		print("Application closed without calculation")
		# Close all cv2 windows
		cv2.destroyAllWindows()
		break
