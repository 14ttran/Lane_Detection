# Lane Detection
OpenCV implementation process for simple lane detection.

Tested using Python 3 and data from the BDD100K dataset.
See: https://bdd-data.berkeley.edu/

# Procedure 
1. Rotate image to correct orientation
2. Convert image to grayscale
3. Run Gaussian smoothing on the full video
4. Isolate a region of interest (ROI) to include only the lanes markings directly in front of the ego vehicle
5. Canny edge detection of the ROI
6. Line detection with hough transform and filtering for vertical lines
7. Post processing and averaging on the hough lines
8. Overlaying main lane lines on original image

# Results
Current implementation ideal results:
![stay_in_yo_lane_ex](https://user-images.githubusercontent.com/16512161/55677015-3db2a500-5895-11e9-9da1-5af28a3ed536.gif)


