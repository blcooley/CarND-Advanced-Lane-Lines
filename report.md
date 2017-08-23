**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output_images/calibration.png "Undistorted"
[image2]: ./output_images/road_transformed.png "Road Transformed"
[image3]: ./output_images/binary_image.png "Binary Example"
[image4]: ./output_images/warped_straight_lines.png "Warp Example"
[image5]: ./output_images/color_fit_lines.jpg "Fit Visual"
[image6]: ./output_images/example_output.png "Output"
[video1]: ./output_images/project_output.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the function `get_calibration` in the `lanepipeline.py` file on lines 14-40. 

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

Once the calibration matrix and distortion coefficients are obtained, they can be saved to undistort road images. I do this in the code at line 315 in the process_image function as the first step in the pipeline. An example of applying the undistort function on test1.jpg from the test images folder in the supplied repository is shown below.
![alt text][image2]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

The code I used for generating a binary image is located in lines 52-85.

The basic idea is to first create a thresholded binary image using the S channel in HLS space, then pass that image through a Sobel filter computing gradients in the x direction. These steps are performed in the function get_binary_image (lines 59-61).

The first step, is to call get_hls_image (line 60, which calls the function defined on lines 63-67). The function converts the image to HLS colorspace, then uses a threshold (60-255 in this case) to convert the image to a binary image based on the S channel.

Once the HLS-thresholded binary image is returned, it is passed to abs_sobel_thresh (lines 69-86), which computes gradients in the x-direction. Pixels within the threshold (20-100 in this case) are converted to 1's and other pixels are converted to 0's. This Sobel-thresholded image is returned to the pipeline.

Here's an example of my output for this step.  

![alt text][image3]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform is on lines 45-50 and line 317. Lines 45-50 set up the source and destination points for the transform and compute the warp matrix and its inverse. The warping is performed on line 317 with a call to cv2.warpPerspective, passing in the binary image computed in get_binary_image. I selected points interactively from an matplotlib.pyplot plot and ended up with theh following values:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 261, 676      | 261, 676      | 
| 1039, 676     | 1039, 676     |
| 1039, 547     | 838, 547      |
| 261, 547      | 470, 547      |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

I fit the lane lines in the function `get_lane_lines` located on lines 87-179 of `lanepipeline.py`. In this function, I first compute the histogram of pixels on the bottom half of the image (lines 104-112) to compute the base x value for the left and right sides of the image. Then, I divide the vertical space into 9 windows. Starting at the bottom, I search in a window around the base x value for nonzero pixels. If I find more than 50 pixels in the window, I recenter on the mean x value of the pixels I found. This gives me a set of x, y pixel data that should represent the lines.

Then I fit a second-order polynomial to the data (lines 166-167). I do a sanity check to make sure the lines are relatively parallel and the distance between them isn't unreasonable. The `sanity_check` function and its helper function `check_rat` are contained in lines 181-225.

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

Once I have the fitted lines, I compute the radius of curvature (in meters) using the function `measure_curvature_m` on lines 227-249. This function fits a second-order polynomial to the last set of data and converts it to curvature using the relations on lines 243-244.

I also calculate the position in the function `calculate_position` on lines 257-265. I am using 640 as the center pixel for the car and computing the left and right pixels from the best_fit curve. Finally, the pixel distance is converted to m using 3.7 meters per 700 pixels.

The curvature for each lane line and the position of the car relative to center are added to the video in the `add_text` function on lines 251-255.

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines 267 through 293 in the function `plot_to_road`. This function takes the computed fits, calculate x-values across a range of y-values matching the image height, and calls `cv2.fillPoly` to color the area between the fitted curves. The image is warped back to the driver's perspective using the inverse matrix computed earlier. Finally, the image is added to the image of the road using `cv2.addWeighted`. An example image is shown below.

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./output_images/project_output.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Probably the biggest stumbling block for me in getting a successful pipeline was selecting the source and destination points for the perspective transform. Selecing the points fairly close to the horizon tended to give terrible results. It was only when I moved the points closer to the car itself that I started getting more reliable performance.

I tried a couple of approaches to finding lines. First, I tried using the HLS color map and using the S channel to find the lines. The biggest problem I had with this method was dealing with sections of shadow, which tended to show up in the S channel. Because of the way the shadows appeared in this video, I was able to handle the shadow by passing the binary thresholded S channel image through an x-dimension Sobel operator to deal with the shadows. This worked pretty well in those sections, but it still fails my sanity check at times. The white sections of road (bridges?) are the most difficult to deal with, and on a long section of that color concrete, I think the pipeline might perform very poorly.