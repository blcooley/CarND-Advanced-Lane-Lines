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
[image3]: ./examples/binary_combo_example.jpg "Binary Example"
[image4]: ./examples/warped_straight_lines.jpg "Warp Example"
[image5]: ./examples/color_fit_lines.jpg "Fit Visual"
[image6]: ./examples/example_output.jpg "Output"
[video1]: ./project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the function get_calibration in the lane-pipeline.py file on lines 14-40. 

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

Once the calibration matrix and distortion coefficients are obtained, they can be saved to undistort road images. I do this in the code at line 327 in the process_image function as the first step in the pipeline. An example of applying the undistort function on test1.jpg from the test images folder in the supplied repository is shown below.
![alt text][image2]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

The code I used for generating a binary image is located in lines 52-86.

The basic idea is to first create a thresholded binary image using the S channel in HLS space, then pass that image through a Sobel filter computing gradients in the x direction. These steps are performed in the function get_binary_image (lines 59-61).

The first step, is to call get_hls_image (line 60, which calls the function defined on lines 63-67). The function converts the image to HLS colorspace, then uses a threshold (60-255 in this case) to convert the image to a binary image based on the S channel.

Once, the HLS-thresholded binary image is returned, it is passed to abs_sobel_thresh (lines 69-86), which computes gradients in the x-direction. Pixels within the threshold (20-100 in this case) are converted to 1's and other pixels are converted to 0's. This Sobel-thresholded image is returned to the pipeline.

Here's an example of my output for this step.  

![alt text][image3]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform is on lines 45-50 and line 339. Lines 45-50 set up the source and destination points for the transform and compute the warp matrix and its inverse. The warping is performed on line 339 with a call to cv2.warpPerspective, passing in the binary image computed in get_binary_image. I selected points interactively from an matplotlib.pyplot plot and ended up with theh following values:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 261, 44       | 261, 44       | 
| 1039, 44      | 1039, 44      |
| 1039, 173     | 838, 173      |
| 261, 173      | 470, 173      |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Then I did some other stuff and fit my lane lines with a 2nd order polynomial kinda like this:

![alt text][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines # through # in my code in `my_other_file.py`

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines # through # in my code in `yet_another_file.py` in the function `map_lane()`.  Here is an example of my result on a test image:

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./output_images/project_output.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Probably the biggest hurdle for getting a successful pipeline was selecting the source and destination points for the perspective transform. Selecing the points fairly close to the horizon tended to give terrible results. It was only when I moved the points closer to the car itself that I started getting more reliable performance.

I tried a couple of approaches to finding lines. I ended up using the HLS color map and using the S channel to find the lines. The biggest problem I had with my pipeline was dealing with sections of shadow. Typically, the S channel will pick up many pixels in the areas of shadow. Because they were relatively small, I was able to ignore them with my sanity checks. However, on a heavily shaded road this would fail miserably.

I also tried combining this with gradients by passing the binary thresholded S channel image through an x-dimension Sobel operator to deal with the shadows. This worked pretty better in those sections, but I found overall that it was more difficult to get the pipeline to perform reliably over the entire track. However, to make the pipeline more robust, I think it's certainly necessary to come up with some combination of these two detectors.