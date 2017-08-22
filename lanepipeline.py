# Calibrate camera

import numpy as np
import cv2
import glob
from Line import Line

import matplotlib
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def get_calibration():
    # prepare object points
    nx = 9
    ny = 6

    imgpoints = []
    objpoints = []

    # Make a list of calibration images
    images = glob.glob('camera_cal/calibration*.jpg')

    # Prepare object points
    objp = np.zeros((ny * nx, 3), np.float32)
    objp[:,:2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)

    for fname in images:
        img = mpimg.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

        if ret == True:
            imgpoints.append(corners)
            objpoints.append(objp)

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    return mtx, dist

# Get calibration matrices
mtx, dist = get_calibration()

# Set up perspective transform coordinates
src = np.float32([[261, 44], [1039, 44], [838, 173], [470, 173]])
dst = np.float32([[261, 44], [1039, 44], [1039, 173], [261, 173]])

M = cv2.getPerspectiveTransform(src, dst)
Minv = cv2.getPerspectiveTransform(dst, src)

def hls_select(img, thresh=(0, 255)):
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    s_channel = hls[:,:,2]
    binary_output = np.zeros_like(s_channel)
    binary_output[(s_channel > thresh[0]) & (s_channel <= thresh[1])] = 1
    return binary_output

def get_binary_image(image, mtx, dist):
    hls_img = get_hls_image(image, mtx, dist)
    return abs_sobel_thresh(np.dstack((hls_img, hls_img, hls_img)))
    
def get_hls_image(image, mtx, dist):
    # Create thresholded binary image
    image_undistorted = cv2.undistort(image, mtx, dist, None, mtx)
    image_size = (image_undistorted.shape[1], image_undistorted.shape[0])
    return hls_select(image_undistorted, (60, 255))

def abs_sobel_thresh(img, orient='x', thresh_min=20, thresh_max=100):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Apply x or y gradient with the OpenCV Sobel() function
    # and take the absolute value
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0))
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1))
    # Rescale back to 8 bit integer
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    # Create a copy and apply the threshold
    binary_output = np.zeros_like(scaled_sobel)
    # Here I'm using inclusive (>=, <=) thresholds, but exclusive is ok too
    binary_output[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1

    # Return the result
    return binary_output

def get_lane_lines(img, lines):
    # Identify lane line pixels

    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    left_fit = lines[0].best_fit
    right_fit = lines[1].best_fit

    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50

    if left_fit is None or right_fit is None:
        # Take a histogram of the bottom half of the image
        histogram = np.sum(img[img.shape[0]//2:,:], axis=0)
        # Create an output image to draw on and  visualize the result
        out_img = np.dstack((img, img, img))*255
        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = np.int(histogram.shape[0]//2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        # Choose the number of sliding windows
        nwindows = 9
        # Set height of windows
        window_height = np.int(img.shape[0]//nwindows)

        # Current positions to be updated for each window
        leftx_current = leftx_base
        rightx_current = rightx_base

        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []

        # Step through the windows one by one
        for window in range(nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = img.shape[0] - (window+1)*window_height
            win_y_high = img.shape[0] - window*window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin
            # Draw the windows on the visualization image
            cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2) 
            cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2) 
            # Identify the nonzero pixels in x and y within the window
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_left_inds) > minpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:        
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

        # Concatenate the arrays of indices
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
    else:
        left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] - margin))
                          & (nonzerox < (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] + margin))) 
        right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] - margin))
                           & (nonzerox < (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] + margin)))  

                    
    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]


    if len(leftx) > lines[0].min_samp and len(rightx) > lines[1].min_samp:
        fitl = np.polyfit(lefty, leftx, 2)
        fitr = np.polyfit(righty, rightx, 2)

        lines[0].current_fit = fitl
        lines[1].current_fit = fitr
        # Fit a second order polynomial to each
        if sanity_check(lines, fitl, fitr):
            lines[0].update(fitl, leftx, lefty)
            lines[1].update(fitr, rightx, righty)
        
    return lines

def compute_x(fit, y):
    return fit[0]*(y**2) + fit[1]*y + fit[2]
    
def sanity_check(lines, fitl, fitr):
    # If we don't have a best fit yet, skip this step
    for line in lines:
        if line.best_fit is None:
            return True

    if fitl is None or fitr is None:
        return False

    # Check for parallel - quadratic curve, can check two points
    # Get the y values from global src used in perspective transform
    global src
    yb = src[0,1]
    yt = src[2,1]

    left_bottom_x_best = compute_x(lines[0].best_fit, yb)
    left_top_x_best = compute_x(lines[0].best_fit, yt)
    right_bottom_x_best = compute_x(lines[1].best_fit, yb)
    right_top_x_best = compute_x(lines[1].best_fit, yt)

    left_bottom_x_fit = compute_x(fitl, yb)
    left_top_x_fit = compute_x(fitl, yt)
    right_bottom_x_fit = compute_x(fitr, yb)
    right_top_x_fit = compute_x(fitr, yt)

    if left_bottom_x_fit < 0 or left_bottom_x_fit > 1280 or \
       left_bottom_x_fit < 0 or left_bottom_x_fit > 1280 or \
       left_bottom_x_fit < 0 or left_bottom_x_fit > 1280 or \
       left_bottom_x_fit < 0 or left_bottom_x_fit > 1280:
        return False
    
    bottom_diff_best = left_bottom_x_best - right_bottom_x_best
    top_diff_best = left_top_x_best - right_top_x_best

    bottom_diff_fit = left_bottom_x_fit - right_bottom_x_fit
    top_diff_fit = left_top_x_fit - right_top_x_fit

    par_rat_fit = np.abs((bottom_diff_fit - top_diff_fit)/bottom_diff_fit)
    bot_rat = np.abs((bottom_diff_best - bottom_diff_fit)/bottom_diff_best)
    top_rat = np.abs((top_diff_best - top_diff_fit)/top_diff_fit)

    return check_rat(par_rat_fit) and check_rat(bot_rat) and check_rat(top_rat) and bottom_diff_fit < -500 and top_diff_fit < -500

def check_rat(rat, min_rat=0.0, max_rat=0.25):
    return rat > min_rat and rat < max_rat

def measure_curvature(ploty, lines):
    # Calculate radius of curvature
    y_eval = np.max(ploty)
    left_fit = lines[0].best_fit
    right_fit = lines[1].best_fit
    left_curverad = ((1 + (2*left_fit[0]*y_eval + left_fit[1])**2)**1.5) / np.absolute(2*left_fit[0])
    right_curverad = ((1 + (2*right_fit[0]*y_eval + right_fit[1])**2)**1.5) / np.absolute(2*right_fit[0])
    return left_curverad, right_curve_rad

def measure_curvature_m(img, lines):
    ploty = np.linspace(0, img.shape[0]-1, img.shape[0])
    
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension

    y_eval = np.max(ploty)

    leftx, lefty = lines[0].allx, lines[0].ally
    rightx, righty = lines[1].allx, lines[1].ally
    
    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)
    # Calculate the new radii of curvature
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    # Now our radius of curvature is in meters
    lines[0].radius_of_curvature = left_curverad
    lines[1].radius_of_curvature = right_curverad

    return lines

def add_text(img, lines):
    cv2.putText(img, "L RoC: {:5.0f} m".format(lines[0].radius_of_curvature), (0, 60), cv2.FONT_HERSHEY_PLAIN, 3, (255,255,255))
    cv2.putText(img, "R RoC: {:5.0f} m".format(lines[1].radius_of_curvature), (0, 110), cv2.FONT_HERSHEY_PLAIN, 3, (255,255,255))
    cv2.putText(img, "Position: {:3.2f}".format(calculate_position(lines)), (0, 160), cv2.FONT_HERSHEY_PLAIN, 3, (255,255,255))
    return img

def calculate_position(lines, center_pixel_x=640, max_pixel_y=720):
    # Calculate position of vehicle with respect to center
    left_fit = lines[0].best_fit
    right_fit = lines[1].best_fit
    left_x = left_fit[0]*(max_pixel_y**2) + left_fit[1]*max_pixel_y + left_fit[2]
    right_x = right_fit[0]*(max_pixel_y**2) + right_fit[1]*max_pixel_y + right_fit[2]
    center_lane_x = ( left_x + right_x ) // 2
    distance = (center_lane_x - center_pixel_x) * 3.7 / 700
    return distance

def plot_to_road(img, lines):
    # Plot result back to road
    left_fit = lines[0].best_fit
    right_fit = lines[1].best_fit
    #    print("Image is {}, left fit = {}, right fit = {}".format(img.shape, left_fit, right_fit))
    ploty = np.linspace(0, img.shape[0]-1, img.shape[0])
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    # Create an image to draw the lines on
    color_warp = np.zeros_like(img).astype(np.uint8)

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp_upside_down = cv2.warpPerspective(color_warp, Minv, (img.shape[1], img.shape[0]))
    newwarp = cv2.flip(newwarp_upside_down, 0)
    
    # Combine the result with the original image
    result = cv2.addWeighted(img, 1, newwarp, 0.3, 0)
    return result

# Process video 
from moviepy.editor import VideoFileClip

line_left = Line()
line_left.min_samp = 5000
line_right = Line()
line_right.min_samp = 1000
lines = [line_left, line_right]

def lines_string():
    yb = src[0,1]
    yt = src[2,1]
    lb = compute_x(lines[0].best_fit, yb)
    lt = compute_x(lines[0].best_fit, yt)
    rb = compute_x(lines[1].best_fit, yb)
    rt = compute_x(lines[1].best_fit, yt)
    return "{:4.0f}, {:4.0f}, {:4.0f}, {:4.0f}".format(lb, rb, lt, rt)

def process_image(image):
    global lines
    bin_img = get_binary_image(cv2.flip(image, 0), mtx, dist)
    pt_bin_img = cv2.warpPerspective(bin_img, M, (bin_img.shape[1], bin_img.shape[0]), flags = cv2.INTER_LINEAR)
    lines = get_lane_lines(pt_bin_img, lines)
    result = plot_to_road(image, lines)
    lines = measure_curvature_m(result, lines)
    final_image = add_text(result, lines)
    return final_image

clip_output = 'output_images/project_output.mp4'

clip1 = VideoFileClip("project_video.mp4")
laned_clip = clip1.fl_image(process_image)
laned_clip.write_videofile(clip_output, audio=False)
