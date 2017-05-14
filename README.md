## Advanced Lane Finding Project

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

[image1]: ./writeup_img/undistorted.png "Undistorted"
[image2]: ./test_images/test5.jpg "Road Transformed"
[image3]: ./writeup_img/test5_combined_threshold.png "Binary Example"
[image4]: ./writeup_img/src_dst_points_drawn.png "Warp Example"
[image5]: ./writeup_img/color_fit_lines.png "Fit Visual"
[image6]: ./writeup_img/example_output.png "Output"
[image7]: ./writeup_img/histogram.png



---

### 1. Camera calibration

The code for this step is contained in the third code cell of the IPython notebook located in "./Advanced_Lane_Lines.ipynb" 

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:

![alt text][image2]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image (definition in function `combined_threshold()` in 7th code cell). 
* R_channel from RGB color image (R_binary)
* S_channel from HLS color image (S_binary)
* sobel threshold in x direction (gradx)
* sobel threshold in y direction (grady)
* sobel threshold in both x and y direction (mag_binary)
* direction of the sobel gradient (dir_binary)

and then a combined those binary inputs like this:
`(((gradx == 1) | (grady == 1)) & (mag_binary == 1) | ((S_binary == 1) & (R_binary == 1))) & (dir_binary == 1)`

Threshold minimum was set to 100 and threshold maximum was set to 255. For direction sobel threshold was set to (0.7,1.3). Here's an example of my output for this step.  
![alt text][image3]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `warp_img()`, which appears in the 6th code cell of the IPython notebook.  The `warp_img()` function takes as inputs an image (`img`). Source (`src`) and destination (`dst`) points were hardcoded and used for compute the perspective transform and the inverse perspective transform in the setup part of IPython notebook (2nd code cell).  I chose the hardcode the source and destination points:


| Source        | Destination   | 
|:-------------:|:-------------:| 
| 200, 720      | 300, 720      | 
| 595, 450      | 300, 0        |
| 685, 450      | 980, 0        |
| 1100, 720     | 980, 720      |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?
After applying calibration, thresholding, and a perspective transform to a road image, I have a binary image where the lane lines stand out clearly. However, I still need to decide explicitly which pixels are part of the lines and which belong to the left line and which belong to the right line.

I first take a histogram along all the columns in the lower half of the image like this:
```python
import numpy as np
histogram = np.sum(img[img.shape[0]//2:,:], axis=0)
plt.plot(histogram)
```

The result looks like this:

![alt text][image7]

With this histogram I am adding up the pixel values along each column in the image. In my thresholded binary image, pixels are either 0 or 1, so the two most prominent peaks in this histogram will be good indicators of the x-position of the base of the lane lines. I can use that as a starting point for where to search for the lines. From that point, I can use a sliding window, placed around the line centers, to find and follow the lines up to the top of the frame. Sliding window is implemented in 8th code cell w `function sliding_window()`. 

Then I fit my lane lines with a 2nd order polynomial like this:

![alt text][image5]

Now I know where the lines I you have a fit! In the next frame of video I don't need to do a blind search again, but instead I can just search in a margin around the previous line position like in definition `already_detected()` in 9th code cell.


#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.
The radius of curvature was measured like described in this [awesome tutorial](http://www.intmath.com/applications-differentiation/8-radius-curvature.php).
Converting pixel space to real world space involves measuring how long and wide the section of lane is that we're projecting in our warped image. We could do this in detail by measuring out the physical lane in the field of view of the camera, but for this project, I assume that the lane is about 30 meters long and 3.7 meters wide. 

I did this in 11th code cell in my IPython notebook in fuction `measure_curvature_and_offset()`

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I use `Line()` class to decide if I calculate sliding window from scratch or use previous one. Then I plot the average of 5 last polynomial coefficients. Here is an example of my result on a test image:

![alt text][image6]

---

### Pipeline (video)

[![link to my video result](https://img.youtube.com/vi/1EBQkZksBJk/0.jpg)](https://youtu.be/1EBQkZksBJk)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust? 

##### Sanity Check
I implemented check if the lane lines detection makes sense. To confirm that your detected lane lines are real, I consider:

* Checking that they have similar curvature with margin of 200m.
* Checking that they are separated by approximately the right distance horizontally with margin of 0.5m


##### Look-Ahead Filter
When I fit a polynomial, then for each y position, I have an x position that represents the lane center from the last frame. Then I search for the new line within +/- some margin around the old line center.


##### Reset
If my sanity checks (done once on ten frames) return bad detection flag for three times in a row, I start searching from scratch using a histogram and sliding window.

##### Smoothing
Each time I get a new high-confidence measurement, I append it to the list of recent measurements and then take an average over 5 past measurements to obtain the lane position I want to draw onto the image.