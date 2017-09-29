## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

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

[image1]: ./report/binary_s.png "Binary S"
[image2]: ./report/binary_gradient_mag.png "Binary Gradient Mag"
[image3]: ./report/binary_gradient_dir.png "Binary Gradient Dir"
[image4]: ./report/binary_projected.png "Warp Example"
[image5]: ./report/finding_lines.png "Fit Visual"
[image6]: ./report/result.png "Output"
[image7]: ./report/calibration.png "Calibration"
[image8]: ./report/test2.jpg "Test2 Image"
[image9]: ./report/binary_combined.png "Binary combined"
[image10]: ./report/video_thumbnail.png "Video Thumbnail"
[video1]: ./project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README


### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the first code cell of the IPython notebook located in "./examples/example.ipynb" (or in lines # through # of the file called `some_file.py`).  

I start by getting the coefficients to remove lens distortion. I use the provided calibration images and openCV function findChessboardCorners to extract corners from the Chessboard pattern.

These are the object points, while the desination points are a rectangular grid of 9x6. The result can be seen in the following image:

![alt text][image7]

The straight line in the chessboard become straight lines in the image too.

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image8]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

First, I apply an HLS colorspace conversion and keep only the S channel. I apply a threshold to this channel to create the following binary image

![alt text][image1]

Then, I apply a grayscale conversion to the original image, and create binary images for gradient magnitude and gradient direction.

![alt text][image2]
![alt text][image3]

Finally, I merge all the binary images together using an AND operation.

![alt text][image9]

The end result looks good for the project images, although it has difficulty generalizing. A better approach might be to not use binary images until the end, to keep more information of the image.

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform uses the OpenCV functions getPerspectiveTransform to prepare the transform and the function warpPerspective to apply it. It can be found in the line 80 of the second code cell in Jupyter Notebook.


```python
src = np.float32([src_top_left,src_top_right,src_bottom_left,src_bottom_right]) 
    dst = np.float32([dst_top_left,dst_top_right,dst_bottom_left,dst_bottom_right])  

    M = cv2.getPerspectiveTransform(src, dst)

    img_size = (img.shape[1],img.shape[0])
    warped = cv2.warpPerspective(combined, M, img_size, flags=cv2.INTER_LINEAR)
```
The source and destination points are the following

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 540, 460      | 0, 0          | 
| 740, 460      | 1280, 0       |
| 0, 720        | 0, 720        |
| 1280, 720     | 1280, 720     |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

After this I applied the moving histogram approach described in Udacity's example. This starts at line 86. The method can be tuned with different starting points, number of boxes, etc.
I fit a polinomial and draw it on the image using the polyfit function in numpy

![alt text][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

The radius of curvature is calculated using the coefficients of the polinomial at the beginning of the curve. This happens in line 209

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I then painted the area between the lines with the fillPoly function.
I overlayed the combined binary image that is used to detect the points. All of this is done in the projected space and is then reprojected using the warpPerspective function with the inverse transform parameters.
The resulting image:

![alt text][image6]

---

### Pipeline (video)

#### Watch the resulting video here

Watch the result video here:

[![alt text][image10]](https://www.youtube.com/watch?v=2zLzhUYHkxk)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Going back to conventional computer vision as opposed to machine learning has made me aware of many of the advantages of deep learning.
Using this traditional approach it is very easy to end up overfitting for the test data, even if we do it manually. It is also very difficult to continue improving your algorithm after doing the basic tasks.

In general though I think the algorithm works well and can be used for lane keeping.

Things that could improve are:
-Using binary images only at the end, keeping image information until the last merge and using multiplications instead.
-Using hough lines
