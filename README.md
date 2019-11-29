[//]: # (Image References)

[cam_cal]: ./media/cam_cal.png "Camera Calibration"
[dist_cor]: ./media/dist_correction.png "Distortion Correction"
[bin]: ./media/binary.png "Camera Calibration"
[warp]: ./media/warp.png "Warp"
[lane_blocks]: ./media/lane_blocks.png "Lane Blocks"
[area]: ./media/lane_area.png "Lane Area"
[measures]: ./media/lane_area_measures.png "Measures"
[gif_result]: ./media/output.gif "Road Lane Detection"


# **Advanced Lane Finding**

---
<p align="center">
	<img src="/media/output.gif" alt="result"
	title="result"  />
</p>

### Objectives:

1. compute camera calibration to correct for distortion in input images
2. apply color masking to input image to identify lane lines 
3. create birds-eye-view of road
4. identify lane pixels and compute confidence window
5. fit polynomial to confidence window
6. calculate turn-curvature & vehicle position on road
7. return appended video


```python
import os, glob
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

plt.rcParams['figure.figsize'] = [15,15]
plt.rcParams['image.cmap'] = 'gray'
```


```python
def compare_pictures(img1, img2, title1, title2):
    fig = plt.figure(figsize=(15,15))
    fig.add_subplot(1, 2, 1, title=title1)
    plt.imshow(img1)
    fig.add_subplot(1, 2, 2, title=title2)
    plt.imshow(img2)
    plt.show()
plt.show()
```

## 1. Camera Calibration

Camera Calibration function takes a list of chessboard images to identify and draw the chessboard corners if found.


```python
def camera_calibration(images, nx,ny):
    
    subpix = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    objpoints = [] 
    imgpoints = [] 
    images_with_corners = []
    counter = 0    
    
    objp = np.zeros((ny*nx,3), np.float32)
    objp[:,:2] = np.mgrid[0:nx, 0:ny].T.reshape(-1,2)
    
    for image in images:
        img = mpimg.imread(image)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (nx,ny), None)
        if ret == True:
            counter += 1
            corners = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),subpix)
            imgpoints.append(corners); objpoints.append(objp)
            img = cv2.drawChessboardCorners(img, (nx,ny), corners, ret)
            images_with_corners.append(img)   
    print(f'{counter}/{len(images)} corners found')
    return objpoints, imgpoints, images_with_corners
```

From the returned imagepoints and objectpoints a distortion matrix can be computed and applied to the picture


```python
def cal_undistort(img, objpoints, imgpoints):
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img.shape[1:], None, None)
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    return undist
```


```python
objpoints, imgpoints, images = camera_calibration(glob.glob('camera_cal/*jpg'), 9, 6)

chess_org = mpimg.imread(glob.glob('camera_cal/*jpg')[1])
chess_dist  = cal_undistort(images[0], objpoints, imgpoints)

compare_pictures(
    chess_org, 
    chess_dist, 'Original', 'Undistorted with Markers')
```

    17/20 corners found

![alt text][cam_cal]


## 2. Color Masking

The input image is converted into HSV Format to filter out all white and yellow colors between a given range. With the or statement a binary image is returned with all pixels that fall in between the chosen color range. 


```python
img = mpimg.imread(glob.glob('test_images/*jpg')[0])
img_undist = cal_undistort(img, objpoints, imgpoints)
img_copy = cv2.cvtColor(img_undist, cv2.COLOR_RGB2HSV)

compare_pictures(
    img, 
    img_undist, 'Original', 'Undistorted')
```

![alt text][dist_cor]


```python
def get_binary(img):
    
    LOW_WHITE = np.array([0, 0, 200], dtype=np.uint8)
    HIGH_WHITE = np.array([90,70,255], dtype=np.uint8)
    LOW_YELLOW = np.array([10,120,200], dtype=np.uint8)
    HIGH_YELLOW = np.array([30,255,255], dtype=np.uint8)

    wmask = cv2.inRange(img, LOW_WHITE, HIGH_WHITE)
    ymask = cv2.inRange(img, LOW_YELLOW, HIGH_YELLOW)
    binary = cv2.bitwise_or(ymask, wmask)
    
    return binary

binary = get_binary(img_copy)

compare_pictures(
    img, 
    binary, 'Original', 'binary')
```

![alt text][bin]


## 3. Birds-eye-view

The goal of creating a birds-eye-view is converting the image so that the lanes are as parallel as possible. Therefore an isosceles trapezoid is streched from fixed coordinates with the warp_perspective function. This function will also compress our image back to normal if source and destination are swapped out. 


```python
def warp_perspective(img, src, dst):
    M = cv2.getPerspectiveTransform(src, dst)
    warped_image = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR)
    return warped_image
```


```python
src = np.float32(
[[150,720], # bottom left
 [1130, 720], # bottom right
 [570,450], # top left
 [670, 450]]) # top right

dst = np.float32(
[[100,720],
 [1130, 720],
 [150,0],
 [1080, 0]])
```


```python
binary_warped = warp_perspective(binary, src, dst)
compare_pictures(
    binary, 
    binary_warped, 'binary', 'warped')
```

![alt text][warp]


## 4. Lane Pixels

When we start looking for lane pixels an either side its best to start from the bottom, since the top will be stretched the most and therefore fuzzy. In general we want to cut the image into smaller slices and search for the x-Value with the highest sum of pixels on both sides. We pass the image and the coordinates of the slice from the y-axis. 


```python
def get_base_nonlinear(binary_warped, y_coords):
    
    start = y_coords[0]
    end = y_coords[1]   
    
    mid_x = int(binary_warped.shape[1] / 2)
    left_base = np.argmax(np.sum(binary_warped[start:end,:mid_x],axis=0))
    right_base = np.argmax(np.sum(binary_warped[start:end,mid_x:],axis=0)) + mid_x

    return left_base, right_base
```

If we have successfully made a detection we can use the information of the last frame for the next. If it's the first detection, we won't have any information on the previous lane. 

Here I keep track of the change of the delta.  If the Delta in x for two following frames is so high, that it is unlikely a correct observation I copy the moving delta from the other lane (if detected) to get an approximate frame for the missing lane (e.g. when dotted). 

If both lanes are failed to detect, the last observation gets copied.


```python
def correct_base(left_base, right_base, last_window, window_size=75):
    
    if last_window[0] != None:
        last_left = last_window[0]
        last_right = last_window[1]
        delta_left = last_left - left_base
        delta_right = last_right - right_base
        right_found = True
        left_found = True

        # if proximity between the new and old window is not given then we assume we have trouble identifying it
        if abs(delta_left) > window_size:
            left_found = False

        if abs(delta_right) > window_size:
            right_found = False

        # if one lane is found, dubplicate the shift on the other lane
        if left_found and not right_found:
            right_base = last_right - delta_left
        elif right_found and not left_found:
            left_base = last_left - delta_right
        # if none are found take last observation
        elif not left_found and not right_found:
            right_base = last_right
            left_base = last_left
            
    return left_base, right_base
```

With the base finding and correction we can now implement the function to create confidence windows in the plot. The function accepts the binary_warped image and given y_coordinates and returns the x,y coordinates for the confidence rectangle around the centers of the left and right lane. 


```python
def create_window_nonlinear(binary_warped, y_coords, last_window, window_size = 75, threshold = 10):
    
    start = y_coords[0]
    end = y_coords[1]
    
    left_base, right_base = get_base_nonlinear(binary_warped, [start, end])
    left_base, right_base = correct_base(left_base, right_base, last_window)
    
    start_x_left, end_x_left = left_base - window_size, left_base + window_size
    start_x_right, end_x_right = right_base - window_size, right_base + window_size 
    
    pts_left = [(start_x_left, start), (end_x_left, end)]
    pts_right = [(start_x_right, start), (end_x_right, end)]
    
    return pts_left, pts_right, left_base, right_base, int(np.mean(y_coords))
```

To iterate through the complete image we initialize the current bases with None so the first observation will not get corrected. First I started with a fixed window size for the base finding function, but noticed that it failed to detect the lane in cases where dotted lanes lead to errors finding the base. Therefore I start with a big window (25% of img) and decrease the size every window by 30% to a minimum of at least 40 pixels. This leads to a small error on the bottom of the screen but increases the quality for pixels further down the road. 

A rectangle with the returned coords will be drawn in an empty image. Additionally we initialize lists to track the x and y position we will calculate a polyfit later on.  


```python
def get_poly_bases_nonlinear(binary_warped):
    
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255

    left_base = None
    right_base = None
    
    last_left_base = None
    last_right_base = None

    left_i = []
    right_i = []
    y_vals = []

    start = binary_warped.shape[0]
    smaller_each_step = 0.7
    smallest_size = 40
    size = int(start / 4)

    while start > 0:
        
        start_y = start
        end_y = start_y - size
        
        last_left_base = left_base
        last_right_base = right_base

        pts_left, pts_right, left_base, right_base, y_val = create_window_nonlinear(
            binary_warped,
            [end_y, start_y],
            [left_base, right_base])
        
        if last_left_base != left_base and last_right_base != right_base:
            cv2.rectangle(out_img,pts_left[0],
                pts_left[1],(0,255,0), 5) 
            cv2.rectangle(out_img,pts_right[0],
                pts_right[1],(0,255,0), 5)

            left_i.append(left_base); right_i.append(right_base); y_vals.append(y_val)
        
        start = end_y
        size = int(size * smaller_each_step)
        if size < smallest_size:
            size = smallest_size
        
    left_i = np.array(left_i)
    right_i = np.array(right_i)
    y_vals = np.array(y_vals)
    
    return left_i, right_i, y_vals, out_img
l,r,y, i = get_poly_bases_nonlinear(binary_warped)
plt.imshow(i);
```

![alt text][lane_blocks]


With the 3 arrays with all the information (x,y) about the lanes, we fit a polynomial to the gathered points and plot it with an area on an empty frame. This also takes the last observation into account and tries to implement a smoothing effect by averaging it with the last observation. Additionally the last observation also serves as a fallback lane if the detected lane differs to much from the last observation. If the lane is slowly shifting above the given confidence interval (between 0.25 and 4 of the last observation), it will be ignored for 10 steps before starting to average the lane again. 


```python
ign = False # ignore current lane
ign_steps = 0

def draw_poly(binary_warped, last_left_fit, last_right_fit):
    
    global ign, ign_steps
    
    out_lanes = np.zeros_like(np.dstack((binary_warped, binary_warped, binary_warped))*255)
    
    left_i, right_i, y_vals, intervals = get_poly_bases_nonlinear(binary_warped)

    left_fit = np.polyfit(y_vals, left_i, 2)
    right_fit = np.polyfit(y_vals, right_i, 2)

    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])
    
    # validating and avaraging current observation
    if last_left_fit[0] != None:
        if abs(last_left_fit[0]) * 0.25 < abs(left_fit[0]) < abs(last_left_fit[0]) * 4:
            left_fitx = np.mean([left_fit[0], last_left_fit[0]])*ploty**2 + np.mean([left_fit[1], last_left_fit[1]])*ploty + np.mean([left_fit[2], last_left_fit[2]])
            left_fit = (np.array(left_fit) + np.array(last_left_fit)) / 2.0
        else:
            if ign_steps > 11:
                ign_steps = 0
                left_fitx = np.mean([left_fit[0], last_left_fit[0]])*ploty**2 + np.mean([left_fit[1], last_left_fit[1]])*ploty + np.mean([left_fit[2], last_left_fit[2]])
            else:
                left_fitx = last_left_fit[0]*ploty**2 + last_left_fit[1]*ploty + last_left_fit[2]
                ign = True
            ign_steps += 1
                   
    else:
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        

    if last_right_fit[0] != None:
        if abs(last_right_fit[0]) * 0.25 < abs(right_fit[0]) < abs(last_right_fit[0]) * 4:
            right_fitx = np.mean([right_fit[0], last_right_fit[0]])*ploty**2 + np.mean([right_fit[1], last_right_fit[1]])*ploty + np.mean([right_fit[2], last_right_fit[2]])
            right_fit = (np.array(right_fit) + np.array(last_right_fit)) / 2.0      
        else:
            if ign_steps > 11:
                ign_steps = 0
                right_fitx = np.mean([right_fit[0], last_right_fit[0]])*ploty**2 + np.mean([right_fit[1], last_right_fit[1]])*ploty + np.mean([right_fit[2], last_right_fit[2]])
            else:
                right_fitx = last_right_fit[0]*ploty**2 + last_right_fit[1]*ploty + last_right_fit[2]
                ign = True
            ign_steps += 1
            
    else:
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    
    # Plots the left and right polynomials on the lane lines
    left_lane = np.vstack((left_fitx, ploty)).T.astype(np.int32)
    right_lane = np.vstack((right_fitx, ploty)).T.astype(np.int32)
    area = np.concatenate((left_lane, right_lane[::-1]))

    cv2.polylines(out_lanes, [left_lane], isClosed=False, color=(141,2,31), thickness=60)
    cv2.polylines(out_lanes, [right_lane], isClosed=False, color=(141,2,31), thickness=60)
    cv2.fillConvexPoly(out_lanes, area, color=(80,200,122))
    
    return out_lanes, left_fit, right_fit, left_lane, right_lane
```

Now we can outwarp the perspective and weight the two images within the plot. As can be seen, the identified area has a high proximity with the actual lane.


```python
lanes, left_fit, right_fit, left_lane, right_lane = draw_poly(binary_warped, [None], [None])
lanes = warp_perspective(lanes, dst, src)

final_img = cv2.addWeighted(img,0.9,lanes,.7,0)
plt.imshow(final_img);
```

![alt text][area]

## 6. Turn-curvature & vehicle position on road

Next we want to calculate the radius of the current curve. We assume a lane width to be around 3 meters at the cars position at y = 720. Inspecting above picture shows that the lane takes around 900 pixels, therefore we can calculate the length of one pixel at this position is around 3 cm. With building an avaraging function for both lanes, we can calculate the approximate radius of the current curve.


```python
CM_PER_VERTICAL_PIXEL = 1 / 3

left_curve = ((1 + (2*left_fit[0]*img.shape[0] + left_fit[1])**2)**1.5) / np.absolute(2*left_fit[0])
right_curve = ((1 + (2*right_fit[0]*img.shape[0] + right_fit[1])**2)**1.5) / np.absolute(2*right_fit[0])

curve_radius = round((left_curve + right_curve) * CM_PER_VERTICAL_PIXEL,1)
```

For determining the position of the car on the road we check for the average center between the two lanes and track the delta to the actual image center in the middle of the x-axis. We check for the first 20 Pixels from bottom up und avarage the deltas to get the lane_dif for the upcoming road segment.


```python
cm_per_pixel = 3.7 / 1000
image_center = int(img.shape[0] / 2)


lane_diffs = [((COORDS[0] + COORDS[1]) / 2 - image_center) * cm_per_pixel 
              for COORDS in zip(right_lane[::-1][:20].T[0], left_lane[::-1][:20].T[0])]
lane_dif = round(np.mean(lane_diffs), 3)
```

To display the information on the image we set up a standard text for both measures.


```python
curv_text = f'Radius: {round(curve_radius,2)} m' if curve_radius > 10000 else f'Radius: {round(curve_radius / 1000,2)} km'
lane_text  = f'Position: {abs(lane_dif)} m'

cv2.putText(final_img, lane_text,(750, 75),cv2.FONT_HERSHEY_SIMPLEX,1.5,(255, 255, 255),4)
cv2.putText(final_img, curv_text,(750, 150),cv2.FONT_HERSHEY_SIMPLEX,1.5,(255, 255, 255),4)

plt.imshow(final_img);
```

![alt text][measures]


The final output of the image has an appended lane and lane area, as well as the information of the position on the road and the curve radius of the current path. For video processing we include the calculations also as as function.


```python
def get_curve_text(left_fit, right_fit, img):
    
    METERS_PER_VERTICAL_PIXEL = 30
    left_curve = ((1 + (2*left_fit[0]*img.shape[0] + left_fit[1])**2)**1.5) / np.absolute(2*left_fit[0])
    right_curve = ((1 + (2*right_fit[0]*img.shape[0] + right_fit[1])**2)**1.5) / np.absolute(2*right_fit[0])
    curve_radius = round((left_curve + right_curve) * METERS_PER_VERTICAL_PIXEL,1)
    curve_text = f'Radius: {round(curve_radius/1000,2)} m'
    cv2.putText(img, curve_text,(750, 150),cv2.FONT_HERSHEY_SIMPLEX,1.5,(255, 255, 255),4)
    
    return img

def get_position_text(left_lane, right_lane, img):
    
    cm_per_pixel = 3.7 / 1000
    image_center = 1280 / 2

    lane_diffs = [((COORDS[0] + COORDS[1]) / 2 - image_center) * cm_per_pixel 
                  for COORDS in zip(right_lane[::-1][:20].T[0], left_lane[::-1][:20].T[0])]
    lane_dif = round(np.mean(lane_diffs), 3)
    
    lane_text =  f'Position: {abs(lane_dif)} m'
    cv2.putText(img, lane_text,(750, 75),cv2.FONT_HERSHEY_SIMPLEX,1.5,(255, 255, 255),4)
    
    return img
```

## Process Video 

Dealing with a stream of image data like in a camera means applying the above steps and returning an appended video. Steps included are:
1. the conversion into HSV Color
2. undistorting the image from the camera calibration
3. filtering color compontents to receive a binary picture
4. create birds-eye-view
5. fit polynomial to identified lane points
6. draw and distort the ploted image on the input
7. calculate curvature & position on road
8. store observations for next frame


```python
from moviepy.editor import VideoFileClip
from IPython.display import HTML

input_vid = 'project_video.mp4'
output_vid = 'project_video_altered.mp4'
```


```python
last_left_fit, last_right_fit, last_lanes = [None], [None], None
ign_steps = 0
ign = False
```


```python
def video_img_processing(img):
    
    global last_left_fit, last_right_fit, last_lanes, ign
    
    img_copy = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    img_copy = cal_undistort(img_copy, objpoints, imgpoints)
    
    binary = get_binary(img_copy)
    binary_warped = warp_perspective(binary, src, dst)
    
    lanes, left_fit, right_fit, left_lane, right_lane = draw_poly(binary_warped, last_left_fit, last_right_fit)
 
    lanes = warp_perspective(lanes, dst, src)
    final_img = cv2.addWeighted(img,0.9,lanes,.7,0)
    
    final_img = get_position_text(left_lane, right_lane, final_img)
    final_img = get_curve_text(left_fit, right_fit, final_img)
    
    if not ign:
        last_left_fit = left_fit
        last_right_fit = right_fit
        last_lanes = lanes
    ign = False
    
    return final_img
```


```python
last_left_fit, last_right_fit, last_lanes = [None], [None], None
ign_steps = 0
ign = False

clip = VideoFileClip(input_vid)
clip_annotated = clip.fl_image(video_img_processing).subclip(0,5)
%time clip_annotated.write_videofile(output_vid, audio=False)
```
   
![alt text][gif_result]

## Weaknesses and shortcomings

Changing weather conditions like intense light or dark shadows will harm the quality of the binary image with lots of input. 

Additionally the performance of the image processing is not quite as fast as the images come in. Expecting a 25fps Video, the programm will need 1s per frame on a good computer, therefore some steps that repeat and dont change over time could be avoided. Some ways to reduce processing may be working with deques for past oberservations and average those, so not every frame has to be evalued. Additionally one could apply some machine learning for correctly identifying lanes, which would also reduce the verification steps. 
