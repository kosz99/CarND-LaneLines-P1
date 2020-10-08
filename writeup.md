# Udacity Self Driving Car Nanodegree Program: Finding Lane Lines on the Road

## Overview
The goal of this project is make a pipeline that finds lane lines on the road.

## Pipeline
**1 .Import packages**
```
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
```
**2. Convert image to grayscale**
```
def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
```

**3. Apply Gaussian smoothing**
```
def gaussian_blur(img, kernel_size):
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
```
I used kernel_size = 9. Feel free to change these values.

**4. Apply Canny Edge Detection function**
```
def canny(img, low_threshold, high_threshold):
    return cv2.Canny(img, low_threshold, high_threshold)
```
I set low_threshold = 50 and high_threshold = 150

**5. Region_of_interest**
```
rows = image.shape[0]
vertices = np.array([[(110, rows), (430, 330), (540, 330), (900, rows)]], dtype=np.int32)

def region_of_interest(img, vertices):
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image
```
**6. Hough Transform and Draw Lines function**
This is the most important part of project. Hough Transform generates lines contain 4 points(x1, y1, x2, y2).
```
rho = 2
theta = np.pi/180
threshold = 15
min_line_length = 40
max_line_gap = 20


def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img
```
Next, I set variables which store previous line values. These variables are necessary to compute next lines.  
```
prev_left_slope = float("nan")
prev_right_slope = float("nan")
prev_right_x_avg = float("nan")
prev_right_y_avg = float("nan")
prev_left_x_avg = float("nan")
prev_left_y_avg = float("nan")
```
First step to compute lane lines is calculate slope and remove too vertical and too horizontal slopes. Next I seperate left lines from right lines.
```
    for line in lines:
        for x1,y1,x2,y2 in line:
            if(x1 != x2):
                slope = (y2-y1)/(x2-x1)
                if((slope>0.5)and(slope<1.2)):
                    right_slope.append(slope)
                    right_x.append(x1)
                    right_y.append(y1)
                if((slope>-1.2)and(slope<-0.5)):
                    left_slope.append(slope)
                    left_x.append(x1)
                    left_y.append(y1)
```
I use x1 and y1 to calculate b coefficient(b = y1 - slope*x1).
### Final Step
Before final lane line I compute mean of slope, x1 and y1, if prevous values don't exist mean values become final line coefficients, but if prevous values exist final coefficients contain mean values and previous values.
```
    if ((len(right_slope)==0)and(np.isfinite(prev_right_slope))):
        slope_right_avg = prev_right_slope
        right_x_avg = prev_right_x_avg
        right_y_avg = prev_right_y_avg
        new_slope_right = slope_right_avg    
        b_right = right_y_avg - new_slope_right*right_x_avg
        yr1 = 330
        yr2 = img.shape[0]
        xr1 = int((yr1 - b_right)/new_slope_right)
        xr2 = int((yr2-b_right)/new_slope_right)
        cv2.line(img, (xr1, yr1), (xr2, yr2), color, thickness)
        
        
    if ((len(right_slope)!=0)and(np.isfinite(prev_right_slope))):
        slope_right_avg = np.mean(right_slope)
        right_x_avg = np.mean(right_x)
        right_y_avg = np.mean(right_y)
        new_slope_right = alpha*slope_right_avg + (1-alpha)*prev_right_slope
        new_right_x_avg = alpha*right_x_avg + (1-alpha)*prev_right_x_avg
        new_right_y_avg = alpha*right_y_avg + (1-alpha)*prev_right_y_avg
        b_right = new_right_y_avg - new_slope_right*new_right_x_avg
        yr1 = 330
        yr2 = img.shape[0]
        xr1 = int((yr1 - b_right)/new_slope_right)
        xr2 = int((yr2-b_right)/new_slope_right)
        cv2.line(img, (xr1, yr1), (xr2, yr2), color, thickness)
        prev_right_slope = new_slope_right
        prev_right_x_avg = new_right_x_avg
        prev_right_y_avg = new_right_y_avg
    
    if ((len(right_slope)!=0)and(np.isnan(prev_right_slope))):
        slope_right_avg = np.mean(right_slope)
        right_x_avg = np.mean(right_x)
        right_y_avg = np.mean(right_y)
        new_slope_right = slope_right_avg    
        b_right = right_y_avg - new_slope_right*right_x_avg
        yr1 = 330
        yr2 = img.shape[0]
        xr1 = int((yr1 - b_right)/new_slope_right)
        xr2 = int((yr2-b_right)/new_slope_right)
        cv2.line(img, (xr1, yr1), (xr2, yr2), color, thickness)
        prev_right_slope = new_slope_right
        prev_right_x_avg = right_x_avg
        prev_right_y_avg = right_y_avg
```
## Reflections
Generally red lines(indicators) indicate lane lines, indicators are jittery. Perhaps using HSV instead grayscale would make indicators more stable. 



