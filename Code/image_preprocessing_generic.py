

import cv2
import numpy as np

img = cv2.imread("test_2.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

"Finding threshold"
th, threshed = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU)

"Finding rectangle or bounding box"
pts = cv2.findNonZero(threshed)
ret = cv2.minAreaRect(pts)

(cx,cy), (w,h), ang = ret

if w>h:
    w,h = h,w
    ang += 90

"Rotating image for alignment of box"
M = cv2.getRotationMatrix2D((cx,cy), 0, 1.0)
rotated = cv2.warpAffine(threshed, M, (img.shape[1], img.shape[0]))
output = cv2.warpAffine(threshed, M, (img.shape[1], img.shape[0]))
BUFFER = 1

def buffer_size(hist):
    size_buffer = []
    concurrent_minima = 0
    minima = min(hist)
    i = 0
    while i < len(hist):
        while i < len(hist) and hist[i] > minima:
            concurrent_minima+=1
            i+=1
        if concurrent_minima != 0:
            size_buffer.append(concurrent_minima)
            concurrent_minima = 0
        i += 1
    return (min(size_buffer), minima)

def bounds(hist, dim_init, dim_size):

    size_buf,threshold = buffer_size(hist)
    bound_limits = [0]
    for y in range(dim_init, dim_size-size_buf):
        if(hist[y]>threshold and (max([hist[val] for val in range(y+1, y+size_buf+1)], default=0)<=threshold)):
            bound_limits.append(y+BUFFER)
        elif(hist[y]<=threshold and (min([hist[val] for val in range(y+1, y+size_buf+1)], default=0)>threshold)):
            bound_limits.append(y-BUFFER)
    bound_limits = list(set(bound_limits))
    bound_limits.sort()
    return (threshold, size_buf, bound_limits)

"Drawing lines or seperation of lines"
hist = cv2.reduce(rotated,1, cv2.REDUCE_AVG).reshape(-1)

HEIGHT, WIDTH = img.shape[:2]
line_threshold, line_buffer, line_bounds = bounds(hist, 0, HEIGHT-1)

output2 =cv2.cvtColor(rotated, cv2.COLOR_GRAY2BGR) 
output = cv2.cvtColor(rotated, cv2.COLOR_GRAY2BGR)

for vals in line_bounds:
    cv2.line(output, (0,vals),(WIDTH, vals), (0,255,0))

"Generation of letters and saving of words in words directory"
prev_y = line_bounds[0] 

cnt_lines = 1
cnt_letters = 1

for y in line_bounds[1:]:
    if (y - prev_y) > line_buffer:

        hist = cv2.reduce(rotated[prev_y:y, :], 0, cv2.REDUCE_AVG).reshape(-1)
        emp = max(hist)
        if emp > 5:
            cv2.imwrite("line/lines_"+str(cnt_lines)+".png", output2[prev_y-BUFFER:y+BUFFER, :])
            letter_threshold, letter_buffer, letter_bounds = bounds(hist, 0, WIDTH-1)
            if letter_bounds:
                prev_x = letter_bounds[0]
        
                for x in letter_bounds[1:]:
                    hist2 = cv2.reduce(rotated[prev_y:y,prev_x:x],0, cv2.REDUCE_AVG).reshape(-1)
                    if (x - prev_x) > letter_buffer//2: 
                        cv2.imwrite("letter/letters_"+str(cnt_letters)+".png", output2[prev_y-BUFFER:y+BUFFER, prev_x:x+BUFFER])
                        cv2.line(output, (x,prev_y),(x, y), (255,0,0))
                        cv2.line(output, (prev_x,prev_y),(prev_x, y), (255,0,0))
                        cnt_letters +=1
                    prev_x = x 
            cnt_lines += 1
    prev_y = y
import matplotlib.pyplot as plt

plt.imshow(output)
plt.show()
    
cv2.imwrite("result.png", output)
