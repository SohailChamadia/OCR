import cv2

img = cv2.imread("test.jpg")
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
M = cv2.getRotationMatrix2D((cx,cy), ang, 1.0)
output = rotated = cv2.warpAffine(threshed, M, (img.shape[1], img.shape[0]))

"Drawing lines or seperation of lines"
hist = cv2.reduce(rotated,1, cv2.REDUCE_AVG).reshape(-1)

th = 2
H,W = img.shape[:2]
bounds = [y for y in range(H-1) if ((hist[y]>th and hist[y+1]<=th) or (hist[y]<=th and hist[y+1]>th))]
bounds.append(H)
output = cv2.cvtColor(rotated, cv2.COLOR_GRAY2BGR)

"Generation of words and saving of words in words directory"
prev=-1
cnt=1
cnt2=1
for y in reversed(bounds):
    if (prev-y)>10:
        hist = cv2.reduce(rotated[y:prev,:],0, cv2.REDUCE_AVG).reshape(-1)
        bounds_words = [y for y in range(W-4) if (hist[y]>th and (max(hist[y+1],hist[y+2],hist[y+3],hist[y+4])<=th) or 
                        (hist[y]>th and (max(hist[y-1],hist[y-2],hist[y-3],hist[y-4])<th))) ]                
        prev_x=-1
        for x in reversed(bounds_words):
            if (prev_x-x)>15:
                hist2 = cv2.reduce(rotated[y:prev,x:prev_x],0, cv2.REDUCE_AVG).reshape(-1)
                emp = max(hist2)
                if emp > 50: 
                    cv2.imwrite("word/words_"+str(cnt)+".png", output[y-3:prev+3,x-1:prev_x+1])
                    opt = output[y-3:prev+3,x-1:prev_x+1]
                    x1 = x
                    for itr in range(len(hist2)-1):
                        if hist2[itr] < th:
                            cv2.imwrite("letter/letters_"+str(cnt2)+".png", output[y-3:prev+3,x1-1:itr+x+1])
                            x1 = x + itr 
                            cnt2 += 1
                    cv2.imwrite("letter/letters_"+str(cnt2)+".png", output[y-3:prev+3,x1-1:itr+x+1])
                    cnt2 += 1
                    cnt+=1
            prev_x=x
    else:
        bounds.remove(y)
    prev=y
cv2.imwrite("word/test.png", output[y:prev,x:prev_x])    

cv2.imwrite("result.png", output)