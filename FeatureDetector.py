import cv2
import numpy as np

#,0 will covert the image into gray scale

img1=cv2.imread('ImageQuery/book1.JPG',0)
img2=cv2.imread('ImagesTrain/book1.JPG',0)

orb=cv2.ORB_create()
#keypoints and descriptor with orb feature detetcor as it is free and fast
kp1 ,des1=orb.detectAndCompute(img1,None)
kp2 ,des2=orb.detectAndCompute(img2,None)

print(des1.shape)
#imgKp1=cv2.drawKeypoints(img1,kp1,None)
#imgKp2=cv2.drawKeypoints(img2,kp2,None)

#cv2.imshow('KP1',imgKp1)
#cv2.imshow('KP2',imgKp2)

#will return the match
bf= cv2.BFMatcher()
matches=bf.knnMatch(des1,des2,k=2)

#0.75 is good approximation
good=[]
for m,n in matches:
    if m.distance<0.75 *n.distance:
        good.append([m])

#number of matches we are getting
print(len(good))

img3=cv2.drawMatchesKnn(img1,kp1,img2,kp2,good,None,flags=2)


cv2.imshow('img1',img1)
cv2.imshow('img2',img2)
cv2.imshow('img3',img3)
cv2.waitKey(0)


