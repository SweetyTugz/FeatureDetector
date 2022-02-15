import os
import cv2
import numpy as np

path='ImageQuery'
images=[]
classNames=[]

orb=cv2.ORB_create()

mylist=os.listdir(path)
print(mylist)

print('Total classes detected', len(mylist))

for cl in mylist:
    imgcur=cv2.imread(f'{path}/{cl}',0)
    images.append(imgcur)
    classNames.append(os.path.splitext(cl)[0])
print(classNames)


def findDes(images):
    desList=[]
    for image in images:
        kp,des=orb.detectAndCompute(image,None)
        desList.append(des)
    return desList


def findId(img,desList):
    kp2,des2=orb.detectAndCompute(img,None)
    bf = cv2.BFMatcher()
    matchList=[]
    finalVal=-1
    try:
        for des in desList:
             matches = bf.knnMatch(des, des2, k=2)
             good = []
             for m, n in matches:
                if m.distance < 0.75 * n.distance:
                    good.append([m])
             matchList.append(len(good))
    except:
        pass

    if matchList!=0:
        if max(matchList)> 1:
            finalVal=matchList.index(max(matchList))
    return finalVal





desList=findDes(images)
print(len(desList))

cap=cv2.VideoCapture(0)
while True:
    success,img2=cap.read()
    imgOrignal=img2.copy()
    img2=cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
    id=findId(img2,desList)
    print(classNames[id])
    if id!=-1:
        cv2.putText(imgOrignal,classNames[id],(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)

    cv2.imshow('img2',imgOrignal)
    cv2.waitKey(1)


