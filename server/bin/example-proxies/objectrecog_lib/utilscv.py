#!/usr/bin/env python

import cv2


setattr(cv2, 'LINE_AA', cv2.CV_AA)
# Auxiliary function to draw text with contrast in an image:
def draw_str(dst, (x, y), s):
    cv2.putText(dst, s, (x + 1, y + 1), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 0),
                thickness=2, lineType=cv2.LINE_AA)
    cv2.putText(dst, s, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255),
                lineType=cv2.LINE_AA)


# Auxiliary function to trim an ROI over an image of given size:
def fixroi(roi, imshape):
    if roi == ((-1, -1), (-1, -1)):
        rroi = ((0, 0), (imshape[1], imshape[0]))
    else:
        rroi = ((max(0, min(roi[0][0], roi[1][0])),
                 max(0, min(roi[0][1], roi[1][1]))),
                (min(imshape[1], max(roi[0][0], roi[1][0])),
                 min(imshape[0], max(roi[0][1], roi[1][1]))))
    return rroi


# Obtain the subpicture given by an ROI:
def subimg(pimg, proi):
    return pimg[proi[0][1]:proi[1][1], proi[0][0]:proi[1][0]]


# Write the subpicture given by an ROI:
def setsubimg(img1, img2, proi):
    img1[proi[0][1]:proi[1][1], proi[0][0]:proi[1][0]] = img2
