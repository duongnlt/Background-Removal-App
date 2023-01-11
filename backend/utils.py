import cv2
import numpy as np


def order_points(pts):
    # initialzie a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype="float32")
    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    # return the ordered coordinates
    return rect

def get_max_hw(point):
    rect = order_points(np.array(point.squeeze()))
    (tl, tr, br, bl) = rect
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    return maxWidth, maxHeight


def get_transform_matrix(point, maxWidth, maxHeight):
    pts1 = order_points(point.squeeze().copy().astype(np.float32))
    pts2 = np.float32([[0, 0], [maxWidth, 0], [maxWidth, maxHeight], [0, maxHeight]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    return matrix


def get_largest_poly_with_coord(mask_img, reshape=False):
    # mask_img: gray img
    _, mask_img = cv2.threshold(mask_img, 127, 255, 0)
    kernel = np.ones((3, 3), np.uint8)
    mask_img = cv2.dilate(mask_img, kernel, iterations=3)
    contours = cv2.findContours(mask_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]
    # find the biggest countour (c) by the area
    c = max(contours, key=cv2.contourArea)

    rect = cv2.minAreaRect(c)
    poly = np.int0(cv2.boxPoints(rect))
    point = np.array(poly).reshape(-1, 2).astype(int)

    return point

def make_warp_img(img, mask_img):
    warped_point = get_largest_poly_with_coord(mask_img)
    maxWidth, maxHeight = get_max_hw(warped_point)
    matrix = get_transform_matrix(warped_point, maxWidth, maxHeight)
    warped_img = cv2.warpPerspective(img, matrix, (maxWidth, maxHeight))
    return warped_img