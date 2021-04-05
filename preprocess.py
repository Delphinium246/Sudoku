import cv2
import imutils
import numpy as np

from skimage.segmentation import clear_border
from imutils.perspective import four_point_transform
from collections import defaultdict


def find_puzzle(imgPath, debug=False):
    img = cv2.imread(imgPath, cv2.CV_8UC1)
    grayimg = cv2.imread('./img_crp.jpg')
    gray = cv2.cvtColor(grayimg, cv2.COLOR_BGR2GRAY)
    # cv2.imshow('image', gray)
    # cv2.waitKey()
    proc = cv2.GaussianBlur(img.copy(), (9, 9), 0)
    proc = cv2.adaptiveThreshold(proc, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    # cv2.imshow('image', proc)
    # cv2.waitKey()
    proc = cv2.bitwise_not(proc, proc)
    kernel = np.array([[0., 1., 0.], [1., 1., 1.], [0., 1., 0.]], np.uint8)
    proc = cv2.dilate(proc, kernel)
    # cv2.imshow('image', proc)
    # cv2.waitKey()
    cnts = cv2.findContours(proc.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    puzzleCnt = None
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            puzzleCnt = approx
            break
    if puzzleCnt is None:
        raise Exception(("Could not find Sudoku puzzle outline. "
                         "Try debugging your thresholding and contour steps."))

    # output = img.copy()
    # cv2.drawContours(output, [puzzleCnt], -1, (0, 255, 0), 2)
    # cv2.imshow("Puzzle Outline", output)
    # cv2.waitKey(0)
    puzzle = four_point_transform(img, puzzleCnt.reshape(4, 2))
    warped = four_point_transform(gray, puzzleCnt.reshape(4, 2))
    # cv2.imshow("Puzzle Transform", warped)
    # cv2.waitKey()
    return (puzzle, warped)

def imgcrop(input, x, y):
    w, h = input.shape
    height = h // y
    width = w // x
    crop_img = defaultdict(dict)
    for i in range(0, y):
        for j in range(0, x):
            crop_img[i][j] = input[(height*i):height*(i+1), (j*width):width*(j+1)]
            # cv2.imshow("cropped Image", crop_img)
            # cv2.waitKey()
    return crop_img
    # for i in range(0, y):
    #     for j in range(0, x):


def extract_digit(cell):
    thresh = cv2.threshold(cell, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    thresh = clear_border(thresh)
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,  cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    if len(cnts) == 0:
        return 0
    c = max(cnts, key=cv2.contourArea)
    mask = np.zeros(thresh.shape, dtype="uint8")
    cv2.drawContours(mask, [c], -1, 255, -1)
    (h, w) = thresh.shape
    percentFilled = cv2.countNonZero(mask) / float(w * h)
    # if less than 3% of the mask is filled then we are looking at
    # noise and can safely ignore the contour
    if percentFilled < 0.03:
        return 0
    # apply the mask to the thresholded cell
    digit = cv2.bitwise_and(thresh, thresh, mask=mask)
    # check to see if we should visualize the masking step
    hori = np.concatenate((cell,digit), axis = 1)
    # cv2.imshow("Cell/Digit", hori)
    # cv2.waitKey(0)
    # return the digit to the calling function
    return digit


def main():
    puzzle, warped = find_puzzle('./img1_crp.jpg')
    # cv2.imshow("Puzzle warped", puzzle)
    # cv2.waitKey()
    n = 9
    c = defaultdict(dict)
    cells = imgcrop(puzzle, n, n)
    for i in range(0, n):
        for j in range(0, n):
            c[i][j] = extract_digit(cells[i][j])
    print(type(c))


if __name__ == '__main__':
    main()
