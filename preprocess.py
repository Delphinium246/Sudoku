import random
import os
import cv2
import imutils
import numpy as np
import tensorflow as tf

from skimage.segmentation import clear_border
from imutils.perspective import four_point_transform
from collections import defaultdict
from keras.models import load_model

model = load_model('./cnn_model.h5')

v = 0


def find_puzzle(imgPath, debug=False):
    img = cv2.imread(imgPath, cv2.CV_8UC1)
    grayimg = cv2.imread(imgPath)
    gray = cv2.cvtColor(grayimg, cv2.COLOR_BGR2GRAY)
    proc = cv2.GaussianBlur(img.copy(), (9, 9), 0)
    proc = cv2.adaptiveThreshold(proc, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    proc = cv2.bitwise_not(proc, proc)
    kernel = np.array([[0., 1., 0.], [1., 1., 1.], [0., 1., 0.]], np.uint8)
    proc = cv2.dilate(proc, kernel)
    cnts = cv2.findContours(proc.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
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
    return puzzle, warped


def imgcrop(input, x, y):
    w, h = input.shape
    height = h // y
    width = w // x
    crop_img = defaultdict(dict)
    for i in range(0, y):
        for j in range(0, x):
            crop_img[i][j] = input[(height * i):height * (i + 1), (j * width):width * (j + 1)]
    return crop_img


def extract_digit(cell):
    thresh = cv2.threshold(cell, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    thresh = clear_border(thresh)
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    if len(cnts) == 0:
        return 'X'
    c = max(cnts, key=cv2.contourArea)
    mask = np.zeros(thresh.shape, dtype="uint8")
    cv2.drawContours(mask, [c], -1, 255, -1)
    (h, w) = thresh.shape
    percentFilled = cv2.countNonZero(mask) / float(w * h)
    if percentFilled < 0.03:
        return 0
    digit = cv2.bitwise_and(thresh, thresh, mask=mask)
    digit = cv2.resize(digit, (28, 28), interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(r'./digit_data/temp_' + str(random.randint(100,999)) + '.jpg', digit)
    return 1


def pre_process_image(img, skip_dilate=False):
    proc = cv2.GaussianBlur(img.copy(), (9, 9), 0)
    proc = cv2.adaptiveThreshold(proc, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    proc = cv2.bitwise_not(proc, proc)
    if not skip_dilate:
        kernel = np.array([[0., 1., 0.], [1., 1., 1.], [0., 1., 0.]], np.uint8)
        proc = cv2.dilate(proc, kernel)
    return proc


def main():
    # directory = r'./sudoku_out/'
    # # output = r'./sudoku_out/'
    # # i = 0
    # for filename in os.listdir(directory):
    #     if filename.endswith(".png"):
    #         f = os.path.join(directory, filename)
    #         print(f)
    #         puzzle, warped = find_puzzle(f)
            # i = i+1
    # for i in range(15, 16, 1):
    #     print('./sudoku/sudoku'+str(i)+'.png')
    #     puzzle, warped = find_puzzle('./sudoku/sudoku'+str(i)+'.png')
    puzzle, warped = find_puzzle('./sudoku9.PNG')
    n = 9
    c = defaultdict(dict)
    cells = imgcrop(puzzle, n, n)
    for i in range(0, n):
        for j in range(0, n):
            c[i][j] = extract_digit(cells[i][j])
            # for i in range(0, n):
            #     print(c[i])


if __name__ == '__main__':
    main()
