import cv2
import numpy as np
import math
import glob
from os.path import exists
from numpy.lib.function_base import bartlett
import requests
import zipfile
from threading import Thread, Lock
from queue import Queue
import concurrent.futures
from QRMatrix import QRMatrixDecoder
from pyzbar import pyzbar


def prepareDataset():
    file_name = "1d_barcode_hough.zip"
    if not exists(file_name):
        r = requests.get("http://artelab.dista.uninsubria.it/downloads/datasets/barcode/hough_barcode_1d/" + file_name, allow_redirects=True)
        open(file_name, 'wb').write(r.content)
    if not exists("1d_barcode_hough"):
        with zipfile.ZipFile(file_name, 'r') as zip_ref:
            zip_ref.extractall()

def four_point_transform(image, rect, square = False):
        (tl, tr, br, bl) = rect
        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))
        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))

        if square:
            if maxHeight < maxWidth:
                maxWidth = maxHeight
            else:
                maxHeight = maxWidth

        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]], dtype = "float32")
        M = cv2.getPerspectiveTransform(rect, dst)

        warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
        return warped, M

def rotate_image(image, angle):
  image_center = tuple(np.array(image.shape[1::-1]) / 2)
  rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
  result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
  return result

def rotate_point(origin, point, angle):
    ox, oy = origin
    px, py = point

    qx = ox + np.cos(np.radians(angle)) * (px - ox) - np.sin(np.radians(angle)) * (py - oy)
    qy = oy + np.sin(np.radians(angle)) * (px - ox) + np.cos(np.radians(angle)) * (py - oy)
    return qx, qy

def rotate_boundary(origin, boundary, angle):
    for i, point in enumerate(boundary):
        boundary[i] = rotate_point(origin, point, angle)
    return boundary

def decodeBarcode(bars):
    if sum(bars[:3]) + sum(bars[27:32]) + sum(bars[-3:]) != 11:
        return

    bars = np.append(bars[3:27], bars[32:-3])
    if (bars[0] + bars[2]) % 2 != 0:
        bars = np.flip(bars)
    bars = np.array_split(bars, 12)
    
    if np.any([sum(b) != 7 for b in bars]):
        return

    first_digit_dictionary = {(1,1,1,1,1,1): 0, (1,1,0,1,0,0): 1, (1,1,0,0,1,0): 2, (1,1,0,0,0,1): 3, (1,0,1,1,0,0): 4, (1,0,0,1,1,0): 5, (1,0,0,0,1,1): 6, (1,0,1,0,1,0): 7, (1,0,1,0,0,1): 8, (1,0,0,1,0,1): 9}
    value_dictionary = {
        (3,2,1,1): 0, (2,2,2,1): 1, (2,1,2,2): 2, (1,4,1,1): 3, (1,1,3,2): 4, (1,2,3,1): 5, (1,1,1,4): 6, (1,3,1,2): 7, (1,2,1,3): 8, (3,1,1,2): 9
        ,(1,1,2,3): 0, (1,2,2,2): 1, (2,2,1,2): 2, (1,1,4,1): 3, (2,3,1,1): 4, (1,3,2,1): 5, (4,1,1,1): 6, (2,1,3,1): 7, (3,1,2,1): 8, (2,1,1,3): 9
        }
    barcode_value = [value_dictionary.get(tuple(i), None) for i in bars]
    if None in barcode_value:
        return

    first_digit = first_digit_dictionary.get(tuple((b[1] + b[3]) % 2 for b in bars[:6]), None)

    if first_digit is None:
        return

    barcode_value.insert(0, first_digit)

    check_digit = (sum(barcode_value[:-1:2]) + sum(barcode_value[1::2])*3) % 10
    if barcode_value[12] != (check_digit if check_digit == 0 else 10 - check_digit):
        return

    return barcode_value

def readBarcodeLine(line):
    bars = np.array([])
    #0 - black - even index
    #1 - white - odd index
    for i in np.int0(line/255):
        if len(bars) == 0 and i == 1:
            continue
        if len(bars) == 0 or len(bars)%2 == i:
            bars = np.append(bars, 1)
        else:
            bars[-1] += 1

    if len(bars) < 59:
        return

    for i in range(0, len(bars) - 59, 2):
        bars_window = bars[0+i:59+i]
        barcodeWidth = sum(bars_window)
        averageBarWidth = barcodeWidth / 95
    
        bars_window = np.int0(np.round(bars_window / averageBarWidth))
        barcode = decodeBarcode(bars_window)
        if barcode:
            return barcode
    return None

def contourDistanceTo(contour, center):
    M = cv2.moments(contour)
    center_X = int(M["m10"] / M["m00"])
    center_Y = int(M["m01"] / M["m00"])
    contour_center = (center_X, center_Y)

    return np.linalg.norm(center-contour_center)


def findBarcode(I, sobel_ksize, pre_blur_ksize, post_blur_ksize, thresh, scale, angle, morfRectH, morfRectW, erosionIterations, dilationIterations):
    sobel_ksize = -1 if sobel_ksize == 1 else sobel_ksize
    pre_blur_ksize = -1 if pre_blur_ksize == 1 else pre_blur_ksize
    post_blur_ksize = -1 if post_blur_ksize == 1 else post_blur_ksize

    layers = []
    I_original = I
    layers.append(I_original)

    I = cv2.resize(I_original, (int(I_original.shape[1] * scale / 100), int(I_original.shape[0] * scale / 100)))
    layers.append(I)

    I = rotate_image(I, angle)
    layers.append(I)

    if pre_blur_ksize != 0:
        I = cv2.GaussianBlur(I, (pre_blur_ksize, pre_blur_ksize), 0)
    layers.append(I)

    I = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
    layers.append(I)

    I = cv2.Laplacian(I, cv2.CV_16U, ksize=9)
    layers.append(I)

    if post_blur_ksize != 0:
        I = cv2.GaussianBlur(I, (post_blur_ksize, post_blur_ksize), 0)
    layers.append(I)

    _,I = cv2.threshold(I, 65535-10000, 65535, cv2.THRESH_BINARY)
    layers.append(I)

    I = np.uint8(I)
    layers.append(I)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 7))
    I = cv2.morphologyEx(I, cv2.MORPH_CLOSE, kernel)
    layers.append(I)

    I = cv2.erode(I, None, iterations = erosionIterations)
    layers.append(I)

    I = cv2.dilate(I, None, iterations = dilationIterations)
    layers.append(I)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (morfRectH, morfRectW))
    I = cv2.morphologyEx(I, cv2.MORPH_CLOSE, kernel)
    layers.append(I)

    contours, _ = cv2.findContours(I, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key = cv2.contourArea, reverse = True)
    image_center = tuple(np.int32(np.array(I_original.shape[1::-1]) / 2))

    boxes = []
    for contour in contours:
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = box / (scale / 100)
        box = rotate_boundary(image_center, box, angle)
        box = np.int0(box)
        boxes.append(box)
   

    I_visualizer = I_original.copy()
    if len(boxes) > 0:
        cv2.drawContours(I_visualizer, boxes[1:], -1, (0,255,0), 3)
        cv2.drawContours(I_visualizer, [boxes[0]], -1, (0,0,255), 3)
    layers.append(I_visualizer)

    return boxes, layers

def extractBarcode(I, box):
    I_barcode = four_point_transform(I, np.float32(box))
    I_barcode = cv2.rotate(I_barcode, cv2.ROTATE_90_COUNTERCLOCKWISE) if I_barcode.shape[0] > I_barcode.shape[1] else I_barcode
    #I_barcode = cv2.fastNlMeansDenoisingColored(I_barcode,None,10,10,7,21)
    I_barcode, _ = cv2.cvtColor(I_barcode, cv2.COLOR_BGR2GRAY)
    #_,I_barcode = cv2.threshold(I_barcode, 125, 255, cv2.THRESH_BINARY)
    I_barcode = cv2.adaptiveThreshold(I_barcode,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,15,2)
    return I_barcode

def readBarcode(I_barcode):
    for i, line in enumerate(I_barcode):
        barcode_value = readBarcodeLine(line)
        if barcode_value:
            I_barcode = cv2.cvtColor(I_barcode, cv2.COLOR_GRAY2BGR)
            cv2.line(I_barcode, (0, i), (len(line), i), (0,0,255), 1)
            return barcode_value, I_barcode
    return None, None

def foo(image, sobel_ksize, pre_blur_ksize, post_blur_ksize, thresh, scale, angle, morfRectH, morfRectW, erosionIterations, dilationIterations):
    print(image)
    I_original = cv2.imread(image)
    boxes, layers = findBarcode(I_original, sobel_ksize, pre_blur_ksize, post_blur_ksize, thresh, scale, angle, morfRectH, morfRectW, erosionIterations, dilationIterations)
    if len(boxes) > 0:
        I_barcode = extractBarcode(I_original, boxes[0])
        barcode, I_barcode = readBarcode(I_barcode)
        if barcode:
            print(barcode)
            return barcode

def benchmark(x):
    sobel_ksize = cv2.getTrackbarPos('sobel_ksize',"options")
    pre_blur_ksize = cv2.getTrackbarPos('pre_blur_ksize',"options")
    post_blur_ksize = cv2.getTrackbarPos('post_blur_ksize',"options")
    thresh = cv2.getTrackbarPos('thresh',"options")
    scale = cv2.getTrackbarPos('scale',"options")
    angle = cv2.getTrackbarPos('angle',"options")
    morfRectH = cv2.getTrackbarPos('morfRectH',"options")
    morfRectW = cv2.getTrackbarPos('morfRectW',"options")
    erosionIterations = cv2.getTrackbarPos('erosionIterations',"options")
    dilationIterations = cv2.getTrackbarPos('dilationIterations',"options")

    images = glob.glob("1d_barcode_hough\\Original\\*.jpgbarcodeOrig.png")

    barcodes = np.array([])

    with concurrent.futures.ThreadPoolExecutor(8) as executor:
        futures = [executor.submit(foo, image, sobel_ksize, pre_blur_ksize, post_blur_ksize, thresh, scale, angle, morfRectH, morfRectW, erosionIterations, dilationIterations) for image in images]
        barcodes = np.array([f.result() for f in futures])

    count = len(barcodes)
    succ = np.count_nonzero(barcodes)
    print(f'{succ}/{count} ({round(succ/count*100, 2)}%)')

def calculate(x):
    
    image = cv2.getTrackbarPos('image',"options")
    layer = cv2.getTrackbarPos('layer',"options")
    sobel_ksize = cv2.getTrackbarPos('sobel_ksize',"options")
    pre_blur_ksize = cv2.getTrackbarPos('pre_blur_ksize',"options")
    post_blur_ksize = cv2.getTrackbarPos('post_blur_ksize',"options")
    thresh = cv2.getTrackbarPos('thresh',"options")
    scale = cv2.getTrackbarPos('scale',"options")
    angle = cv2.getTrackbarPos('angle',"options")
    morfRectH = cv2.getTrackbarPos('morfRectH',"options")
    morfRectW = cv2.getTrackbarPos('morfRectW',"options")
    erosionIterations = cv2.getTrackbarPos('erosionIterations',"options")
    dilationIterations = cv2.getTrackbarPos('dilationIterations',"options")

    images = glob.glob("1d_barcode_hough\\Original\\*.jpgbarcodeOrig.png")

    I_original = cv2.imread(images[image])
    

    boxes, layers = findBarcode(I_original, sobel_ksize, pre_blur_ksize, post_blur_ksize, thresh, scale, angle, morfRectH, morfRectW, erosionIterations, dilationIterations)
    cv2.imshow("5", layers[layer if layer < len(layers) else len(layers) - 1])
    
    if len(boxes) > 0:
        I_barcode = extractBarcode(I_original, boxes[0])

        cv2.imshow("4", I_barcode)
        barcode, I_barcode = readBarcode(I_barcode)
        if barcode:
            cv2.imshow("4", I_barcode)
            print(barcode)



def barcode():
    prepareDataset()
    cv2.namedWindow("options", cv2.WINDOW_GUI_EXPANDED)
    cv2.resizeWindow("options", 1800, 400)
    cv2.createTrackbar("image", "options", 0,364, calculate)
    cv2.createTrackbar("layer", "options", 13,13, calculate)
    cv2.createTrackbar("scale", "options", 100,100, calculate)
    cv2.createTrackbar("angle", "options", 0,360, calculate)
    cv2.createTrackbar("sobel_ksize", "options", 1, 20, calculate)
    cv2.createTrackbar("pre_blur_ksize", "options", 9, 20, calculate)
    cv2.createTrackbar("post_blur_ksize", "options", 0, 20, calculate)
    cv2.createTrackbar("thresh", "options", 225,255, calculate)
    cv2.createTrackbar("morfRectH", "options", 21,100, calculate)
    cv2.createTrackbar("morfRectW", "options", 7,100, calculate)
    cv2.createTrackbar("erosionIterations", "options", 14,100, calculate)
    cv2.createTrackbar("dilationIterations", "options", 16,100, calculate)
    cv2.createTrackbar("benchmark", "options", 0,1, benchmark)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

def processQrcode(I, FIPs):
    return


def findQrcode(I, pre_blur_ksize, scale, angle, ksize, c, erosionIterations, xdisloc, ydisloc, dislocRange):
    pre_blur_ksize = -1 if pre_blur_ksize == 1 else pre_blur_ksize

    layers = []
    I_original = I
    layers.append(I_original)

    I = cv2.resize(I_original, (int(I_original.shape[1] * scale / 100), int(I_original.shape[0] * scale / 100)))
    layers.append(I)

    I = cv2.copyMakeBorder(I, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=(255,255,255))
    layers.append(I)

    I = rotate_image(I, angle)
    layers.append(I)

    if pre_blur_ksize != 0:
        I = cv2.GaussianBlur(I, (pre_blur_ksize, pre_blur_ksize), 0)
    layers.append(I)

    I = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
    layers.append(I)
    
    I = cv2.adaptiveThreshold(I,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,ksize,c)
    layers.append(I)

    I = 255-I
    layers.append(I)
    
    # https://docs.opencv.org/master/d9/d8b/tutorial_py_contours_hierarchy.html
    # Hierarchy Representation in OpenCV
    # Each contour has its own information regarding what hierarchy it is, who is its child, who is its parent etc. 
    # OpenCV represents it as an array of four values : [Next, Previous, First_Child, Parent]
    contours, hierarchy = cv2.findContours(I, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    hierarchy = hierarchy[0]

    FIPs = []
    
    I_visualizer = cv2.cvtColor(I, cv2.COLOR_GRAY2BGR)
    for component in zip(contours, hierarchy):
        currentContour = component[0]
        currentHierarchy = component[1]

        if (currentHierarchy[0] < 0) and (currentHierarchy[2] < 0) and (currentHierarchy[2] < 0) and (currentHierarchy[3] > 0):

            parent_idx = currentHierarchy[3]
            parent_hier = hierarchy[parent_idx]

            if (parent_hier[0] < 0) and (parent_hier[1] < 0) and (parent_hier[3] > 0):
                area = cv2.contourArea(currentContour)
                parea = cv2.contourArea(contours[parent_idx])
                if abs(0.30-area/parea) < 0.08:
                    M = cv2.moments(currentContour)
                    #approx = cv2.approxPolyDP(currentContour, 0.01 * cv2.arcLength(currentContour, True), True)
                    x = int(M["m10"] / M["m00"])
                    y = int(M["m01"] / M["m00"])
                    outerSquare = np.squeeze(cv2.approxPolyDP(contours[parent_hier[3]], 3, True), 1)
                    innerSquare = np.squeeze(cv2.approxPolyDP(contours[parent_idx], 3, True), 1)
                    FIPs.append([np.array([x,y]), innerSquare, outerSquare])

                    for i in outerSquare:
                            I_visualizer = cv2.circle(I_visualizer, i, radius=2, color=(0, 0, 255), thickness=-1)

                    for i in innerSquare:
                            I_visualizer = cv2.circle(I_visualizer, i, radius=2, color=(0, 0, 255), thickness=-1)

    for fip in FIPs:
        I_visualizer = cv2.circle(I_visualizer, fip[0], radius=5, color=(0, 0, 255), thickness=-1)
    layers.append(I_visualizer)

    lenFIPs = len(FIPs)
    if lenFIPs == 3:
        A = FIPs[0][0]
        B = FIPs[1][0]
        C = FIPs[2][0]
        a = B-C
        b = A-C
        c = A-B
        Ad = np.arccos(np.dot(b,c)/(np.linalg.norm(b) * np.linalg.norm(c)))
        Cd = np.arccos(np.dot(b,a)/(np.linalg.norm(b) * np.linalg.norm(a)))
        Bd = math.pi - Ad - Cd
        #Largest angle
        Corner_index = max(range(3), key=[Ad, Bd, Cd].__getitem__)
        #FIPs.insert(0, FIPs.pop(Corner_index))
        FIPs[0], FIPs[Corner_index] = FIPs[Corner_index], FIPs[0]
        Corner = FIPs[0][0]
        I_visualizer = cv2.circle(I_visualizer, Corner, radius=5, color=(255, 0, 0), thickness=-1)
        I_visualizer = cv2.line(I_visualizer, Corner, FIPs[1][0], color=(0, 255, 0), thickness=2)
        I_visualizer = cv2.line(I_visualizer, Corner, FIPs[2][0], color=(0, 255, 0), thickness=2)

        #Center
        A = FIPs[1][0]
        B = FIPs[2][0]
        Center = np.rint((A+B)/2).astype(int)
        I_visualizer = cv2.circle(I_visualizer, Center, radius=5, color=(255, 0, 0), thickness=-1)

        
        for FIP in FIPs:
            corners = FIP[1]
            dists = [np.linalg.norm(x - Center) for x in corners]
            Corner_index = min(range(len(dists)), key=dists.__getitem__)
            dists[Corner_index], dists[0] = dists[0], dists[Corner_index]
            corners[Corner_index], corners[0] = np.copy(corners[0]), np.copy(corners[Corner_index])
            I_visualizer = cv2.circle(I_visualizer, corners[0], radius=2, color=(255, 0, 0), thickness=-1)
            
            Corner_index = max(range(len(dists)), key=dists.__getitem__)
            corners[Corner_index], corners[1] = np.copy(corners[1]), np.copy(corners[Corner_index])
            I_visualizer = cv2.circle(I_visualizer, corners[1], radius=2, color=(0, 255, 0), thickness=-1)

            corners = FIP[2]
            dists = [np.linalg.norm(x - Center) for x in corners]
            Corner_index = min(range(len(dists)), key=dists.__getitem__)
            dists[Corner_index], dists[0] = dists[0], dists[Corner_index]
            corners[Corner_index], corners[0] = np.copy(corners[0]), np.copy(corners[Corner_index])
            I_visualizer = cv2.circle(I_visualizer, corners[0], radius=2, color=(255, 0, 0), thickness=-1)
            
            Corner_index = max(range(len(dists)), key=dists.__getitem__)
            corners[Corner_index], corners[1] = np.copy(corners[1]), np.copy(corners[Corner_index])
            I_visualizer = cv2.circle(I_visualizer, corners[1], radius=2, color=(0, 255, 0), thickness=-1)

            dists = [np.linalg.norm(x - FIPs[0][0]) for x in corners[2:]]
            Corner_index = max(range(len(dists)), key=dists.__getitem__)
            corners[Corner_index + 2], corners[2] = np.copy(corners[2]), np.copy(corners[Corner_index + 2])
            I_visualizer = cv2.circle(I_visualizer, corners[2], radius=2, color=(255, 255, 0), thickness=-1)
        
        A = FIPs[1][2][1]
        ACv = FIPs[1][2][2] - A
        B = FIPs[2][2][1]
        BCv = FIPs[2][2][2] - B
        ABv = B - A
        BAv = A - B

        Alpha = np.arcsin(np.dot(ACv, ABv) / (np.linalg.norm(ACv) * np.linalg.norm(ABv)))
        Beta = np.arcsin(np.dot(BCv, BAv) / (np.linalg.norm(BCv) * np.linalg.norm(BAv)))
        Gamma = math.pi - Alpha - Beta
        
        c = np.linalg.norm(BAv)
        b = c * np.sin(Beta) / np.sin(Gamma)
        a = c * np.sin(Alpha) / np.sin(Gamma)

        fourthCorner = line_intersection([FIPs[1][2][1], FIPs[1][2][2]], [FIPs[2][2][1], FIPs[2][2][2]])
        fourthCorner[0] += xdisloc
        fourthCorner[1] += ydisloc
        I_visualizer = cv2.circle(I_visualizer, fourthCorner, radius=2, color=(0, 0, 255), thickness=-1)


        dislocRange = dislocRange
        rangex = range(fourthCorner[0]-dislocRange, fourthCorner[0]+dislocRange+1)
        rangey = range(fourthCorner[1]-dislocRange, fourthCorner[1]+dislocRange+1)

        sobelScoresX = [calculateSobelScore(four_point_transform(I, np.float32([FIPs[0][2][1],FIPs[1][2][1],[x, fourthCorner[1]],FIPs[2][2][1]]), True)[0]) for x in rangex]
        maxSobelScoreX = max(range(len(sobelScoresX)), key=sobelScoresX.__getitem__)
        fourthCornerCorrectedX = rangex.__getitem__(maxSobelScoreX)
        
        if maxSobelScoreX != dislocRange:
            prevSobelScoresX = sobelScoresX[dislocRange]
            ix = 1
            direction = int((maxSobelScoreX - dislocRange) / abs(maxSobelScoreX - dislocRange))
            while sobelScoresX[maxSobelScoreX] > prevSobelScoresX and c < dislocRange * ix:
                fourthCornerCorrectedX = rangex.__getitem__(maxSobelScoreX)
                prevSobelScoresX = sobelScoresX[maxSobelScoreX]
                rangex = range(fourthCorner[0] + 1 + (dislocRange * ix) * direction, fourthCorner[0] + 1 + (dislocRange * (ix + 1)) * direction + 1, direction)
                sobelScoresX = [calculateSobelScore(four_point_transform(I, np.float32([FIPs[0][2][1],FIPs[1][2][1],[x, fourthCorner[1]],FIPs[2][2][1]]), True)[0]) for x in rangex]
                maxSobelScoreX = max(range(len(sobelScoresX)), key=sobelScoresX.__getitem__)
                ix += 1

        sobelScoresY = [calculateSobelScore(four_point_transform(I, np.float32([FIPs[0][2][1],FIPs[1][2][1],[fourthCorner[0] + (maxSobelScoreX - dislocRange), y],FIPs[2][2][1]]), True)[0]) for y in rangey]
        maxSobelScoreY = max(range(len(sobelScoresY)), key=sobelScoresY.__getitem__)
        fourthCornerCorrectedY = rangey.__getitem__(maxSobelScoreY)

        if maxSobelScoreY != dislocRange:
            prevSobelScoresY = sobelScoresY[dislocRange]
            iy = 1
            direction = int((maxSobelScoreY - dislocRange) / abs(maxSobelScoreY - dislocRange))
            while sobelScoresY[maxSobelScoreY] > prevSobelScoresY and c < dislocRange * iy:
                fourthCornerCorrectedY = rangey.__getitem__(maxSobelScoreY)
                prevSobelScoresY = sobelScoresY[maxSobelScoreY]
                rangey = range(fourthCorner[1] + 1 + (dislocRange * iy) * direction, fourthCorner[1] + 1 + (dislocRange * (iy + 1)) * direction + 1, direction)
                sobelScoresY = [calculateSobelScore(four_point_transform(I, np.float32([FIPs[0][2][1],FIPs[1][2][1],[fourthCorner[0] + (maxSobelScoreY - dislocRange), y],FIPs[2][2][1]]), True)[0]) for y in rangey]
                maxSobelScoreY = max(range(len(sobelScoresY)), key=sobelScoresY.__getitem__)
                iy += 1

        fourthCornerCorrected = [fourthCornerCorrectedX, fourthCornerCorrectedY]
        I_visualizer = cv2.circle(I_visualizer, fourthCornerCorrected, radius=2, color=(0, 255, 0), thickness=-1)

        I_extractCorrected, M = four_point_transform(I, np.float32([FIPs[0][2][1], FIPs[1][2][1], fourthCornerCorrected, FIPs[2][2][1]]), True)
        sobelScoreCorrected = calculateSobelScore(I_extractCorrected)
        sobelMask, sobelMaskRows, sobelMaskCols, sobelMaskRowCount, sobelMaskColCount = getSobelMask(I_extractCorrected)
        _, I_extractCorrected = cv2.threshold(I_extractCorrected, 127, 255, cv2.THRESH_BINARY)



        I_extractCorrectedVisualiser = cv2.cvtColor(I_extractCorrected, cv2.COLOR_GRAY2BGR)

        


        I_extractCorrectedVisualiser = cv2.putText(I_extractCorrectedVisualiser, str(sobelScoreCorrected), (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, 2)
        
        FIPs = transformFIPs(FIPs, M)
        I_extractCorrectedVisualiser = cv2.circle(I_extractCorrectedVisualiser, FIPs[0][0], radius=5, color=(255, 0, 0), thickness=-1)
        I_extractCorrectedVisualiser = cv2.circle(I_extractCorrectedVisualiser, FIPs[1][0], radius=5, color=(0, 0, 255), thickness=-1)
        I_extractCorrectedVisualiser = cv2.circle(I_extractCorrectedVisualiser, FIPs[2][0], radius=5, color=(0, 0, 255), thickness=-1)

        midCornerPoint = (FIPs[0][1][0] + FIPs[0][2][0]) / 2
        midCornerPointX = (FIPs[1][1][0] + FIPs[1][2][0]) / 2
        midCornerPointY = (FIPs[2][1][0] + FIPs[2][2][0]) / 2

        I_extractCorrectedVisualiser = cv2.line(I_extractCorrectedVisualiser, np.rint(midCornerPoint).astype(int), np.rint(midCornerPointX).astype(int), color=(0, 255, 0), thickness=1)
        I_extractCorrectedVisualiser = cv2.line(I_extractCorrectedVisualiser, np.rint(midCornerPoint).astype(int), np.rint(midCornerPointY).astype(int), color=(0, 255, 0), thickness=1)

        vectorX = midCornerPointX - midCornerPoint
        vectorY = midCornerPointY - midCornerPoint

        cornerToXLen = np.linalg.norm(vectorX)
        cornerToYLen = np.linalg.norm(vectorY)
        normX = vectorX / cornerToXLen
        normY = vectorY / cornerToYLen
        
        xMidPoints = np.empty((0,2), int)
        yMidPoints = np.empty((0,2), int)

        tmpColor = 255
        tmpStart = None
        tmpPrev = None
        for i in range(math.ceil(cornerToXLen)):
            point = np.rint(midCornerPoint + normX * i).astype(int)
            color = I_extractCorrected[point[1],point[0]]
            if color == tmpColor:
                tmpPrev = point
                continue
            if tmpStart is not None:
                colMidPoint = np.rint((tmpPrev + tmpStart) / 2).astype(int)
                xMidPoints = np.append(xMidPoints, [colMidPoint], axis=0)
                I_extractCorrectedVisualiser = cv2.circle(I_extractCorrectedVisualiser, colMidPoint, radius=1, color=(0, 0, 255), thickness=-1)
            tmpColor = color
            tmpStart = point

        tmpStart = None
        for i in range(math.ceil(cornerToYLen)):
            point = np.rint(midCornerPoint + normY * i).astype(int)
            color = I_extractCorrected[point[1],point[0]]
            if color == tmpColor:
                tmpPrev = point
                continue
            if tmpStart is not None:
                colMidPoint = np.rint((tmpPrev + tmpStart) / 2).astype(int)
                yMidPoints = np.append(yMidPoints, [colMidPoint], axis=0)
                I_extractCorrectedVisualiser = cv2.circle(I_extractCorrectedVisualiser, colMidPoint, radius=1, color=(0, 0, 255), thickness=-1)
            tmpColor = color
            tmpStart = point


        colcount = (len(yMidPoints) if len(yMidPoints) > len(xMidPoints) else len(xMidPoints)) + 14
        result = np.zeros((colcount, colcount), dtype=np.uint8)

        if sobelMaskRowCount == sobelMaskColCount and sobelMaskColCount == colcount:
            indices = np.where(sobelMask==255)
            I_extractCorrectedMasked = cv2.cvtColor(I_extractCorrected, cv2.COLOR_GRAY2BGR)
            I_extractCorrectedMasked[indices[0], indices[1], :] = [255, 0, 255]
            cv2.imshow("I_extractCorrectedMasked", I_extractCorrectedMasked)
            tmpSobelMaskRows = np.append(sobelMaskRows, not sobelMaskRows[-1]) 
            tmpSobelMaskRows = np.where(tmpSobelMaskRows[:-1] != tmpSobelMaskRows[1:])[0]
            tmpSobelMaskCols = np.append(sobelMaskCols, not sobelMaskCols[-1]) 
            tmpSobelMaskCols = np.where(tmpSobelMaskCols[:-1] != tmpSobelMaskCols[1:])[0]
            for x in range(0, len(tmpSobelMaskRows), 2):
                for y in range(0, len(tmpSobelMaskCols), 2):
                    prevPointX = tmpSobelMaskRows[x - 1] if x > 0 else 0
                    prevPointY = tmpSobelMaskRows[y - 1] if y > 0 else 0
                    pointx = prevPointX + math.ceil((tmpSobelMaskRows[x] - prevPointX) / 2)
                    pointy = prevPointY + math.ceil((tmpSobelMaskRows[y] - prevPointY) / 2)
                    result[int(y/2), int(x/2)] = 0 if I_extractCorrected[pointy, pointx] == 255 else 255
                    I_extractCorrectedVisualiser = cv2.circle(I_extractCorrectedVisualiser, [pointy, pointx], radius=1, color=(0, 255, 0), thickness=-1)

        #if sobelMaskRowCount == sobelMaskColCount and sobelMaskColCount == colcount:
        #    indices = np.where(sobelMask==255)
        #    I_extractCorrectedMasked = cv2.cvtColor(I_extractCorrected, cv2.COLOR_GRAY2BGR)
        #    I_extractCorrectedMasked[indices[0], indices[1], :] = [255, 0, 255]
        #    tmpSobelMaskRows = np.append(sobelMaskRows, not sobelMaskRows[-1]) 
        #    tmpSobelMaskRows = np.where(tmpSobelMaskRows[:-1] != tmpSobelMaskRows[1:])[0][::2]
        #    tmpSobelMaskCols = np.append(sobelMaskCols, not sobelMaskCols[-1]) 
        #    tmpSobelMaskCols = np.where(tmpSobelMaskCols[:-1] != tmpSobelMaskCols[1:])[0][::2]
        #    for ix, x in enumerate(tmpSobelMaskRows):
        #        for iy, y in enumerate(tmpSobelMaskCols):
        #            result[ix, iy] = 0 if I_extractCorrected[x - 1, y - 1] == 255 else 255
        #            I_extractCorrectedVisualiser = cv2.circle(I_extractCorrectedVisualiser, [y-1, x-1], radius=1, color=(0, 255, 0), thickness=-1)
        
        cv2.imshow("I_extractCorrectedVisualiser", I_extractCorrectedVisualiser)
        cv2.imshow("I_extractCorrected", I_extractCorrected)
        #asdx = I_extractCorrected.shape[1] / colcount
        #asdy = I_extractCorrected.shape[0] / colcount
        #asdc = np.rint((FIPs[0][1][1] + FIPs[0][2][1]) / 2).astype(int)
        #for y in range(colcount):
        #    for x in range(colcount):
        #        point = np.rint(asdc + [j*asdx,i*asdy]).astype(int)
        #        result[j,i] = 0 if I_extractCorrected[point[1],point[0]] == 255 else 255
        #        if I_extractCorrected[point[1],point[0]] == 255:
        #            I_extractCorrectedVisualiser = cv2.circle(I_extractCorrectedVisualiser, [point[1],point[0]], radius=2, color=(0, 0, 255), thickness=-1)
        #        else:
        #            I_extractCorrectedVisualiser = cv2.circle(I_extractCorrectedVisualiser, [point[1],point[0]], radius=2, color=(0, 255, 0), thickness=-1)
        #blockSize = I_extractCorrected.shape[0] / colcount
        #startPoint = np.rint((FIPs[0][1][1] + FIPs[0][2][1]) / 2).astype(int)
        #startPoint = [startPoint[1], startPoint[0]]
        #I_extractCorrectedVisualiser = cv2.circle(I_extractCorrectedVisualiser, FIPs[0][1][1], radius=1, color=(255, 255, 0), thickness=-1)
        #I_extractCorrectedVisualiser = cv2.circle(I_extractCorrectedVisualiser, FIPs[0][2][1], radius=1, color=(255, 255, 0), thickness=-1)
        #for y in range(colcount):
        #    for x in range(colcount):
        #        #point = np.rint(startPoint + [y * blockSize,x * blockSize]).astype(int)
        #        point = np.rint(startPoint + y * blockSize * normY + x * blockSize * normX).astype(int)
        #        result[y,x] = 0 if I_extractCorrected[point[0],point[1]] == 255 else 255
        #        if I_extractCorrected[point[0],point[1]] == 255:
        #            I_extractCorrectedVisualiser = cv2.circle(I_extractCorrectedVisualiser, [point[1],point[0]], radius=1, color=(0, 0, 255), thickness=-1)
        #        else:
        #            I_extractCorrectedVisualiser = cv2.circle(I_extractCorrectedVisualiser, [point[1],point[0]], radius=1, color=(0, 255, 0), thickness=-1)
        #cv2.imshow("I_extractCorrectedVisualiser", I_extractCorrectedVisualiser)
        #cv2.imshow("I_extractCorrected", I_extractCorrected)
        layers.append(I_visualizer)

        I_regenerated = cv2.copyMakeBorder(np.kron(result, np.ones((5,5))), 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=(255,255,255))
        I_regenerated = cv2.normalize(I_regenerated, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        cv2.imshow("I_regenerated", I_regenerated)

        result = pyzbar.decode(I_regenerated)
       
        if any(result):
            print(result[0].data)

        #decoder = QRMatrixDecoder(np.int0(result))
        #print(decoder.decode())
    elif lenFIPs > 3:
        print(lenFIPs)

    return layers

def transformFIPs(FIPs, M):

    FIPs = [[pointPerspectiveTransform(FIP[0], M), [pointPerspectiveTransform(innerCorner, M) for innerCorner in FIP[1]], [pointPerspectiveTransform(outerCorner, M) for outerCorner in FIP[2]]] for FIP in FIPs]
    return FIPs

def pointPerspectiveTransform(p, M):
    px = (M[0][0]*p[0] + M[0][1]*p[1] + M[0][2]) / ((M[2][0]*p[0] + M[2][1]*p[1] + M[2][2]))
    py = (M[1][0]*p[0] + M[1][1]*p[1] + M[1][2]) / ((M[2][0]*p[0] + M[2][1]*p[1] + M[2][2]))
    return np.rint(np.array([px, py])).astype(int)

def line_intersection(line1, line2):
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
        raise Exception('lines do not intersect')

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return np.rint(np.array([x, y])).astype(int)

def calculateSobelScore(I):
    
    filler = np.full(I.shape[1], 255)
    I = cv2.GaussianBlur(I, (3, 3), 0)

    I_vertical = cv2.Sobel(I, cv2.CV_16S, 1, 0, ksize=1)
    I_vertical = cv2.convertScaleAbs(I_vertical)
    _, I_vertical = cv2.threshold(I_vertical, 180, 255, cv2.THRESH_BINARY)
    cv2.imshow("sobelScoreYOriginal", I_vertical)
    #vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,50))
    #I_vertical = cv2.dilate(I_vertical, vertical_kernel, iterations=9)
    #zerosY = np.count_nonzero(np.sum(I_vertical, axis=0) == 0)
    cols = np.any(I_vertical, axis = 0)
    for i, state in enumerate(cols):
        if state:
            I_vertical[:,i] = filler
    zerosY = np.count_nonzero(cols == False)
    
    if __debug__:
        I_vertical = cv2.cvtColor(I_vertical, cv2.COLOR_GRAY2BGR)
        I_vertical = cv2.putText(I_vertical, str(zerosY), (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, 2)
        cv2.imshow("sobelScoreY", I_vertical)

    I_horizontal = cv2.Sobel(I, cv2.CV_16S, 0, 1, ksize=1)
    I_horizontal = cv2.convertScaleAbs(I_horizontal)
    _, I_horizontal = cv2.threshold(I_horizontal, 180, 255, cv2.THRESH_BINARY)
    cv2.imshow("sobelScoreXOriginal", I_horizontal)
    
    #horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50,1))
    #I_horizontal = cv2.dilate(I_horizontal, horizontal_kernel, iterations=9)
    #zerosX = np.count_nonzero(np.sum(I_horizontal, axis=1) == 0)
    rows = np.any(I_horizontal, axis = 1)
    for i, state in enumerate(rows):
        if state:
            I_horizontal[i,:] = filler
    zerosX = np.count_nonzero(rows == False)
    
    if __debug__:
        I_horizontal = cv2.cvtColor(I_horizontal, cv2.COLOR_GRAY2BGR)
        I_horizontal = cv2.putText(I_horizontal, str(zerosX), (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, 2)
        cv2.imshow("sobelScoreX", I_horizontal)

    return zerosX + zerosY

def getSobelMask(I):
    fillerx = np.full(I.shape[1], 255)
    fillery = np.full(I.shape[0], 255)
    I = cv2.GaussianBlur(I, (3, 3), 0)

    I_vertical = cv2.Sobel(I, cv2.CV_16S, 1, 0, ksize=1)
    I_vertical = cv2.convertScaleAbs(I_vertical)
    _, I_vertical = cv2.threshold(I_vertical, 180, 255, cv2.THRESH_BINARY)

    cols = np.any(I_vertical, axis = 0)
    if cols[len(cols)-1] == True:
        i = len(cols)-1
        while i > 0 and cols[i] == True:
            cols[i] = False
            i -= 1

    colCount = 0
    countState = True
    for i, state in enumerate(cols):
        if state:
            I_vertical[:,i] = fillerx
            countState = True
        elif countState:
            colCount += 1
            countState = False

    I_horizontal = cv2.Sobel(I, cv2.CV_16S, 0, 1, ksize=1)
    I_horizontal = cv2.convertScaleAbs(I_horizontal)
    _, I_horizontal = cv2.threshold(I_horizontal, 180, 255, cv2.THRESH_BINARY)

    rows = np.any(I_horizontal, axis = 1)    
    if rows[len(rows)-1] == True:
        i = len(rows)-1
        while i > 0 and rows[i] == True:
            rows[i] = False
            i -= 1

    rowCount = 0
    countState = True
    for i, state in enumerate(rows):
        if state:
            I_horizontal[i,:] = fillery
            countState = True
        elif countState:
            rowCount += 1
            countState = False

    return np.maximum(I_vertical, I_horizontal), rows, cols, rowCount, colCount

def calculateqr(x):
    image = cv2.getTrackbarPos('image',"options")
    layer = cv2.getTrackbarPos('layer',"options")
    scale = cv2.getTrackbarPos('scale',"options")
    angle = cv2.getTrackbarPos('angle',"options")
    pre_blur_ksize = cv2.getTrackbarPos('pre_blur_ksize',"options")
    ksize = cv2.getTrackbarPos('ksize',"options")
    c = cv2.getTrackbarPos('c',"options")
    erosionIterations = cv2.getTrackbarPos('erosionIterations',"options")
    x = cv2.getTrackbarPos('x',"options")
    y = cv2.getTrackbarPos('y',"options")
    dislocRange = cv2.getTrackbarPos('dislocRange',"options")

    images = glob.glob("2d_barcode\\*.*")

    I_original = cv2.imread(images[image])

    layers = findQrcode(I_original, pre_blur_ksize, scale, angle, ksize, c, erosionIterations, x, y, dislocRange)
    cv2.imshow("5", layers[layer if layer < len(layers) else len(layers) - 1])

def qrcode():
    cv2.namedWindow("options", cv2.WINDOW_GUI_EXPANDED)
    cv2.resizeWindow("options", 1800, 400)
    cv2.createTrackbar("image", "options", 0,208, calculateqr)
    cv2.createTrackbar("x", "options", 0,100, calculateqr)
    cv2.setTrackbarMin('x', 'options', -100)
    cv2.createTrackbar("y", "options", 0,100, calculateqr)
    cv2.setTrackbarMin('y', 'options', -100)
    cv2.createTrackbar("layer", "options", 13,13, calculateqr)
    cv2.createTrackbar("scale", "options", 100,100, calculateqr)
    cv2.createTrackbar("angle", "options", 0,360, calculateqr)
    cv2.createTrackbar("pre_blur_ksize", "options", 0, 20, calculateqr)
    cv2.createTrackbar("ksize", "options", 199, 200, calculateqr)
    cv2.createTrackbar("c", "options", 17, 100, calculateqr)
    cv2.createTrackbar("erosionIterations", "options", 0, 50, calculateqr)
    cv2.createTrackbar("dislocRange", "options", 5, 50, calculateqr)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

qrcode()

