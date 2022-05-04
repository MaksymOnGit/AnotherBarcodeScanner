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
def prepareDataset():
    file_name = "1d_barcode_hough.zip"
    if not exists(file_name):
        r = requests.get("http://artelab.dista.uninsubria.it/downloads/datasets/barcode/hough_barcode_1d/" + file_name, allow_redirects=True)
        open(file_name, 'wb').write(r.content)
    if not exists("1d_barcode_hough"):
        with zipfile.ZipFile(file_name, 'r') as zip_ref:
            zip_ref.extractall()

def four_point_transform(image, rect):
        (tl, tr, br, bl) = rect
        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))
        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))
        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]], dtype = "float32")
        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
        return warped

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
    I_barcode = cv2.cvtColor(I_barcode, cv2.COLOR_BGR2GRAY)
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

def findQrcode(I, pre_blur_ksize, scale, angle, ksize, c, erosionIterations):
    pre_blur_ksize = -1 if pre_blur_ksize == 1 else pre_blur_ksize

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

        
        I_visualizer = cv2.line(I_visualizer, np.int0((FIPs[0][1][0] + FIPs[0][2][0]) / 2), np.int0((FIPs[1][1][0] + FIPs[1][2][0]) / 2), color=(0, 255, 0), thickness=1)
        I_visualizer = cv2.line(I_visualizer, np.int0((FIPs[0][1][0] + FIPs[0][2][0]) / 2), np.int0((FIPs[2][1][0] + FIPs[2][2][0]) / 2), color=(0, 255, 0), thickness=1)
        #x = FIPs[1] - Corner
        #y = FIPs[0] - Corner
        #lenx = np.linalg.norm(x)
        #leny = np.linalg.norm(y)
        #normx = x / lenx
        #normy = y / leny
        
        
        #point = np.int0(Corner + normx * 0)
        #while I[point[1],point[0]] > 0:
        #    I[point[1],point[0]]

        
        #start = None
        #end = None
        #middle = None
        #i = 0
        #point = np.int0(Corner + normx * i)

        #while I[point[1],point[0]] > 0:
        #    point = np.int0(Corner + normx * i)
        #    i += 1

        #while I[point[1],point[0]] == 0:
        #    point = np.int0(Corner + normx * i)
        #    i += 1

        #start = point

        #while I[point[1],point[0]] > 0:
        #    point = np.int0(Corner + normx * i)
        #    i += 1

        #end = point

        #i = 0
        #middle = np.int0((end + start) / 2)
        #point = middle
        #I_visualizer = cv2.circle(I_visualizer, point, radius=2, color=(255, 0, 0), thickness=-1)

        #while I[point[1],point[0]] > 0:
        #    point = np.int0(middle + normy * i)
        #    i += 1

        #middle = point
        

        #I_visualizer = cv2.line(I_visualizer, np.int0(middle), np.int0(middle + (normy * leny)), color=(0, 0, 255), thickness=2)

        #asdx = lenx / 22
        #asdy = leny / 22
        #for i in range(-3,26):
        #    for j in range(-3,26):
        #        I_visualizer = cv2.circle(I_visualizer, np.int0(np.around(Corner + (i * normx * asdx) + (j * normy * asdy))), radius=2, color=(0, 0, 255), thickness=-1)
        layers.append(I_visualizer)
    elif lenFIPs > 3:
        print(lenFIPs)

    return layers

def calculateqr(x):
    image = cv2.getTrackbarPos('image',"options")
    layer = cv2.getTrackbarPos('layer',"options")
    scale = cv2.getTrackbarPos('scale',"options")
    angle = cv2.getTrackbarPos('angle',"options")
    pre_blur_ksize = cv2.getTrackbarPos('pre_blur_ksize',"options")
    ksize = cv2.getTrackbarPos('ksize',"options")
    c = cv2.getTrackbarPos('c',"options")
    erosionIterations = cv2.getTrackbarPos('erosionIterations',"options")

    images = glob.glob("QR*.*")

    I_original = cv2.imread(images[image])

    layers = findQrcode(I_original, pre_blur_ksize, scale, angle, ksize, c, erosionIterations)
    cv2.imshow("5", layers[layer if layer < len(layers) else len(layers) - 1])

def qrcode():
    cv2.namedWindow("options", cv2.WINDOW_GUI_EXPANDED)
    cv2.resizeWindow("options", 1800, 400)
    cv2.createTrackbar("image", "options", 0,30, calculateqr)
    cv2.createTrackbar("layer", "options", 13,13, calculateqr)
    cv2.createTrackbar("scale", "options", 100,100, calculateqr)
    cv2.createTrackbar("angle", "options", 0,360, calculateqr)
    cv2.createTrackbar("pre_blur_ksize", "options", 0, 20, calculateqr)
    cv2.createTrackbar("ksize", "options", 199, 200, calculateqr)
    cv2.createTrackbar("c", "options", 17, 100, calculateqr)
    cv2.createTrackbar("erosionIterations", "options", 0, 50, calculateqr)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

qrcode()

