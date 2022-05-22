import cv2
import numpy as np
import math
import glob
from os.path import exists, isdir
from pyzbar.pyzbar import decode
import requests
import zipfile
import concurrent.futures
import sys


def prepareDataset():
    file_name = "1d_barcode_hough.zip"
    if not exists(file_name):
        r = requests.get("http://artelab.dista.uninsubria.it/downloads/datasets/barcode/hough_barcode_1d/" + file_name, allow_redirects=True)
        open(file_name, 'wb').write(r.content)
    if not exists("1d_barcode_hough"):
        with zipfile.ZipFile(file_name, 'r') as zip_ref:
            zip_ref.extractall()

def barThread(image, sobel_ksize, pre_blur_ksize, post_blur_ksize, thresh, scale, angle, morfRectH, morfRectW, erosionIterations, dilationIterations):
    print(image)
    I_original = cv2.imread(image)
    from Barcode import Barcode
    barcode = Barcode(I_original, scalePercent=scale, rotationInDegrees=angle, blurKernelSize=pre_blur_ksize, erosionIterations=erosionIterations, dilationIterations=dilationIterations, makeProcessingLayers=True)
    barcodeVal = barcode.readBarcode()
    return barcodeVal

def qrThread(image, pre_blur_ksize, scale, angle, ksize, c, erosionIterations, x, y, dislocRange):
    print(image)
    I_original = cv2.imread(image)
    layers, result = findQrcode(I_original, pre_blur_ksize, scale, angle, ksize, c, erosionIterations, x, y, dislocRange)
    if result:
        print(result)
        return result

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
        futures = [executor.submit(barThread, image, sobel_ksize, pre_blur_ksize, post_blur_ksize, thresh, scale, angle, morfRectH, morfRectW, erosionIterations, dilationIterations) for image in images]
        barcodes = np.array([f.result() for f in futures])

    count = len(barcodes)
    succ = np.count_nonzero(barcodes)
    print(f'{succ}/{count} ({round(succ/count*100, 2)}%)')

def benchmarkQr(x):
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

    barcodes = np.array([])

    with concurrent.futures.ThreadPoolExecutor(8) as executor:
        futures = [executor.submit(qrThread, image, pre_blur_ksize, scale, angle, ksize, c, erosionIterations, x, y, dislocRange) for image in images]
        barcodes = np.array([f.result() for f in futures])

    count = len(barcodes)
    succ = np.count_nonzero(barcodes)
    print(f'{succ}/{count} ({round(succ/count*100, 2)}%)')




def qrcode():
    cv2.namedWindow("options", cv2.WINDOW_GUI_EXPANDED)
    cv2.resizeWindow("options", 1800, 400)
    cv2.createTrackbar("image", "options", 0,208, calculateqr)
    cv2.createTrackbar("x", "options", 1,100, calculateqr)
    cv2.setTrackbarMin('x', 'options', -100)
    cv2.createTrackbar("y", "options", -1,100, calculateqr)
    cv2.setTrackbarMin('y', 'options', -100)
    cv2.createTrackbar("layer", "options", 13,13, calculateqr)
    cv2.createTrackbar("scale", "options", 100,100, calculateqr)
    cv2.createTrackbar("angle", "options", 0,360, calculateqr)
    cv2.createTrackbar("pre_blur_ksize", "options", 0, 20, calculateqr)
    cv2.createTrackbar("ksize", "options", 199, 200, calculateqr)
    cv2.createTrackbar("c", "options", 17, 100, calculateqr)
    cv2.createTrackbar("erosionIterations", "options", 0, 50, calculateqr)
    cv2.createTrackbar("dislocRange", "options", 5, 50, calculateqr)
    cv2.createTrackbar("benchmark", "options", 0, 2, benchmarkQr)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

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

    layers, _ = findQrcode(I_original, pre_blur_ksize, scale, angle, ksize, c, erosionIterations, x, y, dislocRange)
    cv2.imshow("5", layers[layer if layer < len(layers) else len(layers) - 1])

def findQrcode(I, pre_blur_ksize, scale, angle, ksize, adaptiveThresholdConstant, erosionIterations, xdisloc, ydisloc, dislocRange):

    from QrCode import QrCode
    qrCode = QrCode(I, makeProcessingLayers = True, scalePercent=scale, rotationInDegrees=angle, blurKernelSize=pre_blur_ksize, thresholdKernelSize=ksize, adaptiveThresholdConstant=adaptiveThresholdConstant)
    detectSuccess, detectResult = qrCode.tryDetect()
    if not detectSuccess or not detectResult:
        return qrCode.processingLayers, None
    
    decodeSuccess, detectResult = qrCode.tryDecode(dislocRange, xdisloc, ydisloc)
    if not decodeSuccess or not detectResult:
        return qrCode.processingLayers, None

    print(detectResult)
    return qrCode.processingLayers, detectResult


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

    from Barcode import Barcode
    barcode = Barcode(I_original, scalePercent=scale, rotationInDegrees=angle, blurKernelSize=pre_blur_ksize, erosionIterations=erosionIterations, dilationIterations=dilationIterations, makeProcessingLayers=True)
    barcodeVal = barcode.readBarcode()
    layers = barcode.processingLayers

    cv2.imshow("5", layers[layer if layer < len(layers) else len(layers) - 1])

    if(barcodeVal):
        print(barcodeVal)

def scanFolder(folder):
    if folder[:-1] != "\\":
        folder += "\\"
    images = glob.glob(folder + '*.png') + glob.glob(folder + '*.jpg') + glob.glob(folder + '*.gif')
    from Barcode import Barcode
    from QrCode import QrCode

    for imagePath in images:
        image = cv2.imread(imagePath)
        qrCode = QrCode(image)
        detectSuccess, detectResult = qrCode.tryDetect()
        if detectSuccess and detectResult:
            decodeSuccess, decodeResult = qrCode.tryDecode()
            if decodeSuccess and decodeResult:
                print(imagePath + "\t" + str(decodeResult))
                next

        barcode = Barcode(image)
        barcodeVal = barcode.readBarcode()
        if barcodeVal:
            print(imagePath + "\t" + str(barcodeVal))
            next

def main():
    args = sys.argv[1:]
    if len(args) == 0:
        print("No arguments were given")
        return

    if args[0] == "debug":
        if args[1] == "bar":
            barcode()
        elif args[1] == "qr":
            qrcode()
    elif isdir(args[0]):
        scanFolder(args[0])

if __name__ == "__main__":
    main()