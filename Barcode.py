import cv2
import numpy as np

class Barcode(object):
    def __init__(self, inputImage,  scalePercent = 100, rotationInDegrees = 0, blurKernelSize = 9, erosionIterations = 14, dilationIterations = 16, morfRectH = 21, morfRectW = 7, makeProcessingLayers = False):
        self.__inputImage = inputImage
        self.__scalePercent = scalePercent
        self.__rotationInDegrees = rotationInDegrees
        self.__blurKernelSize = blurKernelSize
        self.__erosionIterations = erosionIterations
        self.__dilationIterations = dilationIterations
        self.__makeProcessingLayers = makeProcessingLayers
        self.__morfRectH = morfRectH
        self.__morfRectW = morfRectW


        self.processingLayers = None
        if makeProcessingLayers:
            self.processingLayers = []
            self.processingLayers.append(inputImage)

        self.__preprocessedImage = None
        self.__preprocessImage()

    def __preprocessImage(self):
        underPreprocessImage = self.__inputImage
        # Convert the image to grayscale because the colour is not playing any role in QR code detection
        underPreprocessImage = cv2.cvtColor(underPreprocessImage, cv2.COLOR_BGR2GRAY)
        self.__dealWithProcessingLayers(underPreprocessImage)

        # Resize the image if needed
        if self.__scalePercent != 100:
            underPreprocessImage = cv2.resize(underPreprocessImage, (int(underPreprocessImage.shape[1] * self.__scalePercent / 100), int(underPreprocessImage.shape[0] * self.__scalePercent / 100)))
            self.__dealWithProcessingLayers(underPreprocessImage)
        # Rotate the image if needed
        if self.__rotationInDegrees != 0:
            imageCenter = tuple(np.array(underPreprocessImage.shape[1::-1]) / 2)
            rotationMatrix = cv2.getRotationMatrix2D(imageCenter, self.__rotationInDegrees, 1.0)
            underPreprocessImage = cv2.warpAffine(underPreprocessImage, rotationMatrix, underPreprocessImage.shape[1::-1], flags=cv2.INTER_LINEAR)
            self.__dealWithProcessingLayers(underPreprocessImage)

        # Blur the image if needed
        if self.__blurKernelSize != 0:
            underPreprocessImage = cv2.GaussianBlur(underPreprocessImage, (self.__blurKernelSize, self.__blurKernelSize), 0)
            self.__dealWithProcessingLayers(underPreprocessImage)

        underPreprocessImage = cv2.Laplacian(underPreprocessImage, cv2.CV_16U, ksize=9)
        self.__dealWithProcessingLayers(underPreprocessImage)

        _,underPreprocessImage = cv2.threshold(underPreprocessImage, 65535-10000, 65535, cv2.THRESH_BINARY)
        self.__dealWithProcessingLayers(underPreprocessImage)

        underPreprocessImage = np.uint8(underPreprocessImage)
        self.__dealWithProcessingLayers(underPreprocessImage)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 7))
        underPreprocessImage = cv2.morphologyEx(underPreprocessImage, cv2.MORPH_CLOSE, kernel)
        self.__dealWithProcessingLayers(underPreprocessImage)

        underPreprocessImage = cv2.erode(underPreprocessImage, None, iterations = self.__erosionIterations)
        self.__dealWithProcessingLayers(underPreprocessImage)

        underPreprocessImage = cv2.dilate(underPreprocessImage, None, iterations = self.__dilationIterations)
        self.__dealWithProcessingLayers(underPreprocessImage)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (self.__morfRectH, self.__morfRectW))
        underPreprocessImage = cv2.morphologyEx(underPreprocessImage, cv2.MORPH_CLOSE, kernel)
        self.__dealWithProcessingLayers(underPreprocessImage)

        self.__preprocessedImage = underPreprocessImage

    def readBarcode(self):
        if self.__preprocessedImage is None:
            raise Exception("Image has not yet been preprocessed.")

        contours, _ = cv2.findContours(self.__preprocessedImage, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key = cv2.contourArea, reverse = True)
        imageCenter = tuple(np.int32(np.array(self.__inputImage.shape[1::-1]) / 2))

        boxes = []
        for contour in contours:
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            box = box / (self.__scalePercent / 100)
            box = self.__rotateBoundary(imageCenter, box, self.__rotationInDegrees)
            box = np.int0(box)
            boxes.append(box)
   

        if self.__makeProcessingLayers:
            visualizer = self.__inputImage.copy()
            if len(boxes) > 0:
                cv2.drawContours(visualizer, boxes[1:], -1, (0,255,0), 3)
                cv2.drawContours(visualizer, [boxes[0]], -1, (0,0,255), 3)
            self.__dealWithProcessingLayers(visualizer)

        for box in boxes:
            extracted = self.__extractBarcode(box)
            barcode = self.__readBarcode(extracted)
            if barcode:
                return barcode

        return None

    def __readBarcode(self, barcode):
        for i, line in enumerate(barcode):
            barcodeValue = self.__readBarcodeLine(line)
            if barcodeValue:
                if self.__makeProcessingLayers:
                    visualizer = cv2.cvtColor(barcode, cv2.COLOR_GRAY2BGR)
                    cv2.line(visualizer, (0, i), (len(line), i), (0,0,255), 1)
                    self.__dealWithProcessingLayers(visualizer)
                return barcodeValue
        return None

    def __readBarcodeLine(self, line):
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
            barsWindow = bars[0+i:59+i]
            barcodeWidth = sum(barsWindow)
            averageBarWidth = barcodeWidth / 95
    
            barsWindow = np.int0(np.round(barsWindow / averageBarWidth))
            barcode = self.__decodeBarcode(barsWindow)
            if barcode:
                return barcode
        return None

    def __decodeBarcode(self, bars):
        if sum(bars[:3]) + sum(bars[27:32]) + sum(bars[-3:]) != 11:
            return

        bars = np.append(bars[3:27], bars[32:-3])
        if (bars[0] + bars[2]) % 2 != 0:
            bars = np.flip(bars)
        bars = np.array_split(bars, 12)
    
        if np.any([sum(b) != 7 for b in bars]):
            return

        firstDigitDictionary = {(1,1,1,1,1,1): 0, (1,1,0,1,0,0): 1, (1,1,0,0,1,0): 2, (1,1,0,0,0,1): 3, (1,0,1,1,0,0): 4, (1,0,0,1,1,0): 5, (1,0,0,0,1,1): 6, (1,0,1,0,1,0): 7, (1,0,1,0,0,1): 8, (1,0,0,1,0,1): 9}
        valueDictionary = {
            (3,2,1,1): 0, (2,2,2,1): 1, (2,1,2,2): 2, (1,4,1,1): 3, (1,1,3,2): 4, (1,2,3,1): 5, (1,1,1,4): 6, (1,3,1,2): 7, (1,2,1,3): 8, (3,1,1,2): 9
            ,(1,1,2,3): 0, (1,2,2,2): 1, (2,2,1,2): 2, (1,1,4,1): 3, (2,3,1,1): 4, (1,3,2,1): 5, (4,1,1,1): 6, (2,1,3,1): 7, (3,1,2,1): 8, (2,1,1,3): 9
            }
        barcodeValue = [valueDictionary.get(tuple(i), None) for i in bars]
        if None in barcodeValue:
            return

        firstDigit = firstDigitDictionary.get(tuple((b[1] + b[3]) % 2 for b in bars[:6]), None)

        if firstDigit is None:
            return

        barcodeValue.insert(0, firstDigit)

        checkDigit = (sum(barcodeValue[:-1:2]) + sum(barcodeValue[1::2])*3) % 10
        if barcodeValue[12] != (checkDigit if checkDigit == 0 else 10 - checkDigit):
            return

        return barcodeValue

    def __dealWithProcessingLayers(self, layer):
        if self.__makeProcessingLayers:
            self.processingLayers.append(np.copy(layer))

    def __rotatePoint(self, origin, point, angle):
        ox, oy = origin
        px, py = point

        qx = ox + np.cos(np.radians(angle)) * (px - ox) - np.sin(np.radians(angle)) * (py - oy)
        qy = oy + np.sin(np.radians(angle)) * (px - ox) + np.cos(np.radians(angle)) * (py - oy)
        return qx, qy

    def __rotateBoundary(self, origin, boundary, angle):
        for i, point in enumerate(boundary):
            boundary[i] = self.__rotatePoint(origin, point, angle)
        return boundary

    def __extractBarcode(self, box):
        barcode, _ = self.__fourPointTransform(self.__inputImage, np.float32(box))
        barcode = cv2.rotate(barcode, cv2.ROTATE_90_COUNTERCLOCKWISE) if barcode.shape[0] > barcode.shape[1] else barcode
        barcode = cv2.cvtColor(barcode, cv2.COLOR_BGR2GRAY)
        barcode = cv2.adaptiveThreshold(barcode, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,15,2)
        return barcode

    def __fourPointTransform(self, image, rect):
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
            [0, maxHeight - 1]], dtype = np.float32)
        transformationMatrix = cv2.getPerspectiveTransform(rect, dst)

        warpedImage = cv2.warpPerspective(image, transformationMatrix, (maxWidth, maxHeight))
        return warpedImage, transformationMatrix