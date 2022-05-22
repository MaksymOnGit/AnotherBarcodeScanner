from platform import version
import cv2
import numpy as np

class QrCode(object):
    def __init__(self, inputImage, scalePercent = 100, rotationInDegrees = 0, blurKernelSize = 0, adaptiveThreshold = True, threshold = 225, thresholdKernelSize = 199, adaptiveThresholdConstant = 17, makeProcessingLayers = False):
        self.__inputImage = inputImage
        self.__blurKernelSize = blurKernelSize
        self.__adaptiveThreshold = adaptiveThreshold
        self.__threshold = threshold # used only when adaptiveThreshold is False
        self.__scalePercent = scalePercent
        self.__rotationInDegrees = rotationInDegrees
        self.__thresholdKernelSize = thresholdKernelSize
        self.__adaptiveThresholdConstant = adaptiveThresholdConstant
        self.__makeProcessingLayers = makeProcessingLayers
        self.processingLayers = None
        if makeProcessingLayers:
            self.processingLayers = []
            self.processingLayers.append(inputImage)

        self.__preprocessedImage = None

        self.__preprocessImage()

        self.qrCodeDetected = None

        self.__finderPatterns = None
        self.__finderPatternsTransformed = None
        self.__fourthCorner = None


    def __preprocessImage(self):
        underPreprocessImage = self.__inputImage

        # Convert the image to grayscale because the colour is not playing any role in QR code detection
        underPreprocessImage = cv2.cvtColor(underPreprocessImage, cv2.COLOR_BGR2GRAY)
        self.__dealWithProcessingLayers(underPreprocessImage)

        # Resize the image if needed
        if self.__scalePercent != 100:
            underPreprocessImage = cv2.resize(underPreprocessImage, (int(underPreprocessImage.shape[1] * self.__scalePercent / 100), int(underPreprocessImage.shape[0] * self.__scalePercent / 100)))
            self.__dealWithProcessingLayers(underPreprocessImage)

        # A white border for the image in case the input barcode image has no quiet zone
        underPreprocessImage = cv2.copyMakeBorder(underPreprocessImage, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=(255,255,255))
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

        if self.__adaptiveThreshold:
            underPreprocessImage = cv2.adaptiveThreshold(underPreprocessImage, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, self.__thresholdKernelSize, self.__adaptiveThresholdConstant)
        else:
            underPreprocessImage = cv2.threshold(underPreprocessImage, self.__threshold, 255, cv2.THRESH_BINARY_INV)[1]

        self.__dealWithProcessingLayers(underPreprocessImage)

        self.__preprocessedImage = underPreprocessImage

    def tryDetect(self):
        try:
            result = self.detect()
            return True, result
        except Exception as e:
            print(e)
        return False, None

    def detect(self):
        if self.__preprocessedImage is None:
            raise Exception("Image has not yet been preprocessed.")

        contours, hierarchy = cv2.findContours(self.__preprocessedImage, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        hierarchy = hierarchy[0]

        if self.__makeProcessingLayers:
            visualizer = cv2.cvtColor(self.__preprocessedImage, cv2.COLOR_GRAY2BGR)
            for i in contours:
                visualizer = cv2.drawContours(visualizer, [i], -1, list(np.random.random(size=3) * 256), 2,)
            self.__dealWithProcessingLayers(visualizer)

        finderPatterns = []
        for component in zip(contours, hierarchy):
            currentContour = component[0]

            # Hierarchy representation [Next, Previous, First_Child, Parent]
            currentHierarchy = component[1]

            # Skipping elements that are not lonely child and have their own child(s).
            if (
                (currentHierarchy[0] > 0)       # If not the only element in current group (Has next element)
                or (currentHierarchy[1] > 0)    # If not the only element in currentt group (Has previouse element)
                or (currentHierarchy[2] > 0)    # If a parent himself (Has a child element)
                or (currentHierarchy[3] < 0)    # If don't have a parent
            ): continue

            currentContoursParent = contours[currentHierarchy[3]]
            currentContoursParentsHierarhy = hierarchy[currentHierarchy[3]]

            # Skipping elements where parent of the child is not a lonely child.
            if (
                (currentContoursParentsHierarhy[0] > 0)    # If not the only element in current group (Has next element) 
                or (currentContoursParentsHierarhy[1] > 0) # If not the only element in currentt group (Has previouse element)
                or (currentContoursParentsHierarhy[3] < 0) # If don't have a parent
                ): continue

            # If the element succeeded in the previous checks that mean the element is the only one in the group and has no child 
            # and its parent is the only one in his group too and has a parent himself

            # The parent's area should be larger by roughly a third than the child is (Less than 8% deviation allowed)
            area = cv2.contourArea(currentContour)
            parentsArea = cv2.contourArea(currentContoursParent)
            if abs(0.30-area/parentsArea) > 0.08: continue

            # TODO: compare parent and grandparent areas

            # Getting the center point of the currentContour
            moments = cv2.moments(currentContour)
            x = moments["m10"] / moments["m00"]
            y = moments["m01"] / moments["m00"]
            center = np.rint([x,y]).astype(int)

            outerSquare = np.squeeze(cv2.approxPolyDP(contours[currentContoursParentsHierarhy[3]], 3, True), 1)
            innerSquare = np.squeeze(cv2.approxPolyDP(currentContoursParent, 3, True), 1)
            finderPatterns.append([center, innerSquare, outerSquare])

            
        if self.__makeProcessingLayers:
            visualizer = cv2.cvtColor(self.__preprocessedImage, cv2.COLOR_GRAY2BGR)

            for finderPattern in finderPatterns:
                cv2.circle(visualizer, finderPattern[0], radius=5, color=(0, 0, 255), thickness=-1)

                for corner in finderPattern[1]:
                        visualizer = cv2.circle(visualizer, corner, radius=2, color=(0, 0, 255), thickness=-1)

                for corner in finderPattern[2]:
                        visualizer = cv2.circle(visualizer, corner, radius=2, color=(0, 0, 255), thickness=-1)
                
            self.__dealWithProcessingLayers(visualizer)

        finderPatternCount = len(finderPatterns)

        if finderPatternCount < 3:
            self.qrCodeDetected = False
            return False

        if finderPatternCount > 3:
            raise Exception("Multiple finder patterns were found. Images with numerous QR codes are not supported at the moment.")
        
        self.qrCodeDetected = True
        self.__finderPatterns = finderPatterns
        return True

    def tryDecode(self, scanningRange = 5, offsetX = 1, offserY = -1):
        try:
            result = self.decode(scanningRange, offsetX, offserY)
            return True, result
        except Exception as e:
            pass
        return False, None
    def decode(self, scanningRange = 5, offsetX = 1, offserY = -1):
        if not self.qrCodeDetected:
            raise Exception("QR code has not been found.")

        self.__organiseFindingPatterns()
        self.__calucalteFourthCorner(scanningRange, offsetX, offserY)

        extractedQrCode, transformationMatrix = self.__fourPointTransform(self.__preprocessedImage, np.float32([self.__finderPatterns[0][2][1],self.__finderPatterns[1][2][1], self.__fourthCorner, self.__finderPatterns[2][2][1]]))
        self.__transformFinderPatternsPoints(transformationMatrix)
        _, extractedQrCode = cv2.threshold(extractedQrCode, 127, 255, cv2.THRESH_BINARY)
        
        sobelMask, sobelMaskRows, sobelMaskCols, sobelMaskRowCount, sobelMaskColCount = self.__getSobelMask(extractedQrCode)

        colcount, xMidPoints, yMidPoints, midCornerPointX, midCornerPointY, midCornerPoint = self.__countDataDimensions(extractedQrCode, self.__finderPatternsTransformed)

        if self.__makeProcessingLayers:
            sobelLayers = []
            sobelScore = self.__calculateSobelScore(colcount, extractedQrCode, sobelLayers)

            coloredExtract = cv2.cvtColor(extractedQrCode, cv2.COLOR_GRAY2BGR)
            coloredExtract = cv2.putText(coloredExtract, str(sobelScore), (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, 2)
            top = cv2.hconcat([coloredExtract, cv2.cvtColor(sobelLayers[2], cv2.COLOR_GRAY2BGR), sobelLayers[3]])
            left = cv2.vconcat([cv2.cvtColor(sobelLayers[0], cv2.COLOR_GRAY2BGR), sobelLayers[1]])
            right = np.full((extractedQrCode.shape[0]*2, extractedQrCode.shape[0]*2, 3), (255,255,255), dtype=np.uint8)
            collage = cv2.vconcat([top, cv2.hconcat([left, right])])
            self.__dealWithProcessingLayers(collage)


        result = np.zeros((colcount, colcount), dtype=np.uint8)

        if sobelMaskRowCount != sobelMaskColCount or sobelMaskColCount != colcount:
            raise Exception("Unable to decode QR code.")

        indices = np.where(sobelMask==255)
        tmpSobelMaskRows = np.append(sobelMaskRows, not sobelMaskRows[-1]) 
        tmpSobelMaskRows = np.where(tmpSobelMaskRows[:-1] != tmpSobelMaskRows[1:])[0]
        tmpSobelMaskCols = np.append(sobelMaskCols, not sobelMaskCols[-1]) 
        tmpSobelMaskCols = np.where(tmpSobelMaskCols[:-1] != tmpSobelMaskCols[1:])[0]

        extractedQrDotsVisualiser = cv2.cvtColor(extractedQrCode, cv2.COLOR_GRAY2BGR) if self.__makeProcessingLayers else None

        
        colcount = (len(yMidPoints) if len(yMidPoints) > len(xMidPoints) else len(xMidPoints)) + 14
        result = np.zeros((colcount, colcount), dtype=np.uint8)

        for x in range(0, len(tmpSobelMaskRows), 2):
            for y in range(0, len(tmpSobelMaskCols), 2):
                prevPointX = tmpSobelMaskRows[x - 1] if x > 0 else 0
                prevPointY = tmpSobelMaskRows[y - 1] if y > 0 else 0
                pointx = prevPointX + np.ceil((tmpSobelMaskRows[x] - prevPointX) / 2).astype(int)
                pointy = prevPointY + np.ceil((tmpSobelMaskRows[y] - prevPointY) / 2).astype(int)
                result[int(y/2), int(x/2)] = 0 if extractedQrCode[pointy, pointx] == 255 else 255
                if extractedQrDotsVisualiser is not None:
                    extractedQrDotsVisualiser = cv2.circle(extractedQrDotsVisualiser, [pointy, pointx], radius=1, color=(0, 255, 0), thickness=-1)

        
        regeneratedQrCode = cv2.copyMakeBorder(np.kron(result, np.ones((5,5))), 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=(255,255,255))
        regeneratedQrCode = cv2.normalize(regeneratedQrCode, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        if self.__makeProcessingLayers:
            extractedQrVisualiser = cv2.cvtColor(extractedQrCode, cv2.COLOR_GRAY2BGR)
            extractedQrVisualiser = cv2.circle(extractedQrVisualiser, self.__finderPatternsTransformed[0][0], radius=5, color=(255, 0, 0), thickness=-1)
            extractedQrVisualiser = cv2.circle(extractedQrVisualiser, self.__finderPatternsTransformed[1][0], radius=5, color=(0, 0, 255), thickness=-1)
            extractedQrVisualiser = cv2.circle(extractedQrVisualiser, self.__finderPatternsTransformed[2][0], radius=5, color=(0, 0, 255), thickness=-1)
            extractedQrVisualiser = cv2.line(extractedQrVisualiser, np.rint(midCornerPoint).astype(int), np.rint(midCornerPointX).astype(int), color=(0, 255, 0), thickness=1)
            extractedQrVisualiser = cv2.line(extractedQrVisualiser, np.rint(midCornerPoint).astype(int), np.rint(midCornerPointY).astype(int), color=(0, 255, 0), thickness=1)
            for point in xMidPoints:
                extractedQrVisualiser = cv2.circle(extractedQrVisualiser, point, radius=1, color=(0, 0, 255), thickness=-1)
            for point in yMidPoints:
                extractedQrVisualiser = cv2.circle(extractedQrVisualiser, point, radius=1, color=(0, 0, 255), thickness=-1)
        
            extractedMaskedQrVisualiser = cv2.cvtColor(extractedQrCode, cv2.COLOR_GRAY2BGR)
            extractedMaskedQrVisualiser[indices[0], indices[1], :] = [255, 0, 255]
            top = cv2.hconcat([extractedQrVisualiser, extractedMaskedQrVisualiser, extractedQrDotsVisualiser])
            bottom = cv2.copyMakeBorder(cv2.cvtColor(regeneratedQrCode, cv2.COLOR_GRAY2BGR), 0, 0, 0, top.shape[1] - regeneratedQrCode.shape[1], cv2.BORDER_CONSTANT, value=(255,255,255))
            self.__dealWithProcessingLayers(cv2.vconcat([top, bottom]))

            
        from pyzbar import pyzbar
        result = pyzbar.decode(regeneratedQrCode)
       
        if not any(result):
            raise Exception("Unable to decode QR code.")
        return result[0].data

    def __countDataDimensions(self, image, finderPatterns):
        midCornerPoint = (finderPatterns[0][1][0] + finderPatterns[0][2][0]) / 2
        midCornerPointX = (finderPatterns[1][1][0] + finderPatterns[1][2][0]) / 2
        midCornerPointY = (finderPatterns[2][1][0] + finderPatterns[2][2][0]) / 2

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
        for i in range(np.ceil(cornerToXLen).astype(int)):
            point = np.rint(midCornerPoint + normX * i).astype(int)
            color = image[point[1],point[0]]
            if color == tmpColor:
                tmpPrev = point
                continue
            if tmpStart is not None:
                colMidPoint = np.rint((tmpPrev + tmpStart) / 2).astype(int)
                xMidPoints = np.append(xMidPoints, [colMidPoint], axis=0)
            tmpColor = color
            tmpStart = point

        tmpStart = None
        for i in range(np.ceil(cornerToYLen).astype(int)):
            point = np.rint(midCornerPoint + normY * i).astype(int)
            color = image[point[1],point[0]]
            if color == tmpColor:
                tmpPrev = point
                continue
            if tmpStart is not None:
                colMidPoint = np.rint((tmpPrev + tmpStart) / 2).astype(int)
                yMidPoints = np.append(yMidPoints, [colMidPoint], axis=0)
            tmpColor = color
            tmpStart = point

        colcount = (len(yMidPoints) if len(yMidPoints) > len(xMidPoints) else len(xMidPoints)) + 14
        return colcount, xMidPoints, yMidPoints, midCornerPointX, midCornerPointY, midCornerPoint


    def __organiseFindingPatterns(self):
        # After this method the self.__finderPatterns will contain data in the following order

        # self.__finderPatterns[0]          is the top left finder pattern
        # self.__finderPatterns[0][1][0]    is the closest corner of the inner outline to the center of the QR code
        # self.__finderPatterns[0][1][1]    is the furthest corner of the inner outline from the center of the QR code
        # self.__finderPatterns[0][2][0]    is the closest corner of the outer outline to the center of the QR code
        # self.__finderPatterns[0][2][1]    is the furthest corner of the outer outline from the center of the QR code
        # self.__finderPatterns[1]          one of the remaining finder patterns
        # self.__finderPatterns[1][1][0]    is the closest corner of the inner outline to the center of the QR code
        # self.__finderPatterns[1][1][1]    is the furthest corner of the inner outline from the center of the QR code
        # self.__finderPatterns[1][1][2]    is the furthest corner of the inner outline from the center of the top left finder pattern
        # self.__finderPatterns[1][2][0]    is the closest corner of the outer outline to the center of the QR code
        # self.__finderPatterns[1][2][1]    is the furthest corner of the outer outline from the center of the QR code
        # self.__finderPatterns[1][2][2]    is the furthest corner of the inner outline from the center of the top left finder pattern
        # self.__finderPatterns[2]          one of the remaining finder patterns
        # self.__finderPatterns[2][1][0]    is the closest corner of the inner outline to the center of the QR code
        # self.__finderPatterns[2][1][1]    is the furthest corner of the inner outline from the center of the QR code
        # self.__finderPatterns[2][1][2]    is the furthest corner of the inner outline from the center of the top left finder pattern
        # self.__finderPatterns[2][2][0]    is the closest corner of the outer outline to the center of the QR code
        # self.__finderPatterns[2][2][1]    is the furthest corner of the outer outline from the center of the QR code
        # self.__finderPatterns[2][2][2]    is the furthest corner of the inner outline from the center of the top left finder pattern
        if self.__finderPatterns is None:
            raise Exception("__finderPatterns is None.")

        # Searching for top left finder patterns by calculating angles of triangle formed by FPs the largest is the needed one in all perspectives
        A = self.__finderPatterns[0][0]
        B = self.__finderPatterns[1][0]
        C = self.__finderPatterns[2][0]
        a = B-C
        b = A-C
        c = A-B
        Ad = np.arccos(np.dot(b,c)/(np.linalg.norm(b) * np.linalg.norm(c)))
        Cd = np.arccos(np.dot(b,a)/(np.linalg.norm(b) * np.linalg.norm(a)))
        Bd = np.pi - Ad - Cd
        # Getting the index of the pattern with the largest angle
        cornerIndex = max(range(3), key=[Ad, Bd, Cd].__getitem__)
        # Changing setting the position of the top left corner pattern to 0
        self.__finderPatterns[0], self.__finderPatterns[cornerIndex] = self.__finderPatterns[cornerIndex], self.__finderPatterns[0]

        Corner = self.__finderPatterns[0][0]
        A = self.__finderPatterns[1][0]
        B = self.__finderPatterns[2][0]
        
        # Center of the QR code calculated from midpoint of the line connecting bottom left and top right finder patterns
        Center = np.rint((A+B)/2).astype(int)
        
        # Determining the inner and outer corners of the finder pattern outlines by the distance from the center of the QR code
        for finderPattern in self.__finderPatterns:
            # Inner outline
            corners = finderPattern[1]
            distancesFromTheCenter = [np.linalg.norm(x - Center) for x in corners]

            # Index of the closest from the center, corner of the inner outline
            cornerIndex = min(range(len(distancesFromTheCenter)), key=distancesFromTheCenter.__getitem__)
            distancesFromTheCenter[cornerIndex], distancesFromTheCenter[0] = distancesFromTheCenter[0], distancesFromTheCenter[cornerIndex]
            # Setting it as the 0 index element
            corners[cornerIndex], corners[0] = np.copy(corners[0]), np.copy(corners[cornerIndex])
            
            # Index of the furthest from the center, corner of the inner outline
            cornerIndex = max(range(len(distancesFromTheCenter)), key=distancesFromTheCenter.__getitem__)
            # Setting it as the 1 index element
            corners[cornerIndex], corners[1] = np.copy(corners[1]), np.copy(corners[cornerIndex])

            # Outer outline
            corners = finderPattern[2]
            distancesFromTheCenter = [np.linalg.norm(x - Center) for x in corners]

            # Index of the closest from the center, corner of the outer outline
            cornerIndex = min(range(len(distancesFromTheCenter)), key=distancesFromTheCenter.__getitem__)
            distancesFromTheCenter[cornerIndex], distancesFromTheCenter[0] = distancesFromTheCenter[0], distancesFromTheCenter[cornerIndex]
            # Setting it as the 0 index element
            corners[cornerIndex], corners[0] = np.copy(corners[0]), np.copy(corners[cornerIndex])
            
            # Index of the furthest from the center, corner of the outer outline
            cornerIndex = max(range(len(distancesFromTheCenter)), key=distancesFromTheCenter.__getitem__)
            # Setting it as the 1 index element
            corners[cornerIndex], corners[1] = np.copy(corners[1]), np.copy(corners[cornerIndex])

            # Finding the furthest from the top left finder pattern, corner of the outer outline
            distancesFromTheCenter = [np.linalg.norm(x - self.__finderPatterns[0][0]) for x in corners[2:]]
            cornerIndex = max(range(len(distancesFromTheCenter)), key=distancesFromTheCenter.__getitem__)
            corners[cornerIndex + 2], corners[2] = np.copy(corners[2]), np.copy(corners[cornerIndex + 2])

        if self.__makeProcessingLayers:
            visualizer = cv2.cvtColor(self.__preprocessedImage, cv2.COLOR_GRAY2BGR)
            
            visualizer = cv2.circle(visualizer, Corner, radius=5, color=(255, 0, 0), thickness=-1)
            visualizer = cv2.circle(visualizer, A, radius=5, color=(0, 0, 255), thickness=-1)
            visualizer = cv2.circle(visualizer, B, radius=5, color=(0, 0, 255), thickness=-1)
            visualizer = cv2.circle(visualizer, Center, radius=5, color=(255, 0, 0), thickness=-1)
            visualizer = cv2.line(visualizer, Corner, A, color=(0, 255, 0), thickness=2)
            visualizer = cv2.line(visualizer, Corner, B, color=(0, 255, 0), thickness=2)
            
            visualizer = cv2.circle(visualizer, self.__finderPatterns[0][1][0], radius=2, color=(255, 0, 0), thickness=-1)
            visualizer = cv2.circle(visualizer, self.__finderPatterns[0][1][1], radius=2, color=(0, 255, 0), thickness=-1)
            visualizer = cv2.circle(visualizer, self.__finderPatterns[0][2][0], radius=2, color=(255, 0, 0), thickness=-1)
            visualizer = cv2.circle(visualizer, self.__finderPatterns[0][2][1], radius=2, color=(0, 255, 0), thickness=-1)
            visualizer = cv2.circle(visualizer, self.__finderPatterns[1][1][0], radius=2, color=(255, 0, 0), thickness=-1)
            visualizer = cv2.circle(visualizer, self.__finderPatterns[1][1][1], radius=2, color=(0, 255, 0), thickness=-1)
            visualizer = cv2.circle(visualizer, self.__finderPatterns[1][2][0], radius=2, color=(255, 0, 0), thickness=-1)
            visualizer = cv2.circle(visualizer, self.__finderPatterns[1][2][1], radius=2, color=(0, 255, 0), thickness=-1)
            visualizer = cv2.circle(visualizer, self.__finderPatterns[2][1][0], radius=2, color=(255, 0, 0), thickness=-1)
            visualizer = cv2.circle(visualizer, self.__finderPatterns[2][1][1], radius=2, color=(0, 255, 0), thickness=-1)
            visualizer = cv2.circle(visualizer, self.__finderPatterns[2][2][0], radius=2, color=(255, 0, 0), thickness=-1)
            visualizer = cv2.circle(visualizer, self.__finderPatterns[2][2][1], radius=2, color=(0, 255, 0), thickness=-1)
            visualizer = cv2.circle(visualizer, self.__finderPatterns[1][1][2], radius=2, color=(255, 255, 0), thickness=-1)
            visualizer = cv2.circle(visualizer, self.__finderPatterns[1][2][2], radius=2, color=(255, 255, 0), thickness=-1)
            visualizer = cv2.circle(visualizer, self.__finderPatterns[2][1][2], radius=2, color=(255, 255, 0), thickness=-1)
            visualizer = cv2.circle(visualizer, self.__finderPatterns[2][2][2], radius=2, color=(255, 255, 0), thickness=-1)
            
            self.__dealWithProcessingLayers(visualizer)


    def __precalucalteFourthCorner(self):
        lineOne = [self.__finderPatterns[1][2][1], self.__finderPatterns[1][2][2]]
        lineTwo = [self.__finderPatterns[2][2][1], self.__finderPatterns[2][2][2]]
        xDiff = (lineOne[0][0] - lineOne[1][0], lineTwo[0][0] - lineTwo[1][0])
        yDiff = (lineOne[0][1] - lineOne[1][1], lineTwo[0][1] - lineTwo[1][1])

        def det(a, b):
            return a[0] * b[1] - a[1] * b[0]

        div = det(xDiff, yDiff)
        if div == 0:
            raise Exception("Can't precalculate fourth corner.")

        d = (det(*lineOne), det(*lineTwo))
        x = det(d, xDiff) / div
        y = det(d, yDiff) / div
        return np.rint(np.array([x, y])).astype(int)

    def __calucalteFourthCorner(self, scanningRange = 5, offsetX = 0, offserY = 0):

        colcount, _, _, _, _, _ = self.__countDataDimensions(self.__preprocessedImage, self.__finderPatterns)

        fourthCorner = self.__precalucalteFourthCorner()

        fourthCorner[0] += offsetX
        fourthCorner[1] += offserY

        if self.__makeProcessingLayers:
            visualiser = self.processingLayers[-1]
            visualiser = cv2.circle(visualiser, fourthCorner, radius=2, color=(0, 0, 255), thickness=-1)

        scanLimit = np.linalg.norm(self.__finderPatterns[1][2][2] - self.__finderPatterns[2][2][2])
        rangeX = range(fourthCorner[0]-scanningRange, fourthCorner[0]+scanningRange+1)
        rangeY = range(fourthCorner[1]-scanningRange, fourthCorner[1]+scanningRange+1)

        sobelScoresX = [self.__calculateSobelScore(colcount, self.__fourPointTransform(self.__preprocessedImage, np.float32([self.__finderPatterns[0][2][1],self.__finderPatterns[1][2][1],[x, fourthCorner[1]],self.__finderPatterns[2][2][1]]))[0]) for x in rangeX]
        maxSobelScoreX = max(range(len(sobelScoresX)), key=sobelScoresX.__getitem__)
        fourthCornerCorrectedX = rangeX.__getitem__(maxSobelScoreX)

        if maxSobelScoreX != scanningRange:
            prevSobelScoresX = sobelScoresX[scanningRange]
            ix = 1
            direction = int((maxSobelScoreX - scanningRange) / abs(maxSobelScoreX - scanningRange))
            while sobelScoresX[maxSobelScoreX] > prevSobelScoresX and scanLimit > scanningRange * ix:
                fourthCornerCorrectedX = rangeX.__getitem__(maxSobelScoreX)
                prevSobelScoresX = sobelScoresX[maxSobelScoreX]
                rangeX = range(fourthCorner[0] + 1 + (scanningRange * ix) * direction, fourthCorner[0] + 1 + (scanningRange * (ix + 1)) * direction + 1, direction)
                sobelScoresX = [self.__calculateSobelScore(colcount, self.__fourPointTransform(self.__preprocessedImage, np.float32([self.__finderPatterns[0][2][1],self.__finderPatterns[1][2][1],[x, fourthCorner[1]],self.__finderPatterns[2][2][1]]))[0]) for x in rangeX]
                maxSobelScoreX = max(range(len(sobelScoresX)), key=sobelScoresX.__getitem__)
                ix += 1

        sobelScoresY = [self.__calculateSobelScore(colcount, self.__fourPointTransform(self.__preprocessedImage, np.float32([self.__finderPatterns[0][2][1],self.__finderPatterns[1][2][1],[fourthCorner[0] + (maxSobelScoreX - scanningRange), y],self.__finderPatterns[2][2][1]]))[0]) for y in rangeY]
        maxSobelScoreY = max(range(len(sobelScoresY)), key=sobelScoresY.__getitem__)
        fourthCornerCorrectedY = rangeY.__getitem__(maxSobelScoreY)

        if maxSobelScoreY != scanningRange:
            prevSobelScoresY = sobelScoresY[scanningRange]
            iy = 1
            direction = int((maxSobelScoreY - scanningRange) / abs(maxSobelScoreY - scanningRange))
            while sobelScoresY[maxSobelScoreY] > prevSobelScoresY and scanLimit > scanningRange * iy:
                fourthCornerCorrectedY = rangeY.__getitem__(maxSobelScoreY)
                prevSobelScoresY = sobelScoresY[maxSobelScoreY]
                rangeY = range(fourthCorner[1] + 1 + (scanningRange * iy) * direction, fourthCorner[1] + 1 + (scanningRange * (iy + 1)) * direction + 1, direction)
                sobelScoresY = [self.__calculateSobelScore(colcount, self.__fourPointTransform(self.__preprocessedImage, np.float32([self.__finderPatterns[0][2][1],self.__finderPatterns[1][2][1],[fourthCorner[0] + (maxSobelScoreY - scanningRange), y],self.__finderPatterns[2][2][1]]))[0]) for y in rangeY]
                maxSobelScoreY = max(range(len(sobelScoresY)), key=sobelScoresY.__getitem__)
                iy += 1

        fourthCornerCorrected = [fourthCornerCorrectedX, fourthCornerCorrectedY]

        self.__fourthCorner = fourthCornerCorrected
        
        if self.__makeProcessingLayers:
            visualiser = self.processingLayers[-1]
            visualiser = cv2.circle(visualiser, fourthCornerCorrected, radius=2, color=(0, 255, 0), thickness=-1)

    def __fourPointTransform(self, image, rect):
        (tl, tr, br, bl) = rect
        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))
        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))

        if maxHeight < maxWidth:
            maxWidth = maxHeight
        else:
            maxHeight = maxWidth

        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]], dtype = np.float32)
        transformationMatrix = cv2.getPerspectiveTransform(rect, dst)

        warpedImage = cv2.warpPerspective(image, transformationMatrix, (maxWidth, maxHeight))
        return warpedImage, transformationMatrix

    def __transformFinderPatternsPoints(self, transformationMatrix):
        self.__finderPatternsTransformed = [[self.__pointPerspectiveTransform(finderPattern[0], transformationMatrix), [self.__pointPerspectiveTransform(innerCorner, transformationMatrix) for innerCorner in finderPattern[1]], [self.__pointPerspectiveTransform(outerCorner, transformationMatrix) for outerCorner in finderPattern[2]]] for finderPattern in self.__finderPatterns]

    def __pointPerspectiveTransform(self, p, M):
        px = (M[0][0]*p[0] + M[0][1]*p[1] + M[0][2]) / ((M[2][0]*p[0] + M[2][1]*p[1] + M[2][2]))
        py = (M[1][0]*p[0] + M[1][1]*p[1] + M[1][2]) / ((M[2][0]*p[0] + M[2][1]*p[1] + M[2][2]))
        return np.rint(np.array([px, py])).astype(int)


    def __calculateSobelScore(self, colcount, inputImage, debugOutput = None):
        
        filler = np.full(inputImage.shape[1], 255)
        inputImage = cv2.GaussianBlur(inputImage, (3, 3), 0)

        inputImageVerticalSobel = cv2.Sobel(inputImage, cv2.CV_16S, 1, 0, ksize=1)
        inputImageVerticalSobel = cv2.convertScaleAbs(inputImageVerticalSobel)
        _, inputImageVerticalSobel = cv2.threshold(inputImageVerticalSobel, 180, 255, cv2.THRESH_BINARY)

        if debugOutput is not None:
            debugOutput.append(np.copy(inputImageVerticalSobel))

        cols = np.any(inputImageVerticalSobel, axis = 0)
        for i, state in enumerate(cols):
            if state:
                inputImageVerticalSobel[:,i] = filler
        zerosY = np.count_nonzero(cols == False)
    
        colCountY = 0
        countState = True
        colSizesY = (np.where(cols[:-1] != cols[1:])[0] + 1) - np.insert((np.where(cols[:-1] != cols[1:])[0] + 1)[:-1], 0, 0)
        colCountY = colSizesY[::2].size
        #for i, state in enumerate(cols):
        #    if state:
        #        countState = True
        #    elif countState:
        #        colCountY += 1
        #        countState = False

        if debugOutput is not None:
            tmpColored = cv2.cvtColor(inputImageVerticalSobel, cv2.COLOR_GRAY2BGR)
            tmpColored = cv2.putText(tmpColored, str((colCountY / colcount)), (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, 2)
            tmpColored = cv2.putText(tmpColored, str((inputImage.shape[1] / colcount) / np.std(colSizesY)), (5, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, 2)
            debugOutput.append(np.copy(tmpColored))

        inputImageHorizontalSobel = cv2.Sobel(inputImage, cv2.CV_16S, 0, 1, ksize=1)
        inputImageHorizontalSobel = cv2.convertScaleAbs(inputImageHorizontalSobel)
        _, inputImageHorizontalSobel = cv2.threshold(inputImageHorizontalSobel, 180, 255, cv2.THRESH_BINARY)

        if debugOutput is not None:
            debugOutput.append(np.copy(inputImageHorizontalSobel))

        rows = np.any(inputImageHorizontalSobel, axis = 1)
        for i, state in enumerate(rows):
            if state:
                inputImageHorizontalSobel[i,:] = filler
        zerosX = np.count_nonzero(rows == False)
    
        rowCountX = 0
        countState = True
        rowSizesX = (np.where(rows[:-1] != rows[1:])[0] + 1) - np.insert((np.where(rows[:-1] != rows[1:])[0] + 1)[:-1], 0, 0)
        rowCountX = rowSizesX[::2].size
        #for i, state in enumerate(rows):
        #    if state:
        #        countState = True
        #    elif countState:
        #        rowCountX += 1
        #        countState = False
                

        if debugOutput is not None:
            tmpColored = cv2.cvtColor(inputImageHorizontalSobel, cv2.COLOR_GRAY2BGR)
            tmpColored = cv2.putText(tmpColored, str((rowCountX / colcount)), (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, 2)
            tmpColored = cv2.putText(tmpColored, str((inputImage.shape[0] / colcount) / np.std(rowSizesX)), (5, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, 2)
            debugOutput.append(np.copy(tmpColored))

        colCountScore = 0
        if rowCountX != colcount:
            colCountScore -= 100000
        if colCountY != colcount:
            colCountScore -= 100000

        #return (zerosX - zerosX * 0.3 * np.std(rowSizesX)) + (zerosY - zerosY * 0.3 * np.std(colSizesY)) + colCountScore
        #return (1000 - 1000 * np.std(rowSizesX)) + (1000 - 1000 * np.std(colSizesY)) + colCountScore
        return (colCountY / colcount + rowCountX / colcount) + (inputImage.shape[0] / colcount) / np.std(rowSizesX) + (inputImage.shape[1] / colcount) / np.std(colSizesY)

    def __getSobelMask(self, image):
        fillerX = np.full(image.shape[1], 255)
        fillerY = np.full(image.shape[0], 255)
        image = cv2.GaussianBlur(image, (3, 3), 0)

        verticalSobel = cv2.Sobel(image, cv2.CV_16S, 1, 0, ksize=1)
        verticalSobel = cv2.convertScaleAbs(verticalSobel)
        _, verticalSobel = cv2.threshold(verticalSobel, 180, 255, cv2.THRESH_BINARY)

        cols = np.any(verticalSobel, axis = 0)
        if cols[len(cols)-1] == True:
            i = len(cols)-1
            while i > 0 and cols[i] == True:
                cols[i] = False
                i -= 1

        colCount = 0
        countState = True
        for i, state in enumerate(cols):
            if state:
                verticalSobel[:,i] = fillerX
                countState = True
            elif countState:
                colCount += 1
                countState = False

        horizontalSobel = cv2.Sobel(image, cv2.CV_16S, 0, 1, ksize=1)
        horizontalSobel = cv2.convertScaleAbs(horizontalSobel)
        _, horizontalSobel = cv2.threshold(horizontalSobel, 180, 255, cv2.THRESH_BINARY)

        rows = np.any(horizontalSobel, axis = 1)    
        if rows[len(rows)-1] == True:
            i = len(rows)-1
            while i > 0 and rows[i] == True:
                rows[i] = False
                i -= 1

        rowCount = 0
        countState = True
        for i, state in enumerate(rows):
            if state:
                horizontalSobel[i,:] = fillerY
                countState = True
            elif countState:
                rowCount += 1
                countState = False

        return np.maximum(verticalSobel, horizontalSobel), rows, cols, rowCount, colCount

    def __dealWithProcessingLayers(self, layer):
        if self.__makeProcessingLayers:
            self.processingLayers.append(np.copy(layer))

