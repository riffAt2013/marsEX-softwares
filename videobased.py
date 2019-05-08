import numpy as np
import cv2 as cv

#PathGula absolute path hishebe load kor kaj na korle
labelPath = "coco.names"
weightPath = "yolov3.weights"
configPath = "yolov3.cfg"

defaultConfidenceVal = 0.5
defaulThresholdVal = 0.3
np.random.seed(42)
LABELS = open(labelPath).read().strip().split("\n")
COLORS = np.random.randint(0,255, 80, dtype = 'uint8')

net = cv.dnn.readNetFromDarknet(configPath, weightPath)

capture = cv.VideoCapture(0)

while True:
    ret, image = capture.read()

    if ret == True:
        imHeight, imWidth = image.shape[:2]

        layerName = net.getLayerNames()
        layerName = [layerName[i[0]-1] for i in net.getUnconnectedOutLayers()]

        blobCreation = cv.dnn.blobFromImage(image, 1/255.0, (416,416), swapRB = True, crop = False)
        net.setInput(blobCreation)
        layerOutputs = net.forward(layerName)

        boxes = []
        confidenceVals = []
        classIds = []

        for output in layerOutputs:
            for detection in output:
                scores = detection[5:]
                classID = np.argmax(scores)
                confidenceVal = scores[classID]

                if confidenceVal > defaultConfidenceVal:
                    box = detection[0:4]*np.array([imWidth, imHeight,imWidth, imHeight])
                    (centerX, centerY, width, height) = box.astype("int")

                    x = int(centerX - (width/2))
                    y = int(centerY - (width/2))

                    boxes.append([x, y, int(width), int(height)])
                    confidenceVals.append (float(confidenceVal))
                    classIds.append(classID)

            # Here is the problem: To apply non-maxima suppression, we need to use NMSBoxes which is built in
            # But its not working. We might need to create our own NMS function through imutils :(
        idxs = cv.dnn.NMSBoxes(boxes, confidenceVals, defaultConfidenceVal, defaulThresholdVal)

        if len(idxs)>0:
            for i in idxs.flatten():
                (x,y) = (boxes[i][0], boxes[i][1])
                (w,h) = (boxes[i][2], boxes[i][3])
                # color = [int(c) for c in COLORS[classIds[i]]]
                color = (0,255,0)

                cv.rectangle((image), (x,y), (x+w,y+h), color, 2)
                text = str(LABELS[classIds[i]])+ " "+ str(round(confidenceVals[i],2)*100)+ " %"
                cv.putText(image, text, (x, y-5), cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # optional resizing window size for HQ images
            # cv.namedWindow('Output', cv.WINDOW_NORMAL)
            # cv.resizeWindow('Output', (640,480))
        cv.imshow("Output", image)
        if cv.waitKey(30) & 0xFF == ord('q'):
            break
    else:
        break



capture.release()
cv.destroyAllWindows()
