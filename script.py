import cv2
import numpy as np
# Loading  Yolo weights 
net = cv2.dnn.readNet("./Yolo/yolov3_custom_last.weights", "./Yolo/yolov3_custom.cfg")
classes = list()
f= open("./Yolo/obj.names", "r")#read .names file
for line in f.readlines():#loop each line in file
    classes.append(line.strip())
layer_names = net.getLayerNames()#get layes names
layer=list()#init layes list
for i in net.getUnconnectedOutLayers():
    layer.append(layer_names[i[0] - 1])
colors = np.random.uniform(0, 255, size=(len(classes), 3))

#provide video path in string or 0== internal webcam 2== external webcam
cap = cv2.VideoCapture(2)
if (cap.isOpened()== False): 
    print("Error opening video stream or file")


while(cap.isOpened()):
    # get video Frame by Frame
    ret, frame = cap.read()
    if ret == True:

        img=frame#set img to frame.
        height, width,_ = img.shape
        blobObject = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blobObject)
        outputs = net.forward(layer)#get next layer
        class_ids = list()
        confidences = list()
        boxes = list()
        for out in outputs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:#tresh >0.5

                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    # get points for rectangle
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
        dec = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)#number of dectected objects in frame

        for i in range(len(boxes)):
            if i in dec:
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])#get label
                color = colors[i]#each classs has specific color defined above
                cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)#create rectangle
                cv2.putText(img, label, (10,50), cv2.FONT_ITALIC, 2, color, 3)
        cv2.imshow("Image", img)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    else:
        break

 



cap.release()
cv2.destroyAllWindows()
