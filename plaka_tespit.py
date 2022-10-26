



import numpy as np
import cv2

import keras_ocr


net = cv2.dnn.readNet("yolov4plate.weights", "yolov4plate.cfg")
classes = ['plate',]
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))
# kullnacağımız resim
img = cv2.imread("i12.jpg")

height, width, channels = img.shape

pipeline = keras_ocr.pipeline.Pipeline()

# Detecting objects
blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
net.setInput(blob)
outs = net.forward(output_layers)




class_ids = []
confidences = []
boxes = []
for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.25:
            # Object detected
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)
            # Rectangle coordinates
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)
            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)
            


indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)



font = cv2.FONT_HERSHEY_SIMPLEX
t={}
text=''
new_results=[]
for i in range(len(boxes)):
    if i in indexes:
        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])
        # burada resimde plakanın olduğu kısmı alıp
        # bunu yeni bir olarak bir değişkene atadık
        yeni=img[y:y+h,x:x+w]

        yeni=cv2.resize(yeni,(0,0), fx=3, fy=3)
        
        # burada yazıyı çıkarırken bazı hatalar var onları önlemek için
        # alltaki kodları yazdım. Burayı kurcalamaya gerek yok. Bu hatalar 
        # genelde harflerin sırasına yönelik. Eğer yanlış ya da eksik buldu ise
        # bu başka bir durum oluyor
        results = pipeline.recognize([yeni])
        for r in range(len(results[0])):
            t[r]=results[0][r][1][0][0]
        
        s = sorted(t.items(), key=lambda xx: xx[1])
        for pr in range(len(results[0])):
            new_results.append(results[0][s[pr][0]])
            
        for p in range(len(results[0])):
            text+=new_results[p][0]
        # plakayı yazı olarak alıp harfleri büyük harfe çevirdik.
        text=text.upper()
        
        color=(116,0,102)
        color2= (0,255,255)
        color3= (0,155,255)

        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        cv2.putText(img, text, (0,60 ), font, 2, color3, 3)  
        cv2.putText(img, label, (x, y-10 ), font, 2, color2, 2)
        #cv2.imshow('plaka', yeni)
cv2.imshow("Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()