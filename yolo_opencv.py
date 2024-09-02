import cv2
import argparse
import numpy as np

# command line args
ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required = True, help = 'path to input image') #add_argument( <command-line flags>, <requirement to run>, description of argument>)
ap.add_argument('-c', '--config', required = True, help = 'path to yolo config file')
ap.add_argument('-w', '--weights', required = True, help = 'path to yolo pre-trained weights')
ap.add_argument('cl', '--classes', required = True, help = 'path to text file containing class names')
args = ap.parse_args()

# reading input image using opencv's imread function
image = cv2.imread(args.image)
#image.shape returns a tuple with (<height>, <width>, <# of channels>)
Height = image.shape[0]
Width = image.shape[1]
# normalized scale factor (approx. 1/255)
# multiplying each pixel value by 0.00392 converts the value from the range [0, 255] to [0, 1].
scale = 0.00392
# initialize the classes variable
classes = None
# open arg containing classes in read mode as f 
with open(args.classes, 'r') as f:
    # store each line (class name) in the classes list
    classes = [line.strip() for line in f.readlines()]
# generates a list of random colours for each class
# creates an array of shape (len(classes), <RGB color channels>), with each entry a random value between 0 and 255
# this is just for the bounding boxes colours
COLORS = np.random.uniform(0, 255, size=(len(classes),3))

#net (neural network) = cv2.dnn.readNet(<weights file>, <configuration file>)
net = cv2.dnn.readNet(args.weights, args.config)

# blob - preprocessed image that the network expects as input
# cv2.dnn.blobFromImage( <input image>, <scale pixel values>, <input size required by YOLO model>, <mean subtraction value, not applied here>, <will the image be swapped from RGB to BGR?>, <will the image be cropped?>)
blob = cv2.dnn.blobFromImage(image, scale, (416, 416), (0,0,0), True, crop = False)

# sets the input blob for the network
# represents that the network is ready to process input image encapsulated in blob for object detection
net.setInput(blob)


# function to get the output layer names 
# in the architecture
def get_output_layers(net):
    
    layer_names = net.getLayerNames()
    
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    return output_layers

# function to draw bounding box on the detected object with class name
def draw_bounding_box(img, class_id, confidence, x, y, x_plus_w, y_plus_h):

    label = str(classes[class_id])

    color = COLORS[class_id]

    cv2.rectangle(img, (x,y), (x_plus_w,y_plus_h), color, 2)

    cv2.putText(img, label, (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

# run inference through the network
# and gather predictions from output layers
outs = net.forward(get_output_layers(net))

# initialization
class_ids = []
confidences = []
boxes = []
conf_threshold = 0.5
nms_threshold = 0.4

# for each detetion from each output layer 
# get the confidence, class id, bounding box params
# and ignore weak detections (confidence < 0.5)
for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.5:
            center_x = int(detection[0] * Width)
            center_y = int(detection[1] * Height)
            w = int(detection[2] * Width)
            h = int(detection[3] * Height)
            x = center_x - w / 2
            y = center_y - h / 2
            class_ids.append(class_id)
            confidences.append(float(confidence))
            boxes.append([x, y, w, h])


# apply non-max suppression
indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

# go through the detections remaining
# after nms and draw bounding box
for i in indices:
    i = i[0]
    box = boxes[i]
    x = box[0]
    y = box[1]
    w = box[2]
    h = box[3]
    
    draw_bounding_box(image, class_ids[i], confidences[i], round(x), round(y), round(x+w), round(y+h))

# display output image    
cv2.imshow("object detection", image)

# wait until any key is pressed
cv2.waitKey()
    
 # save output image to disk
cv2.imwrite("object-detection.jpg", image)

# release resources
cv2.destroyAllWindows()
