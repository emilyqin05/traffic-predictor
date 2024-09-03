import cv2
import argparse
import numpy as np

# command line args
ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required = True, help = 'path to input image') #add_argument( <command-line flags>, <requirement to run>, description of argument>)
ap.add_argument('-c', '--config', required = True, help = 'path to yolo config file')
ap.add_argument('-w', '--weights', required = True, help = 'path to yolo pre-trained weights')
ap.add_argument('-cl', '--classes', required = True, help = 'path to text file containing class names')
args = ap.parse_args()

# reading input image using opencv's imread function
image = cv2.imread(args.image)
#image.shape returns a tuple with (<height>, <width>, <# of channels>)
Width = image.shape[1]
Height = image.shape[0]
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


def get_output_layers(net):
    # list of all the layer names in the neural network
    layer_names = net.getLayerNames()

    # gets indices of output layers, final layers where model outputs prediction. is a list of integers/lists
    out_layer_indices = net.getUnconnectedOutLayers()

    # Check if out_layer_indices is a list of lists (older OpenCV versions) or a list of integers (newer versions)
    if isinstance(out_layer_indices[0], list):
        # Flatten the list if it's nested, basically turning a list of lists into a list of integers
        out_layer_indices = [item for sublist in out_layer_indices for item in sublist]

    # Get the output layer names (opencv starts at index 1 instead of 0)
    output_layers = [layer_names[i - 1] for i in out_layer_indices]

    return output_layers


# function to draw bounding box on the detected object with class name
def draw_bounding_box(img, class_id, confidence, x, y, x_plus_w, y_plus_h):

    label = str(classes[class_id])

    color = COLORS[class_id]

    cv2.rectangle(img, (x,y), (x_plus_w,y_plus_h), color, 2)

    cv2.putText(img, label, (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

# run inference through the network and gather predictions from output layers
# net is the pretrained YOLO model. 
# get_output_layers(net) retrieves the names/indices of the output layers 
# net.forward(...) takes the input image and produces predictions at the output layers
# "outs" stores the output predictions (bounding boxes, class IDs, confidence scores)
outs = net.forward(get_output_layers(net))

# initialize empty list to store class IDs (ex. person, car, dog)
class_ids = []
# initialize an empty list to store the confidence scores for each detected object
confidences = []
# initialize an empty list to store the bounding boxes for the detected objects (set of coord)
boxes = []
# sets a threshold for filtering out weak detections. ex. only confidence scores > 0.5 will be considered valid
conf_threshold = 0.5
# sets a threshold for non maximum suppression, remove duplicate bounding boxes
nms_threshold = 0.4

# "outs" is a list of outputs from different layers. each "out" contains multiple detection results, 
# where each detection has info about the object's position, confidence score, and class probabilities
for out in outs:
    # loop iterates over each detection (which contains bounding box coord, confidence scores, class scores)
    for detection in out:
        # find the highest score in the "scores" array, and return the index in class_id
        scores = detection[5:]
        class_id = np.argmax(scores)
        # set the highest score as the confidence
        confidence = scores[class_id]
        if confidence > 0.5:
            # set bounding box
            center_x = int(detection[0] * Width)
            center_y = int(detection[1] * Height)
            w = int(detection[2] * Width)
            h = int(detection[3] * Height)
            x = center_x - w / 2
            y = center_y - h / 2
            class_ids.append(class_id)
            confidences.append(float(confidence))
            boxes.append([x, y, w, h])


# Find the class ID for "Car"
car_class_name = "Car"
car_class_id = None
for i, name in enumerate(class_names):
    if name == car_class_name:
        car_class_id = i
        break

if car_class_id is not None:
    print(f"Class ID for '{car_class_name}' is: {car_class_id}")
else:
    print(f"'{car_class_name}' not found in class names list.")


# apply non-max suppression to filter out overlapping bounding boxes.
# boxes - list of bounding boxes '[x, y, w, h]'
# confidences - list of confidence scores
# conf_threshold - minimum confidence score
# nms_threshold - if intersection over union (IoU) bt 2 boxes is greater than threshold, one box will be suppressed.
indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

# Go through the detections remaining after NMS and draw bounding boxes
for i in indices:
    # `i` is already a scalar, use it directly
    i = int(i)  # Ensure `i` is an integer if it's not already

    # Access the bounding box
    box = boxes[i]
    # extract the x coord of the top left corner
    x = box[0]
    # extract the y coord of the top left corner
    y = box[1]
    # extract width
    w = box[2]
    # extract height
    h = box[3]
    
    # Draw the bounding box on the image
    # <image (on which to draw the bounding box)>, <class ID of detected object>, <confidence score for detection>, <coords for top left + bottom right corners>
    draw_bounding_box(image, class_ids[i], confidences[i], round(x), round(y), round(x + w), round(y + h))

# display the image with the drawn bounding boxes in a window titled "object detection."
# cv2.imshow(<window title>, <og image>)
cv2.imshow("object detection", image)

# wait until any key is pressed
cv2.waitKey()
    
 # save output image to disk
cv2.imwrite("carOutput.jpg", image)

# release resources
cv2.destroyAllWindows()
