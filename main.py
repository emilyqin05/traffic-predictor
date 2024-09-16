"""
emilyqin@Emilys-MacBook-Air-42 ~ % source ~/.bash_profile
emilyqin@Emilys-MacBook-Air-42 ~ % workon cv
(cv) emilyqin@Emilys-MacBook-Air-42 ~ % pip install numpy h5py pillow scikit-image
(cv) emilyqin@Emilys-MacBook-Air-42 ~ % pip install opencv-python
(cv) emilyqin@Emilys-MacBook-Air-42 ~ % pip install dlib
(cv) emilyqin@Emilys-MacBook-Air-42 ~ % pip install tensorflow
(cv) emilyqin@Emilys-MacBook-Air-42 ~ % pip install keras
(cv) emilyqin@Emilys-MacBook-Air-42 ~ % wget https://pjreddie.com/media/files/yolov3.weights
(cv) emilyqin@Emilys-MacBook-Air-42 traffic-predictor % which python

/Users/emilyqin/.virtualenvs/cv/bin/python

*/2 * * * * /bin/bash -c 'source /Users/emilyqin/.virtualenvs/cv/bin/activate && /Users/emilyqin/.virtualenvs/cv/bin/python python main.py' >> 
*/2 * * * * /bin/bash -c 'source /Users/emilyqin/.virtualenvs/cv/bin/activate && /Users/emilyqin/.virtualenvs/cv/bin/python /Users/emilyqin/Desktop/traffic-predictor/main.py' >> /Users/emilyqin/Desktop/traffic-predictor/logfile.log 2>&1
correct cron?
*/2 * * * * /bin/bash -c 'source /Users/emilyqin/.virtualenvs/cv/bin/activate && /Users/emilyqin/.virtualenvs/cv/bin/python /Users/emilyqin/Desktop/traffic-predictor/main.py' >> /Users/emilyqin/Desktop/traffic-predictor/logfile.log 2>&1

*/2 * * * * /bin/bash -c 'source /path/to/your/virtualenv/bin/activate && /path/to/your/virtualenv/bin/python /path/to/your_script.py' >> /path/to/logfile.log 2>&1


in runTrafficPredictor.sh:
#!/bin/zsh
source /Users/emilyqin/.virtualenvs/cv/bin/activate
/Users/emilyqin/.virtualenvs/cv/bin/python /Users/emilyqin/Desktop/traffic-predictor/main.py >> /Users/emilyqin/Desktop/traffic-predictor/logfile.log 2>&1
env > /Users/emilyqin/Desktop/traffic-predictor/env_variables.log

correct cron: 
2 * * * * /bin/zsh /Users/emilyqin/Desktop/traffic-predictor/runTrafficPredictor.sh >> /Users/emilyqin/Desktop/traffic-predictor/cronjob.log 2>&1

com.demo.daemon.plist:
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.demo.daemon.plist</string>

    <key>RunAtLoad</key>
    <true/>

    <key>StartInterval</key>
    <integer>120</integer> <!-- Run every 2 minutes -->

    <key>StandardErrorPath</key>
    <string>/Users/emilyqin/Desktop/traffic-predictor/cronjob.log</string>

    <key>StandardOutPath</key>
    <string>/Users/emilyqin/Desktop/traffic-predictor/cronjob.log</string>

    <key>EnvironmentVariables</key>
    <dict>
      <key>PATH</key>
      <string><![CDATA[/usr/local/bin:/usr/local/sbin:/usr/bin:/bin:/usr/sbin:/sbin]]></string>
    </dict>

    <key>WorkingDirectory</key>
    <string>/Users/emilyqin/Desktop/traffic-predictor</string>


    <key>ProgramArguments</key>
    <array>
    <string>/bin/zsh</string>
    <string>/Users/emilyqin/Desktop/traffic-predictor/runTrafficPredictor.sh</string>
    </array>


</dict>
</plist>



"""

from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.service import Service
from PIL import Image
from io import BytesIO

import cv2
import argparse
import numpy as np
import csv
import os
import datetime
import time
import schedule

def take_screenshot_and_update_csv():
    # Set up the ChromeDriver service
    service = Service(ChromeDriverManager().install())

    # Initialize the Chrome WebDriver using the service
    driver = webdriver.Chrome(service=service)

    # Visit the webpage
    driver.get("https://www.drivebc.ca/mobile/pub/webcams/id/684.html")

    # Save screenshot of the entire page as a PNG
    png = driver.get_screenshot_as_png()

    # Open the image in memory with PIL library
    im = Image.open(BytesIO(png))

    # Define crop points 
    left = 20
    top = 300
    width = 650
    height = 650
    im = im.crop((left, top, (left + width) , (top + height)))

    # Save the cropped image
    im.save('/Users/emilyqin/Desktop/traffic-predictor/cropped_screenshot.png')

    # Close the browser
    driver.quit()
    #time.sleep(3)


    # hardcoded paths to input image, config, weights, and classes
    image_path = '/Users/emilyqin/Desktop/traffic-predictor/cropped_screenshot.png'
    config_path = '/Users/emilyqin/Desktop/traffic-predictor/yolov3.cfg'
    weights_path = '/Users/emilyqin/Desktop/traffic-predictor/yolov3.weights'
    #print('before assignemnt')
    classes_path = '/Users/emilyqin/Desktop/traffic-predictor/yolov3.txt'
    #print ('after assignment')

    #test
    with open('/Users/emilyqin/Desktop/traffic-predictor/logfile.log', 'a') as log_file:
        log_file.write(f"Image path: {image_path}\n")
        log_file.write(f"Config path: {config_path}\n")
        log_file.write(f"Weights path: {weights_path}\n")
        log_file.write(f"Classes path: {classes_path}\n")
        log_file.write(f"Does image path exist? {os.path.exists(image_path)}\n")
    #end test

    #print("Current Working Directory:", os.getcwd())
    #print("Classes Path:", classes_path)

    # reading input image using opencv's imread function
    image = cv2.imread(image_path)
    #image.shape returns a tuple with (<height>, <width>, <# of channels>)
    Width = image.shape[1]
    Height = image.shape[0]
    # normalized scale factor (approx. 1/255)
    # multiplying each pixel value by 0.00392 converts the value from the range [0, 255] to [0, 1].
    scale = 0.00392
    # initialize the classes variable
    classes = None
    # open arg containing classes in read mode as f 
    with open(classes_path, 'r') as f:
        # store each line (class name) in the classes list
        classes = [line.strip() for line in f.readlines()]
    # generates a list of random colours for each class
    # creates an array of shape (len(classes), <RGB color channels>), with each entry a random value between 0 and 255
    # this is just for the bounding boxes colours
    COLORS = np.random.uniform(0, 255, size=(len(classes),3))

    #net (neural network) = cv2.dnn.readNet(<weights file>, <configuration file>)
    net = cv2.dnn.readNet(weights_path, config_path)

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



    # apply non-max suppression to filter out overlapping bounding boxes.
    # boxes - list of bounding boxes '[x, y, w, h]'
    # confidences - list of confidence scores
    # conf_threshold - minimum confidence score
    # nms_threshold - if intersection over union (IoU) bt 2 boxes is greater than threshold, one box will be suppressed.
    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
    num_objects = len(indices)
    #print(f"Number of detected objects: {num_objects}")

    # release resources
    cv2.destroyAllWindows()

    x = datetime.datetime.now()
    date = x.strftime("%x")
    time = x.strftime("%X")

    csvFilePath = '/Users/emilyqin/Desktop/traffic-predictor/information.csv'
    fields=[date,time,num_objects]
    with open(csvFilePath, 'a') as f:
        writer = csv.writer(f)
        writer.writerow(fields)

"""def job_scheduler():
    # Get the current time and day
    now = datetime.datetime.now()
    current_hour = now.hour
    current_day = now.weekday()  # Monday is 0 and Sunday is 6

    # Define conditions for Monday-Friday (0-4) and the two time ranges
    #if 0 <= current_day <= 4:
        # Run only between 7-9 AM or 3-5 PM
        #if (7 <= current_hour < 9) or (15 <= current_hour < 17):
    take_screenshot_and_update_csv()

# Schedule the job every 2 minutes but only execute it during valid hours
schedule.every(2).minutes.do(job_scheduler)

while True:
    schedule.run_pending()
    time.sleep(1)"""

take_screenshot_and_update_csv()