import streamlit as st
import numpy as np
import cv2
import time

st.title("OBJECT DETECTION")


rad = st.sidebar.radio("Navigation", ["Home", "YOLO V3","YOLO V5"])

if rad=="Home":
    st.image("imageyolo.jpg")
    text = "Object detection is a computer vision task that involves identifying and localizing objects within an image or video. The You Only Look Once (YOLO) algorithm is a popular deep learning-based approach for object detection. YOLO utilizes a single neural network to predict bounding boxes and class probabilities for objects in an image, all at once, rather than breaking the image down into subregions. This approach results in a fast and efficient object detection system that can processimages in real-time. YOLO has seen widespread use in various applications, including self driving cars, robotics, and security systems. This paper provides an overview of the YOLO algorithm, including its architecture, training process, and performance evaluation. The results of the YOLO algorithm on several benchmark datasets are presented, demonstrating its effectiveness and efficiency in object detection tasks.images in real-time."
          
    st.write(text)
 



if rad=="YOLO V3":
    rad1 = st.sidebar.radio("Navigate", ["Camera", "Image in your system"])
    
    if rad1=="Camera":
        st.write("Press q to exit the camera")

        camera = cv2.VideoCapture(0)

        # Preparing variables for spatial dimensions of the frames
        h, w = None, None

       
        with open('yolo-coco-data\coco.names') as f:
            # Getting labels reading every line
            # and putting them into the list
            labels = [line.strip() for line in f]


        
        network = cv2.dnn.readNetFromDarknet(r'yolo-coco-data\yolov3.cfg',
                                            r'yolo-coco-data\yolov3.weights')

        # Getting list with names of all layers from YOLO v3 network
        layers_names_all = network.getLayerNames()

        
        layers_names_output = \
            [layers_names_all[i - 1] for i in network.getUnconnectedOutLayers()]

       
        probability_minimum = 0.5

        
        threshold = 0.3

       
        colours = np.random.randint(0, 255, size=(len(labels), 3), dtype='uint8')

        bu = st.button("Camera")
        if bu :
            while True:
                # Capturing frame-by-frame from camera
                _, frame = camera.read()

            
                if w is None or h is None:
                    # Slicing from tuple only first two elements
                    h, w = frame.shape[:2]

                """
                Start of:
                Getting blob from current frame
                """

            
                blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),
                                            swapRB=True, crop=False)

                """
                End of:
                Getting blob from current frame
                """

                """
                Start of:
                Implementing Forward pass
                """

            
                network.setInput(blob)  # setting blob as input to the network
                start = time.time()
                output_from_network = network.forward(layers_names_output)
                end = time.time()

                # Showing spent time for single current frame
                print('Current frame took {:.5f} seconds'.format(end - start))

                """
                End of:
                Implementing Forward pass
                """

                """
                Start of:
                Getting bounding boxes
                """

            
                bounding_boxes = []
                confidences = []
                class_numbers = []

                # Going through all output layers after feed forward pass
                for result in output_from_network:
                    # Going through all detections from current output layer
                    for detected_objects in result:
                        # Getting 80 classes' probabilities for current detected object
                        scores = detected_objects[5:]
                        # Getting index of the class with the maximum value of probability
                        class_current = np.argmax(scores)
                        # Getting value of probability for defined class
                        confidence_current = scores[class_current]

                
                        # Eliminating weak predictions with minimum probability
                        if confidence_current > probability_minimum:
                        
                            box_current = detected_objects[0:4] * np.array([w, h, w, h])

                        
                            x_center, y_center, box_width, box_height = box_current
                            x_min = int(x_center - (box_width / 2))
                            y_min = int(y_center - (box_height / 2))

                            # Adding results into prepared lists
                            bounding_boxes.append([x_min, y_min,
                                                int(box_width), int(box_height)])
                            confidences.append(float(confidence_current))
                            class_numbers.append(class_current)

                """
                End of:
                Getting bounding boxes
                """

                """
                Start of:
                Non-maximum suppression
                """

            
                # https://github.com/opencv/opencv/issues/12789
                results = cv2.dnn.NMSBoxes(bounding_boxes, confidences,
                                        probability_minimum, threshold)

                """
                End of:
                Non-maximum suppression
                """

                """
                Start of:
                Drawing bounding boxes and labels
                """

                # Checking if there is at least one detected object
                # after non-maximum suppression
                if len(results) > 0:
                    # Going through indexes of results
                    for i in results.flatten():
                        # Getting current bounding box coordinates,
                        # its width and height
                        x_min, y_min = bounding_boxes[i][0], bounding_boxes[i][1]
                        box_width, box_height = bounding_boxes[i][2], bounding_boxes[i][3]

                        # Preparing colour for current bounding box
                        # and converting from numpy array to list
                        colour_box_current = colours[class_numbers[i]].tolist()

                    
                        cv2.rectangle(frame, (x_min, y_min),
                                    (x_min + box_width, y_min + box_height),
                                    colour_box_current, 2)

                        # Preparing text with label and confidence for current bounding box
                        text_box_current = '{}: {:.4f}'.format(labels[int(class_numbers[i])],
                                                            confidences[i])

                        # Putting text with label and confidence on the original image
                        cv2.putText(frame, text_box_current, (x_min, y_min - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour_box_current, 2)

                """
                End of:
                Drawing bounding boxes and labels
                """

                """
                Start of:
                Showing processed frames in OpenCV Window
                """

            
                cv2.namedWindow('YOLO v3 Real Time Detections', cv2.WINDOW_NORMAL)
                # Pay attention! 'cv2.imshow' takes images in BGR format
                cv2.imshow('YOLO v3 Real Time Detections', frame)

                # Breaking the loop if 'q' is pressed
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

                """
                End of:
                Showing processed frames in OpenCV Window
                """

        

            # Releasing camera
            camera.release()
            # Destroying all opened OpenCV windows
            cv2.destroyAllWindows()
    
    if rad1=="Image in your system":
        
        image_BGR = cv2.imread(st.text_input("Enter the location of the image for detecting objects"))

        # Showing Original Image
        # Giving name to the window with Original Image
        # And specifying that window is resizable
        cv2.namedWindow('Original Image', cv2.WINDOW_NORMAL)
        # Pay attention! 'cv2.imshow' takes images in BGR format
        # cv2.imshow('Original Image', image_BGR)
        # Waiting for any key being pressed
        cv2.waitKey(0)
        

        # Getting spatial dimension of input image
        h, w = image_BGR.shape[:2]  # Slicing from tuple only first two elements

        
        """
        End of: 
        Reading input image
        """


        """
        Start of:
        Getting blob from input image
        """

        
        blob = cv2.dnn.blobFromImage(image_BGR, 1 / 255.0, (416, 416),swapRB=True, crop=False)

       
        with open(r'yolo-coco-data\coco.names') as f:
            # Getting labels reading every line
            # and putting them into the list
            labels = [line.strip() for line in f]


        # # Check point
        # print('List with labels names:')
        # print(labels)

        # Loading trained YOLO v3 Objects Detector
        # with the help of 'dnn' library from OpenCV
        # Pay attention! If you're using Windows, yours paths might look like:
        # r'yolo-coco-data\yolov3.cfg'
        # r'yolo-coco-data\yolov3.weights'
        # or:
        # 'yolo-coco-data\\yolov3.cfg'
        # 'yolo-coco-data\\yolov3.weights'
        network = cv2.dnn.readNetFromDarknet(r'yolo-coco-data\yolov3.cfg',
                                            r'yolo-coco-data\yolov3.weights')

        # Getting list with names of all layers from YOLO v3 network
        layers_names_all = network.getLayerNames()

        # # Check point

        # print()
        # print(layers_names_all)

        # Getting only output layers' names that we need from YOLO v3 algorithm
        # with function that returns indexes of layers with unconnected outputs
        layers_names_output = \
            [layers_names_all[i - 1] for i in network.getUnconnectedOutLayers()]

        # # Check point
        # print()
        # print(layers_names_output)  # ['yolo_82', 'yolo_94', 'yolo_106']

        # Setting minimum probability to eliminate weak predictions
        probability_minimum = 0.5

        # Setting threshold for filtering weak bounding boxes
        # with non-maximum suppression
        threshold = 0.3

        # Generating colours for representing every detected object
        # with function randint(low, high=None, size=None, dtype='l')
        colours = np.random.randint(0, 255, size=(len(labels), 3), dtype='uint8')

        # # Check point
        # print()
        # print(type(colours))  # <class 'numpy.ndarray'>
        # print(colours.shape)  # (80, 3)
        # print(colours[0])  # [172  10 127]

        """
        End of:
        Loading YOLO v3 network
        """


        """
        Start of:
        Implementing Forward pass
        """

        # Implementing forward pass with our blob and only through output layers
        # Calculating at the same time, needed time for forward pass
        network.setInput(blob)  # setting blob as input to the network
        start = time.time()
        output_from_network = network.forward(layers_names_output)
        end = time.time()

        # Showing spent time for forward pass
        print('Objects Detection took {:.5f} seconds'.format(end - start))

        """
        End of:
        Implementing Forward pass
        """


        """
        Start of:
        Getting bounding boxes
        """

        # Preparing lists for detected bounding boxes,
        # obtained confidences and class's number
        bounding_boxes = []
        confidences = []
        class_numbers = []


        # Going through all output layers after feed forward pass
        for result in output_from_network:
            # Going through all detections from current output layer
            for detected_objects in result:
                # Getting 80 classes' probabilities for current detected object
                scores = detected_objects[5:]
                # Getting index of the class with the maximum value of probability
                class_current = np.argmax(scores)
                # Getting value of probability for defined class
                confidence_current = scores[class_current]

                # # Check point
                # # Every 'detected_objects' numpy array has first 4 numbers with
                # # bounding box coordinates and rest 80 with probabilities for every class
                # print(detected_objects.shape)  # (85,)

                # Eliminating weak predictions with minimum probability
                if confidence_current > probability_minimum:
                    # Scaling bounding box coordinates to the initial image size
                    # YOLO data format keeps coordinates for center of bounding box
                    # and its current width and height
                    # That is why we can just multiply them elementwise
                    # to the width and height
                    # of the original image and in this way get coordinates for center
                    # of bounding box, its width and height for original image
                    box_current = detected_objects[0:4] * np.array([w, h, w, h])

                    # Now, from YOLO data format, we can get top left corner coordinates
                    # that are x_min and y_min
                    x_center, y_center, box_width, box_height = box_current
                    x_min = int(x_center - (box_width / 2))
                    y_min = int(y_center - (box_height / 2))

                    # Adding results into prepared lists
                    bounding_boxes.append(
                        [x_min, y_min, int(box_width), int(box_height)])
                    confidences.append(float(confidence_current))
                    class_numbers.append(class_current)

        """
        End of:
        Getting bounding boxes
        """


        """
        Start of:
        Non-maximum suppression
        """

        # Implementing non-maximum suppression of given bounding boxes
        # With this technique we exclude some of bounding boxes if their
        # corresponding confidences are low or there is another
        # bounding box for this region with higher confidence

        # It is needed to make sure that data type of the boxes is 'int'
        # and data type of the confidences is 'float'
        # https://github.com/opencv/opencv/issues/12789
        results = cv2.dnn.NMSBoxes(bounding_boxes, confidences,
                                probability_minimum, threshold)

        """
        End of:
        Non-maximum suppression
        """


        """
        Start of:
        Drawing bounding boxes and labels
        """

        # Defining counter for detected objects
        counter = 1

        # Checking if there is at least one detected object after non-maximum suppression
        if len(results) > 0:
            # Going through indexes of results
            for i in results.flatten():
                # Showing labels of the detected objects
                print('Object {0}: {1}'.format(counter, labels[int(class_numbers[i])]))

                # Incrementing counter
                counter += 1

                # Getting current bounding box coordinates,
                # its width and height
                x_min, y_min = bounding_boxes[i][0], bounding_boxes[i][1]
                box_width, box_height = bounding_boxes[i][2], bounding_boxes[i][3]

                # Preparing colour for current bounding box
                # and converting from numpy array to list
                colour_box_current = colours[class_numbers[i]].tolist()

                # # # Check point
                # print(type(colour_box_current))  # <class 'list'>
                # print(colour_box_current)  # [172 , 10, 127]

                # Drawing bounding box on the original image
                cv2.rectangle(image_BGR, (x_min, y_min),
                            (x_min + box_width, y_min + box_height),
                            colour_box_current, 2)

                # Preparing text with label and confidence for current bounding box
                text_box_current = '{}: {:.4f}'.format(labels[int(class_numbers[i])],
                                                    confidences[i])

                # Putting text with label and confidence on the original image
                cv2.putText(image_BGR, text_box_current, (x_min, y_min - 5),
                            cv2.FONT_HERSHEY_COMPLEX, 0.7, colour_box_current, 2)


        # Comparing how many objects where before non-maximum suppression
        # and left after
        print()
        print('Total objects been detected:', len(bounding_boxes))
        print('Number of objects left after non-maximum suppression:', counter - 1)

        """
        End of:
        Drawing bounding boxes and labels
        """


        # Showing Original Image with Detected Objects
        # Giving name to the window with Original Image
        # And specifying that window is resizable
        cv2.namedWindow('Detections', cv2.WINDOW_NORMAL)
        # Pay attention! 'cv2.imshow' takes images in BGR format
        cv2.imshow('Detections', image_BGR)
        # Waiting for any key being pressed
        cv2.waitKey(0)
        # Destroying opened window with name 'Detections'
        # cv2.destroyWindow('Detections')


        """
        Some comments

        With OpenCV function 'cv2.dnn.blobFromImage' we get 4-dimensional
        so called 'blob' from input image after mean subtraction,
        normalizing, and RB channels swapping. Resulted shape has:
        - number of images
        - number of channels
        - width
        - height
        E.G.: blob = cv2.dnn.blobFromImage(image, scalefactor=1.0, size, mean, swapRB=True)
        """



            

                        


