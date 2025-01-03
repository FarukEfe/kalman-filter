from kalmanfilter import KalmanFilter
from gradient import gradient
import cv2, numpy as np

class Detector:
    
    def __init__(self, videoPath, configPath, modelPath, classesPath):
        self.videoPath = videoPath
        self.configPath = configPath
        self.modelPath = modelPath
        self.classesPath = classesPath
        self.classesList = None

        # Initialize NN

        self.net = cv2.dnn.DetectionModel(self.modelPath, self.configPath)
        self.net.setInputSize(512,512)
        self.net.setInputScale(1/255.0)
        self.net.setInputMean((127.5,127.5,127.5))
        self.net.setInputSwapRB(True)

        self.readClasses()
    
    def readClasses(self):
        with open(self.classesPath, 'r') as f:
            self.classesList = f.read().splitlines()
        
        self.classesList.insert(0,'__VOID__')
    
    def onVideo(self):
        cap = cv2.VideoCapture(self.videoPath)
        kf = KalmanFilter()

        if (cap.isOpened() is False):
            print("Error opening file...")
            return
        
        while True:
            '''Read Frame'''
            (success, frame) = cap.read()
            if success is False: break # Once there's no frame, finish

            '''Detect Objects'''
            ids, confidences, bboxs = self.net.detect(frame, confThreshold=0.5)
            # Format data
            bboxs = list(bboxs)
            confidences = list(np.array(confidences).reshape(1,-1)[0])
            confidences = list(map(float, confidences))
            bboxIdx = cv2.dnn.NMSBoxes(bboxs, confidences, score_threshold=0.5, nms_threshold=0.2)

            for i in range(0, len(bboxIdx)):
                # Reduce single dimensions & get information at index i
                    bbox = bboxs[np.squeeze(bboxIdx[i])]
                    classConfidence = confidences[np.squeeze(bboxIdx[i])]
                    classLabelID = np.squeeze(ids[np.squeeze(bboxIdx[i])])
                    classLabel = self.classesList[classLabelID]
                    # Get x, y, width and height of the bounding box at index i
                    x,y,w,h = bbox
                    # Use built-in cv2 method to draw the bounding box around the detected object
                    # Make text to display over each box
                    text = "{}:{:.4f}".format(classLabel, classConfidence)
                    # Convert colors into an integer list
                    cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_COMPLEX, 1, 2)

                    '''Show Kalman Filter Prediction on Object'''
                    p_x, p_y = kf.predict(x,y)
                    p_x, p_y = p_x[0], p_y[0]
                    # Get gradient of vision and kalman filter based on probability
                    f_x, f_y = gradient((x,y), (p_x,p_y), classConfidence)
                    f_x, f_y = int(f_x), int(f_y)
                    cv2.rectangle(frame, (f_x, f_y), (f_x+w, f_y+h), color=(255,255,255), thickness=1)

            '''Proceed to next frame with user input'''
            cv2.imshow("Frame", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'): break
        
        cv2.destroyAllWindows()
