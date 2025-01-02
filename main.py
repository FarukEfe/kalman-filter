from Detector import Detector
import os

def main():
    videoPath = 0
    # It's a good practice to use os.path.join when getting a file directory
    # since it's avoiding potential directory issues of hardcoding the directory path
    configPath = os.path.join("model_data","ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt")
    modelPath = os.path.join("model_data", "frozen_inference_graph.pb")
    classesPath = os.path.join("model_data", "coco.names")

    detector = Detector(videoPath, configPath, modelPath, classesPath)
    detector.onVideo()

if __name__ == '__main__':
    main()
