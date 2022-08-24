from deepface import DeepFace
import numpy as np

import cv2
vidcap = cv2.VideoCapture('video/Stephen Curry Interview (Game 5 Preview) _ Warriors vs Celtics _ 2022 NBA Finals.mp4')

assert vidcap.isOpened()

fps_in = vidcap.get(cv2.CAP_PROP_FPS)
fps_out = 3

index_in = -1
index_out = -1

AnalyResult = []
while True:
    success = vidcap.grab()
    if not success: break
    index_in += 1

    out_due = int(index_in / fps_in * fps_out)
    if out_due > index_out:
        success, frame = vidcap.retrieve()
        if not success: break
        index_out += 1

        # do something with `frame`
        try:
          predictions = DeepFace.analyze(frame)

          print(predictions)
          AnalyResult.append(predictions)
        except:
          pass


import pickle


# save
with open('faceAnalysisData.pickle', 'wb') as f:
    pickle.dump(AnalyResult, f, pickle.HIGHEST_PROTOCOL)

# load
# with open('faceAnalysisData.pickle', 'rb') as f:
#     data = pickle.load(f)