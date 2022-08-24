# 개발완료했으나 맥의 m1 processor 는 실행하는데 문제가 있다는 것을 발견함
# 실행하면...

# (py37OpenCV) kimkwangil@kimuiMacBookPro Finsh_RealtimeFacialExpression % python FacialExpRecog_realTime.py
# (py37OpenCV) kimkwangil@kimuiMacBookPro Finsh_RealtimeFacialExpression % python RealtimeFacialExpRecog_img.py
# zsh: illegal hardware instruction  python RealtimeFacialExpRecog_img.py

# 그래서 pc에서 실행하며됨. 단, webcam이 장착되어 있어야 함


from deepface import DeepFace

import cv2
img = cv2.imread('img/face_01.png')
import matplotlib.pylab as plt

#plt.imshow(img)

#plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

predictions = DeepFace.analyze(img)

print(predictions)

faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = faceCascade.detectMultiScale(gray, 1.1, 4)

for(x,y,w,h) in faces:
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

#plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(img, predictions['dominant_emotion'],
           (50,50),
           font,
           3,
           (0, 0, 255),
           2,
           cv2.LINE_4);


plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

