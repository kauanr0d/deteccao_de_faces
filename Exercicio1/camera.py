import dlib
import cv2
import os

dir = os.path.dirname(os.path.abspath(__file__))
caminho_detector = os.path.join(dir,'./cascades/haarcascade_frontalface_default.xml')
detector_face = cv2.CascadeClassifier(caminho_detector)

#recebendo dados da webcam:

video_capture = cv2.VideoCapture(0)

while True:
    ok, frame = video_capture.read()

    imagem_cinza = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    deteccoes = detector_face.detectMultiScale(imagem_cinza,minNeighbors=3,minSize=(70,60))
    for(x,y,w,h) in deteccoes:
        print(w,h)
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,205,255),5)
    cv2.imshow('Janela',frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()