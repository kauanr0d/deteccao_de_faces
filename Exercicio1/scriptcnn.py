import os 
import cv2
import dlib

dir = os.path.dirname(os.path.abspath(__file__))

caminho_imagem = os.path.join(dir,'./imagens/people3.jpg')
imagem = cv2.imread(caminho_imagem)
imagem_cinza = cv2.cvtColor(imagem,cv2.COLOR_BGR2GRAY)
caminho_weight = os.path.join(dir,'./weigths/mmod_human_face_detector.dat')
detector_faces = dlib.cnn_face_detection_model_v1(caminho_weight)

deteccoes = detector_faces(imagem_cinza,1)


for faces in deteccoes:
    l,t,r,b, c = faces.left(),faces.top(),faces.right(),faces.bottom(), faces.confidence()
    cv2.rectangle(imagem,(l,t),(r,b),(255,112,0),2)

cv2.imshow('Janela4', imagem)
cv2.waitKey(0)
