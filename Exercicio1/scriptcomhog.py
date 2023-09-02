import os
import cv2
import dlib



dir = os.path.dirname(os.path.abspath(__file__))

caminho_imagem = os.path.join(dir,'./imagens/people3.jpg')

imagem = cv2.imread(caminho_imagem)

detector_face_hog = dlib.get_frontal_face_detector()
deteccoes = detector_face_hog(imagem, 4)

print("detecções:" + str(len(deteccoes)))
for face in deteccoes:
   l,t,r,b = face.left(),face.top(),face.right(),face.bottom()
   cv2.rectangle(imagem, (l,t),(r,b),(255,0,0),2)

cv2.imshow('janela2',imagem)
cv2.waitKey(0)
cv2.destroyAllWindows()

