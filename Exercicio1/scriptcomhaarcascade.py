import os
import cv2

dir = os.path.dirname(os.path.abspath(__file__))

caminho_imagem = os.path.join(dir,'./imagens/people3.jpg')

imagem = cv2.imread(caminho_imagem)
imagem = cv2.resize(imagem,(1400,800))
imagem_cinza = cv2.cvtColor(imagem,cv2.COLOR_BGR2GRAY)

caminho_classificador = os.path.join(dir,'./cascades/haarcascade_frontalface_default.xml')
classificador_faces = cv2.CascadeClassifier(caminho_classificador)
deteccoes_faces = classificador_faces.detectMultiScale(imagem_cinza,scaleFactor=1.13,minNeighbors=1)

print("Detecções:" + str(len(deteccoes_faces)))

for x,y,w,h in deteccoes_faces:
    print(x,y)
    cv2.rectangle(imagem,(x,y),(x+w,y+h),(0,0,255),3)

cv2.imshow('janela',imagem)
cv2.waitKey(0)
cv2.destroyAllWindows()
