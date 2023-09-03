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
    deteccoes = detector_face.detectMultiScale(imagem_cinza,minNeighbors=4)
    #identificando a altura e largura da imagem
    altura, largura = imagem_cinza.shape

    #obtendo os valores do meio da imagem
    centro_x = largura // 2
    centro_y = altura // 2 

    for(x,y,w,h) in deteccoes:
        #print(w,h)
        posicao = ""
        
        centro_rostox = x + w // 2
        centro_rostoy = y + h // 2

        #câmera invertida, as condições irão mostrar a posição real
        if centro_rostox > centro_x and centro_rostoy > centro_y:
            posicao = "Rosto no lado inferior esquerdo"
        elif centro_rostox > centro_x and centro_rostoy < centro_y:
            posicao = "Rosto no lado superior direito"
        elif centro_rostox < centro_x and centro_rostoy > centro_y:
            posicao = "Rosto no lado inferior direito"
        elif centro_rostox < centro_x and centro_rostoy<centro_y:
            posicao = "Rosto no lado superior direito"
        else:
            posicao = "Rosto no centro"
        
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,205,255),2)
        cv2.putText(frame,posicao,(x,y-5),cv2.FONT_HERSHEY_COMPLEX,0.7,(0,205,255),3)


    cv2.imshow('Janela',frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()