import cv2
import numpy as np


def extract_blood_vessels(image_path):
    # Carregar a imagem original
    original_image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

    # Realce de contraste
    contrast_enhanced = cv2.equalizeHist(gray_image)

    # Detecção de bordas
    edges = cv2.Canny(contrast_enhanced, 50, 150, apertureSize=3)

    # Detecção de círculos usando a Transformada de Hough
    circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, dp=1, minDist=100, param1=50, param2=30, minRadius=0,
                               maxRadius=0)

    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            # Desenhar o círculo externo
            cv2.circle(original_image, (i[0], i[1]), i[2], (0, 255, 0), 2)
            # Desenhar o centro do círculo
            cv2.circle(original_image, (i[0], i[1]), 2, (0, 0, 255), 3)

    # Salvar e/ou mostrar a imagem
    cv2.imwrite('blood_vessels_detected.jpg', original_image)

    # Se você quiser mostrar a imagem (descomente as duas linhas abaixo)
    # cv2.imshow('Blood Vessels', original_image)
    # cv2.waitKey(0)


extract_blood_vessels('bases/img_3.png')
