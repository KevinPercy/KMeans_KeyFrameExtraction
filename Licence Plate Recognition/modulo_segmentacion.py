
# coding: utf-8

# In[7]:


import cv2
import math
import numpy as np

def verificar_roi(r):
    # Relaciones de aspecto admitidas para los caracteres de la matrícula
    RELACION_ASPECTO = 0.6
    RELACION_ASPECTO_ERROR = 0.4
    RELACION_ASPECTO_MINIMA = 0.10
    RELACION_ASPECTO_MAXIMA = RELACION_ASPECTO + RELACION_ASPECTO *    RELACION_ASPECTO_ERROR
     # Altura mínima y máxima de los caracteres de la matrícula
    ALTURA_MINIMA = 28
    ALTURA_MAXIMA = 42
    ANCHURA_MAXIMA = 35
    ANCHURA_MINIMA = 10
    # Relación de aspecto actual de la ROI detectada
    relacion_aspecto_roi = float(r.shape[1])/r.shape[0]
    #print ("Aspecto ROI:", relacion_aspecto_roi)
    #print("Altura:", r.shape[0])
    #print("Anchura:", r.shape[1])
    # Comprobamos si cumple las restricciones
    if relacion_aspecto_roi >= RELACION_ASPECTO_MINIMA and     relacion_aspecto_roi < RELACION_ASPECTO_MAXIMA and     r.shape[0] >= ALTURA_MINIMA and r.shape[0] < ALTURA_MAXIMA and 	r.shape[1] >= ANCHURA_MINIMA and r.shape[1] < ANCHURA_MAXIMA:
        return True   
    else:
        return False
    
def centrar_roi(r):
    #holas
    ALTURA_ROI = 28
    ANCHURA_ROI = 32
    ANCHO_BORDE = 4

     # Redimensionar ROI a 28 píxeles de altura manteniendo su aspecto

    relacion_aspecto_roi = float(r.shape[1]) / float(r.shape[0])
    nueva_altura = ALTURA_ROI
    nueva_anchura = int((nueva_altura * relacion_aspecto_roi) + 0.5)
    roi_redim = cv2.resize(r, (nueva_anchura,nueva_altura))

     # Binarizar ROI

    ret,roi_thresh = cv2.threshold(roi_redim,127,255,cv2.THRESH_BINARY_INV)
     # Calcular las dimensiones del borde necesario para centrar ROI en 32x32px

    b_top = ANCHO_BORDE
    b_bottom = ANCHO_BORDE
    b_left = int((ANCHURA_ROI - nueva_anchura)/2)
    b_right = int((ANCHURA_ROI - nueva_anchura)/2)

     # Aplicar borde al ROI

    roi_borde = cv2.copyMakeBorder(roi_thresh,b_top,b_bottom,b_left,b_right,cv2.BORDER_CONSTANT,value=[0,0,0])
     # Redimensionar ROI a 32x32px

    roi_transformado = cv2.resize(roi_borde,(32,32))
    return roi_transformado

def cargar_contenido(ruta):
     # Área en píxeles que debe tener la imagen de la matrícula (235x50px)
    AREA_PIXELES = 11750.0
    ALTURA_MINIMA_CONTORNO = 28
    AREA_MINIMA_CONTORNO = 50

    imagen_original = cv2.imread(ruta)

     # Redimensionar imagen a resolución 242x54
    #print(float(imagen_original.shape[1]))
    r_aspecto = float(imagen_original.shape[1]) / float(imagen_original.shape[0])
    nueva_altura = int(math.sqrt(AREA_PIXELES / r_aspecto) + 0.5)
    nueva_anchura = int((nueva_altura * r_aspecto) + 0.5)
    imagen_original = cv2.resize(imagen_original, (nueva_anchura,nueva_altura))

     # Preprocesamiento de la imagen para facilitar la detección de contornos

    imagen_grises = cv2.cvtColor(imagen_original,cv2.COLOR_BGR2GRAY)
    imagen_desenfocada = cv2.GaussianBlur(imagen_grises,(5,5),0)
    imagen_thresh = cv2.adaptiveThreshold(imagen_desenfocada,255,1,1,11,2)
     # Detección de contornos
    
    _,contornos,_ = cv2.findContours(imagen_thresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    caracteres = np.empty((0,1024))
    posicion_caracteres = []
    total_correcto = 0
    #print(contornos)
    for cnt in contornos:
        if cv2.contourArea(cnt) > AREA_MINIMA_CONTORNO:
            [x,y,w,h] = cv2.boundingRect(cnt)
            print(str(h)+" "+ str(ALTURA_MINIMA_CONTORNO))
            if h > ALTURA_MINIMA_CONTORNO:
                cv2.rectangle(imagen_original,(x,y),(x+w,y+h),(0,0,255),2)
                roi = imagen_desenfocada[y:y+h,x:x+w]
                if verificar_roi(roi):
                    print("hola kevin")
					# VISUALIZAR CONTORNOS PARA DEPURAR
                    cv2.imshow('Resultado',imagen_original)
                    cv2.waitKey(0)
                    total_correcto = total_correcto + 1
                    roi_transformado = centrar_roi(roi)
                    caracter = roi_transformado.reshape((1,1024))
                    caracteres = np.append(caracteres,caracter,0)
                    # Guardamos en el vector la coordenada "x" del caracter
                    posicion_caracteres.append(x)
     # Añadimos la columna con la coordenada "x" del caracter detectado
    caracteres = np.c_[caracteres,posicion_caracteres]
     # Ordenamos por la columna que guarda la posición
    caracteres = caracteres[caracteres[:,1024].argsort()]
     # Eliminamos la columna con la posición después de ordenar
    caracteres = np.delete(caracteres,np.s_[1024], axis = 1)
    return caracteres

cargar_contenido("c:\Kevin\ReconocerPlaca\Matricula\X2C268.jpg")

