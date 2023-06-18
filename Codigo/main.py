
import sys
import cv2
from functions import CodeEPP
import torch
import numpy as np

# cargar el modelo
model = torch.hub.load('..\\Codigo_pytorch\\ObjectDetection\\yolov5_m',
                'custom',path='Codigo\\peso_train\\best.pt', source='local')

# se obtiene las funciones de la clase implementada
modulesEPP = CodeEPP()

# configuracion de video
cap = cv2.VideoCapture('..\\videos\\VALIDACION\\LISTO\\caso1.mp4')


AlertasGeneradas = 0
tiempoAlertaTotal = [10,10,10,10] # total de frames que se analiza para generar alerta
conf_eachEPP = [0.7,0.7,0.7,0.7,0.7] # establece la confiabilidad de cada EPP para la deteccion

contAlertaCasco  = np.array([])
contAlertaLentes = np.array([])
contAlertaGuantes= np.array([])
contAlertaBotas  = np.array([])


alertasSignalVisual = [0,0,0,0] #casco,lentes,guantes,botas

while True:
    ret, img = cap.read()
    if ret:

        #------------------Relacion entre etiquetas y valores asignados (codificados)
        # casco       0
        # trabajador  1
        # guantes     2
        # botas       3
        # lentes      4

        # no_guantes  5
        # no_casco    6
        # no_lentes   7
        # no_botas    8

        # obtiene resultado de detecciones
        results = model(cv2.cvtColor(img,cv2.COLOR_BGR2RGB), size=800)
        
        # obtiene coordenadas de trabajadores y no epp con el nivel de confianza de cada uno
        Workers, NoEpps = modulesEPP.SearchNoEpp(results, [5,6,7,8], conf_eachEPP)
        
        # obtiene alerta_cada_epp (1 o 0 para cada EPP) y coordendas de los EPP que cumplen la regla de explicita
        alert_each_NoEPP, noEPPs_verified = modulesEPP.RulerDetection(Workers, NoEpps)


        # analisis para generacion de alerta
        if np.shape(contAlertaCasco)[0] < int(tiempoAlertaTotal[0]):
            contAlertaCasco = np.append(contAlertaCasco,int(alert_each_NoEPP[0]))

        else:
            if np.sum(contAlertaCasco)> int(int(tiempoAlertaTotal[0])*0.8):

                alertasSignalVisual[0] = 1

                #self.modulesEPP.PublishAlert(canal_rabbit,'WARNING : Trabajador no usa EPP --> CASCO')

            else:
                alertasSignalVisual[0] = 0

            contAlertaCasco[0] = int(alert_each_NoEPP[0])
            contAlertaCasco = np.roll(contAlertaCasco,-1)

        #print(contAlertaCasco)
        #--------------------------------------------------------------------

        if np.shape(contAlertaLentes)[0] < int(tiempoAlertaTotal[1]):
            contAlertaLentes = np.append(contAlertaLentes,int(alert_each_NoEPP[1]))

        else:
            if np.sum(contAlertaLentes)> int(int(tiempoAlertaTotal[1])*0.8):
                #print("SE GENERA ALERTA POR LENTES : ",AlertasGeneradasL)

                alertasSignalVisual[1] = 1

                #modulesEPP.PublishAlert(canal_rabbit,'WARNING : Trabajador no usa EPP --> LENTES')
            else:
                alertasSignalVisual[1] = 0

            contAlertaLentes[0] = int(alert_each_NoEPP[1])
            contAlertaLentes = np.roll(contAlertaLentes,-1)

        #----------------------------------------------------------------------

        if np.shape(contAlertaGuantes)[0] < int(tiempoAlertaTotal[2]):
            contAlertaGuantes = np.append(contAlertaGuantes,int(alert_each_NoEPP[2]))

        else:
            if np.sum(contAlertaGuantes)> int(int(tiempoAlertaTotal[2])*0.8):
                #print("SE GENERA ALERTA POR GUANTES : ",AlertasGeneradasG)

                alertasSignalVisual[2] = 1
                #modulesEPP.PublishAlert(canal_rabbit,'WARNING : Trabajador no usa EPP --> GUANTES')
            else:
                alertasSignalVisual[2] = 0

            contAlertaGuantes[0] = int(alert_each_NoEPP[2])
            contAlertaGuantes = np.roll(contAlertaGuantes,-1)


        # grafica los boxes en la imagen con los valores de confiabilidad y el nombre
        img_labels = modulesEPP.LabelNoEpp(noEPPs_verified, img)
