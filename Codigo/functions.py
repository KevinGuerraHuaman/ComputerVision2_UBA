
import numpy as np
import cv2
import pika

class CodeEPP:
    
    def __init__(self):
        pass

    def SearchNoEpp(self,result_tensor,list_class_no_epp,confianzas,class_work=1):
        """Busca y obtiene las cordenadas de los trabajadores y los no epp

        Args:
            result_tensor ( tensor )       : Resultado de prediccion de model()
            class_work ( int )             : Clase identificador del trabajador
            list_class_no_epp ( list(int) ): Clases de los NO EPP

        Returns:
            flag_no_epp( bool )                  : Detecta si encontro un NO EPP pero falta confirmar si lo tiene puesto
            workersXY( list(float) )             : Almacena coordenadas de cada trabajador
            eppsXY( dict('no_epp',coordenadas) ) : Coordenadas x0,y0,x1,y1
        """
        
        # obtiene resultado y convierte a Array
        results_array = np.array(result_tensor.xyxy[0])
        
        #      xmin    ymin    xmax   ymax  confidence  class    name
        #   749.50   43.50  1148.0  704.5    0.874023      6  no_casco

        workersXY = []
        eppsXY=[]
        
        
        for info_detect in results_array:
            if (int(info_detect[5]) == class_work) and float(info_detect[4])>float(confianzas[0]):
                workersXY.append(info_detect[0:4]) # almacena coordenadas del trabajador
            
            elif int(info_detect[5]) in list_class_no_epp:
                
                if int(info_detect[5])==6  and float(info_detect[4])>float(confianzas[1]):
                    eppsXY.append([[info_detect[5]],info_detect[0:5]]) # almacena coordenadas y confianza de EPP
 
                elif int(info_detect[5])==7 and float(info_detect[4])>float(confianzas[2]):
                    eppsXY.append([[info_detect[5]],info_detect[0:5]])

                elif int(info_detect[5])==5 and float(info_detect[4])>float(confianzas[3]):
                    eppsXY.append([[info_detect[5]],info_detect[0:5]])
                    
                elif int(info_detect[5])==8 and float(info_detect[4])>float(confianzas[4]):
                    eppsXY.append([[info_detect[5]],info_detect[0:5]])
                    
                    
        return workersXY, eppsXY
    
    
    def norma_box(self, w, epp):
        """_summary_

        Args:
            w (array)   : Arreglo de coordenadas de trabajador
            epp (array) : Arreglo de coordenadas de EPP o no EPP

        Returns:
            ndarray     : Matriz 2x2 espacio 2D
        """
        
        ancho,alto = w[2]-w[0],w[3]-w[1]

        mat_out = np.dot(np.array([[1/ancho,0],[0,1/alto]]),np.array([[epp[0],epp[2]],[epp[1],epp[3]]])-np.array([[w[0],w[0]],[w[1],w[1]]]))

        return mat_out
    
    
    def RulerDetection(self,workers,no_epps):
        
        """ Aplica reglas cada EPP o no EPP a cada trabajador
        
        Args:
            workers ( list(array) )     : Lista de coordenadas de cada trabajor encontrado
            no_epps ( list(array) )     : Lista de coordenadas y confianza de no EPP encontrado
            confianzaEPP ( int )        : Porcentaje de confianza
            
        Returns:
            alerta ( bool )                 : Si un EPP o no EPP cumple con la regla
            no_epps_worker ( list(array) )  : Lista de coordenadas de no EPP cumple regla
        """
        
        alertasEPP=[0,0,0,0]
        no_epps_worker = []
        
        for worker in workers:
            #new_xy = norma_box(worker[0],worker[1],worker[2],worker[3],info_epp[0],info_epp[1],info_epp[2],info_epp[3])
            for no_epp in no_epps:
            
                new_xy = self.norma_box(worker,no_epp[1]) # nuevas coordenadas en espacio unitario del epp
                
                if (new_xy[0][0]>-0.01 and new_xy[0][0]<1.1 and new_xy[1][0]>-0.1 and new_xy[1][0]<1.1) and (new_xy[0][1]<1 and new_xy[0][1]>0 and new_xy[1][1]>-0.1 and new_xy[1][1]<1):
                    no_epps_worker.append(no_epp)
                    
                    # no_casco    6
                    if int(no_epp[0][0])==6:
                        alertasEPP[0] = 1
                        alertasEPP[1] = 0
                        alertasEPP[2] = 0
                        alertasEPP[3] = 0
                    # no_lentes   7
                    elif int(no_epp[0][0])==7:
                        alertasEPP[1] = 1
                        alertasEPP[0] = 0
                        alertasEPP[2] = 0
                        alertasEPP[3] = 0
                    # no_guantes  5
                    elif int(no_epp[0][0])==5:
                        alertasEPP[2] = 1
                        alertasEPP[0] = 0
                        alertasEPP[1] = 0
                        alertasEPP[3] = 0
                    # no_botas    8
                    elif int(no_epp[0][0])==8:
                        alertasEPP[3] = 1
                        alertasEPP[0] = 0
                        alertasEPP[1] = 0
                        alertasEPP[2] = 0
                    else:
                        alertasEPP[0] = 0
                        alertasEPP[1] = 0
                        alertasEPP[2] = 0
                        alertasEPP[3] = 0
                else:
                    alertasEPP[0] = 0
                    alertasEPP[1] = 0
                    alertasEPP[2] = 0
                    alertasEPP[3] = 0
        return alertasEPP,no_epps_worker

    def LabelNoEpp(self,no_epps_w, img):
        
        """ Etiqueta la imagen procesada con los EPP o no EPP confirmados
        
        Args:
            no_epps_w ( list(array) ) : Lista de coordenadas de cada no EPP confirmado con reglas
            img ( list(array) )       : Imagen original leida

        Returns:
            img ( ndarray ): Imagen con etiquetas
        """
        
        # casco       0
        # trabajador  1
        # guantes     2
        # botas       3
        # lentes      4
        
        # no_guantes  5
        # no_casco    6
        # no_lentes   7
        # no_botas    8
        
        
        for no_epp in no_epps_w:
            
            # grafica el rectangulo y coloca el nombre de etiqueta
            if int(no_epp[0][0]) == 5:
                cv2.rectangle(img,(int(no_epp[1][0]),int(no_epp[1][1])),(int(no_epp[1][2]),int(no_epp[1][3])),(14,188,236),thickness=2)
                cv2.putText(img, 'NO GUANTES' , (int(no_epp[1][0]),int(no_epp[1][1])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (14,188,236), 2)
                
            elif int(no_epp[0][0]) == 6:
                cv2.rectangle(img,(int(no_epp[1][0]),int(no_epp[1][1])),(int(no_epp[1][2]),int(no_epp[1][3])),(223,25,159),thickness=2)
                cv2.putText(img, 'NO CASCO' , (int(no_epp[1][0]),int(no_epp[1][1])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (223,25,159), 2)
                
            elif int(no_epp[0][0]) == 7:
                cv2.rectangle(img,(int(no_epp[1][0]),int(no_epp[1][1])),(int(no_epp[1][2]),int(no_epp[1][3])),(10,212,86),thickness=2)
                cv2.putText(img, 'NO LENTES' , (int(no_epp[1][0]),int(no_epp[1][1])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (10,212,86), 2)
            
            elif int(no_epp[0][0]) == 8:
                cv2.rectangle(img,(int(no_epp[1][0]),int(no_epp[1][1])),(int(no_epp[1][2]),int(no_epp[1][3])),(4,226,213),thickness=2)
                cv2.putText(img, 'NO BOTAS' , (int(no_epp[1][0]),int(no_epp[1][1])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (4,226,213), 2)
            
        
        
        return img
    
