
# coding: utf-8

# In[20]:



import os
import modulo_segmentacion as segmentacion
import modulo_CNN as CNN

def reconocer_matriculas (ruta_matriculas):
    total_aciertos = 0
    total_fallos = 0
    caracteres_distintos = 0
    print("kevin")
    for ruta, subdirs, ficheros in os.walk(ruta_matriculas):
        print(":)")
        subdirs.sort()
        for nombre_fichero in ficheros:
            ruta_completa = os.path.join(ruta, nombre_fichero)
            print(ruta_completa)
            contenido_matricula = nombre_fichero.rsplit('.', 1)[0]
            caracteres_matricula = segmentacion.cargar_contenido(ruta_completa)
            matricula_reconocida = CNN.reconocer_matricula(caracteres_matricula)
            if contenido_matricula == matricula_reconocida:
                print ("\nCORRECTO: ",contenido_matricula, " = ", matricula_reconocida)
                total_aciertos = total_aciertos + 1
            else:
                caracteres_distintos =                 sum(1 for x,y in zip(contenido_matricula,matricula_reconocida) if x != y)
                print ("\n* ERROR: ", contenido_matricula," <> ",matricula_reconocida)
                print ("* CARACTERES DISTINTOS: ", caracteres_distintos)
                total_fallos = total_fallos + 1
    print ("\n***************")
    print ("***************")
    print (" ACIERTOS:", total_aciertos)
    print (" FALLOS:", total_fallos)

#cargar_contenido("Matricula/LICENCIA.jpg")
######## LLAMADA PRINCIPAL ###########
#reconocer_matriculas("c:/Kevin/ReconocerPlaca/Matricula/")
#reconocer_matriculas("matriculas/reales/")
#reconocer_matriculas("matriculas/usa/")



