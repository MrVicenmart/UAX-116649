# -*- coding: utf-8 -*-
"""
Created on Sun Jul 9 07:38:40 2020

@author: MrVic
"""

'''
Algoritmo Genético para la Optimización de Perfiles Aerodinámicos:
    
Se tiene una población que puede tener perfiles NACA y material diferente.

Los perfiles NACA serán de 4 dígitos, a elegir entre 28 posibles perfiles. (desde 00000 a 11011 en binario)
El material será: madera (00), aluminio (01), material compuesto (10) y titanio (11).

Esto nos da cinco posiciones binarias en perfiles NACA y dos en material. En total serán 7 posiciones binarias.
Se ordenará de la siguiente manera: NACA/material

En total existen 112 posibilidades diferentes dentro de mi población.
'''

# Importo librerías
import random
import importlib
import aeropy
import numpy as np
from numpy.random import default_rng
from aeropy.xfoil_module import *
from aeropy.aero_module import *

# Constantes Relativas al Código Genético
largo = 7 # Longitud del material genético (No se puede cambiar)
num_poblacion = 8 # Población de prueba sobre la que se realiza el algoritmo genético, en este caso son 10 de las posibles 112 soluciones
seleccionados = 4 # Cantidad de padres que tendrán hijos. Número par siempre

# Constantes Relativas a la Aerodinámica
Mach_dado = 0.27 # Para hacer el estudio en régimen incompresible y simplificar cálculos.
Re_dado = 5e5 # Ya que el programa y el estudio preliminar se realiza a bajos números de Reynolds.
angulo_ataque = 8
perfiles_naca = ['naca0006','naca0008','naca0009','naca0010','naca0012','naca0015','naca0018','naca0021','naca0024','naca1408','naca1410','naca1412','naca2408','naca2410','naca2411','naca2412','naca2414','naca2415','naca2418','naca2421','naca2424','naca4412','naca4415','naca4418','naca4421','naca4424','naca6409','naca6412'] # Perfiles entre los que es posible elegir, se pueden poner otros. Si se añaden más de 28 tipos diferentes hay que modificar el código genético.
# El primer perfil es el naca0006 que corresponde con el número binario 00000 y el último es el perfil naca6412 que corresponde al 11011

espesor_naca = [] # Últimos dos dígitos de los perfiles_naca obtenidos automáticamente. Recomiendo no tocar esto.
for i in perfiles_naca:
    espesor_naca.append(i[-2:])
    espesor_array = np.array(espesor_naca,dtype=float)
    
curvatura_naca = [] # Primer digíto de los perfiles_naca obtenidos automáticamente. Recomiento no tocar esto.
for i in perfiles_naca:
    curvatura_naca.append(i[4:-3])
    curvatura_array = np.array(curvatura_naca,dtype=float)

# Constantes Relativas a los Materiales
materiales = ['madera','aluminio','material compuesto','titanio'] # (00, 01, 10, 11)
resistencia_material = np.array([15,300,420,900]) # [MPa] Resistencia del material // Las valores corresponden a los materiales que estén en la misma posición del array.
densidad_material = np.array([1000,2700,1900,4500]) # [kg/m^3] Densidad del material // Las valores corresponden a los materiales que estén en la misma posición del array.
coste_material = np.array([0.82,1.86,7.52,10.20]) # [€/kg] del material // Las valores corresponden a los materiales que estén en la misma posición del array.
dificultad_mecanizado = np.array([4,4,10,20]) # [-] Precio de mecanizado del material (siendo 1 facilidad de mecanizado y siendo 20 dificultad alta de mecanizado) // Las valores corresponden a los materiales que estén en la misma posición del array.

# Constantes Relativas a la Función de Calidad. Indicar la importancia que se le quiere dar a cada uno. Tener en cuenta que este valor multiplica cada uno de los sumandos a los que va asociado.

'''
Si es una empresa tipo NASA:
p_eficiencia = 50
p_resistencia_peso = 50
p_precio = 0

Si es una empresa tipo aerolínea comercial:
p_eficiencia = 38
p_resistencia_peso = 22
p_precio = 40

Si es una empresa tipo avioneta regional:
p_eficiencia = 12
p_resistencia_peso = 9
p_precio = 79
'''

p_eficiencia = 50 # Indica el peso que tiene la eficiencia aerodinámica en la función de calidad
p_resistencia_peso = 50 # Indica el peso que tiene el peso en la función de calidad
p_precio = 0 # Indica el peso que tiene el precio que conlleva usar ese perfil y ese material

# Valores máximos para la función de Calidad
def eficienciaMaxima(): # Hay que saber que perfil es el que da máxima eficienia aerodinámica entre todos a mismo ángulo de ataque. En mi caso es el NACA6412 para alfa=8 y el NACA6409 para el alfa=4.
    aeropy.xfoil_module.call(airfoil='naca6412', alfas=angulo_ataque,output='Polar',Reynolds=Re_dado,Mach=Mach_dado,iteration=50)
    filename = file_name(airfoil='naca6412',alfas=angulo_ataque,reynolds=Re_dado,output='Polar')
    coeficientes = aeropy.xfoil_module.output_reader(filename, output='Polar')
    cl = coeficientes['CL']
    cl_array = np.array(cl,dtype=float)
    cd = coeficientes['CD']
    cd_array = np.array(cd,dtype=float)
    
    return cl_array/cd_array

relacion_resistencia_peso_maxima = resistencia_material[2]/densidad_material[2] # La mejor relación resistencia-peso es la ejercida por el material compuesto
peso_maximo = np.amax(densidad_material)*np.amax(espesor_array) # Corresponde al máximo peso
precio1_maximo = np.amax(coste_material)*peso_maximo # Corresponde al precio máximo del material.
precio2_maximo = np.amax(dificultad_mecanizado)*np.amax(curvatura_array) # Corresponde con la dificultad de mecanizado del titanio multiplicado por la máxima curvatura.
precio_maximo = precio1_maximo+precio2_maximo # Corresponde al precio máximo incluyendo la dificultad de mecanizado.

# Creación de individuos
def crearIndividuos(num_individuos):
    nuevos_individuos = np.random.randint(2, size=(num_individuos,largo))
    
    return nuevos_individuos
    
# Obtención genotipo superficie
def sacarbinarioNaca(individuos):
    genotipoNaca = individuos[0:5]

    return genotipoNaca

# Obtención genotipo color
def sacarbinarioMaterial(individuos):
    genotipoMaterial = individuos[5:7]

    return genotipoMaterial
      
# Obtener el perfil y material óptimo en texto, no puede entrar una matriz.
def sacarPerfilMaterial(individuo):
    binario_perfil = int(sacarbinarioNaca(i)[0]*10000+sacarbinarioNaca(i)[1]*1000+sacarbinarioNaca(i)[2]*100+sacarbinarioNaca(i)[3]*10+sacarbinarioNaca(i)[4]*1)
    posicion_perfil = int(str(binario_perfil),2)
    binario_material = int(sacarbinarioMaterial(i)[0]*10+sacarbinarioMaterial(i)[1]*1)
    posicion_material = int(str(binario_material),2)
    cromosoma = binario_perfil*100+binario_material
    
    informacion = 'Cromosoma: {0:} con puntuación de: {1:.3f}. Su perfil es: {2:} y su material es: {3:}'
    
    return informacion.format(cromosoma, individuo[largo], perfiles_naca[posicion_perfil], materiales[posicion_material]) # Si lo que imprime por pantalla como cromosoma tiene menos dígitos de 7, hay que añadir ceros a la izquierda hasta tener 7 dígitos.
    
# Si no existen, se eliminan, y se crea un sustituo que exista
def sustitucionIndividuo(matriz_a_sustituir):
    for i in matriz_a_sustituir:
        while i[0] == 1 and i[1] == 1 and i[2] == 1:
            i[2] = 0

    return matriz_a_sustituir

# Función de calidad
def calcularFitness(individuos):
    fila=0
    num_rows = np.shape(individuos)[0]
    matriz_anexa = np.zeros((num_rows,1))
    
    for i in individuos:
        binario_perfil = int(sacarbinarioNaca(i)[0]*10000+sacarbinarioNaca(i)[1]*1000+sacarbinarioNaca(i)[2]*100+sacarbinarioNaca(i)[3]*10+sacarbinarioNaca(i)[4]*1)
        posicion_perfil = int(str(binario_perfil),2)
        binario_material = int(sacarbinarioMaterial(i)[0]*10+sacarbinarioMaterial(i)[1]*1)
        posicion_material = int(str(binario_material),2)

        aeropy.xfoil_module.call(airfoil=perfiles_naca[posicion_perfil],alfas=angulo_ataque,output='Polar',Reynolds=Re_dado,Mach=Mach_dado,iteration=50)
        filename = file_name(airfoil=perfiles_naca[posicion_perfil],alfas=angulo_ataque,reynolds=Re_dado,output='Polar')
        coeficientes = aeropy.xfoil_module.output_reader(filename, output='Polar')
        cl = coeficientes['CL']
        cl_array = np.array(cl,dtype=float)
        cd = coeficientes['CD']
        cd_array = np.array(cd,dtype=float)
        
        # Cada uno de los siguientes sumandos están adimensionalizados y normalizados para que varíen de 0 a 1.
        eficiencia_a = float((cl_array/cd_array)/eficienciaMaxima())
        resistencia_peso_a = (resistencia_material[posicion_material]/densidad_material[posicion_material]*np.amin(espesor_array))/(relacion_resistencia_peso_maxima*espesor_array[posicion_perfil])
        peso_total = densidad_material[posicion_material]*espesor_array[posicion_perfil]
        precio1 = 1-coste_material[posicion_material]*peso_total/(np.amax(coste_material)*np.amax(densidad_material)*np.amax(espesor_array))
        precio2 = 1-dificultad_mecanizado[posicion_material]*curvatura_array[posicion_perfil]/precio2_maximo
        precio_a = (precio1+precio2)/2
        
        fitness = p_eficiencia*eficiencia_a+p_resistencia_peso*resistencia_peso_a+p_precio*precio_a
        
        #print(coeficientes, perfiles_naca[posicion_perfil])
        #print(fitness, eficiencia, material, mecanizado)
        
        matriz_anexa[fila,0] = fitness
        fila+=1
        
    matriz_puntuada = np.hstack((individuos,matriz_anexa))
    
    return matriz_puntuada

# Se ordena la matriz, de menor a mayor y después de mayor a menor
def ordenarMatriz(poblacion):
    matriz_puntuada_ordenada_a =  poblacion[np.argsort(poblacion[:,largo])]
    num_rows = np.shape(poblacion)[0]
    fila=0
    matriz_puntuada_ordenada = np.zeros((num_rows,largo+1))
    for i in matriz_puntuada_ordenada:
        matriz_puntuada_ordenada[fila,:] = matriz_puntuada_ordenada_a[num_rows-fila-1,:]
        fila+=1
        
    return matriz_puntuada_ordenada_a, matriz_puntuada_ordenada

# Se seleccionan los padres por elitismo para generar los hijos en el siguiente paso
def seleccion(poblacion):
    matriz_seleccionados = np.zeros((seleccionados,largo))
    fila=0
    while fila <= seleccionados-1:
        matriz_seleccionados[fila,:] = poblacion[fila,0:largo]
        fila+=1
        
    return matriz_seleccionados

# Se procede a crear la descendencia
def cruzamiento(poblacion_seleccionada):
    descendientes = int(seleccionados/2)
    orden_cruzarse = default_rng().choice(seleccionados, size=seleccionados, replace=False)
    poblacion_preparada = np.zeros((seleccionados,largo))
    poblacion_descendiente = np.zeros((descendientes,largo))
    
    # Se ha ordenado la matriz para crear descendencia de padres aleatorios.
    n=0
    while n < seleccionados:
        poblacion_preparada[orden_cruzarse[n],:] = poblacion_seleccionada[n,:]
        n+=1
    
    t=0
    while t < descendientes:
        s=0
        while s < largo:
            a = random.random()
            if a < 0.5:
                ganador = 2*t
            else:
                ganador = 2*t+1
            poblacion_descendiente[t,s] = poblacion_preparada[ganador,s]
            s+=1
        t+=1
    
    return poblacion_preparada, sustitucionIndividuo(poblacion_descendiente)

# Se procede a mutar a algunos individuos aleatorios no seleccionados
def mutacion(poblacion_inicial, ratio): # El ratio debe estar dado entre 0 y 1. Si es 1, es un 100% de probabilidad de mutar
    posibles_mutados = np.copy(poblacion_inicial[seleccionados:num_poblacion,0:largo])
    matriz_mutados = np.empty((0,largo),int)

    for i in posibles_mutados:
        if ratio > random.random(): # Muta
            punto = random.randint(0,largo-1) #Se elige un punto al azar
            nuevo_valor = random.randint(0,1) #y un nuevo valor para este punto
            
            while nuevo_valor == i[punto]: #Se comprueba que el nuevo valor no sea igual al viejo
                nuevo_valor = random.randint(0,1)

            i[punto] = nuevo_valor #Se aplica la mutación
            
            #Si la mutación forma un individuo con los tres primeros dígitos que sean unos, se modifica la tercera posición para cambiar de 1 a 0.
            while i[0] == 1 and i[1] == 1 and i[2] == 1:
                i[2] = 0
           
            matriz_mutados = np.append(matriz_mutados,np.array([i]), axis=0)
        
    return poblacion_inicial[seleccionados:num_poblacion,0:largo], matriz_mutados

# Se matan a individuos aleatorios (el único inmune a morir es el mejor puntuado) con tal de generar nuevos individuos e incorporar nuevos individuos aleatorios.
def muerteAleatoria(poblacion_final, ratio, cantidad_muertos): # El ratio debe estar dado entre 0 y 1. Si es 1, es un 100% de probabilidad de matar en cada generación la cantidad de muertos especificada.
    b = random.random()
    vivos = num_poblacion-cantidad_muertos    
    a = random.sample(range(1,num_poblacion-1),vivos-1)
    poblacion = np.empty((0,largo),int)
    
    if ratio > b:
        for i in a:
            poblacion = np.append(poblacion, np.array([poblacion_final[i,0:largo]]), axis=0)
        nuevos = sustitucionIndividuo(crearIndividuos(cantidad_muertos))
        poblacion_resultante = ordenarMatriz(calcularFitness(np.vstack((poblacion_final[0,0:largo],poblacion,nuevos))))[1]
    else:
        poblacion_resultante = np.copy(poblacion_final)
    return poblacion_resultante

# Inicio:
m1 = crearIndividuos(num_poblacion) 
#print("Población Inicial:\n%s"%m1)

m2 = sustitucionIndividuo(m1[:,0:largo]) 
#print("Población Inicial Corregida:\n%s"%m2)

m3 = calcularFitness(m2)
#print("Población Inicial Puntuada:\n%s"%m3)
#m103 = np.around(m3, decimals=3)
#print("Población Inicial Puntuada:\n%s"%m103)

m4 = ordenarMatriz(m3)
#print("Poblacion Incial Puntuada Y Ordenada:\n%s"%m4[0])
#print("Poblacion Incial Puntuada Y Ordenada Correctamente:\n%s"%m4[1])
#print("Poblacion Inicial Aleatoria:\n%s"%m4[1]) 
m104 = np.around(m4, decimals=3)
#print("Poblacion Incial Puntuada Y Ordenada:\n%s"%m104[0])
#print("Poblacion Incial Puntuada Y Ordenada Correctamente:\n%s"%m104[1]) 
print("Poblacion Inicial Aleatoria Y Ordenada:\n%s"%m104[1])

m5 = seleccion(m4[1])
#print("Padres para la siguiente generación:\n%s"%m5)

m6 = cruzamiento(m5)
#print("Padres ordenados para la siguiente generación:\n%s"%m6[0])
#print("Hijos:\n%s"%m6[1])

m7 = mutacion(m2, 0.4)
#print("Posibles Mutados:\n%s"%m7[0])
#print("Mutados:\n%s"%m7[1])

m8 = np.vstack((m6[1],m7[1]))
#print("Nuevos Descendientes Añadidos:\n%s"%m8)

m9 = calcularFitness(m8)
#print("Nuevos Descendientes Puntuados:\n%s"%m9)
#m109 = np.around(m9, decimals=3)
#print("Nuevos Descendientes Puntuados:\n%s"%m109)

m10 = np.vstack((m3,m9))
#print("Población Antes de Matar:\n%s"%m10)
#m110 = np.around(m10, decimals=3)
#print("Población Antes de Matar:\n%s"%m110)

m11 = ordenarMatriz(m10)
#print("Nuevos Descendientes Puntuados Y Ordenada:\n%s"%m11[0])
#print("Nuevos Descendientes Puntuados Y Ordenada Correctamente:\n%s"%m11[1])
#m111 = np.around(m11, decimals=3)
#print("Nuevos Descendientes Puntuados Y Ordenada:\n%s"%m111[0])
#print("Nuevos Descendientes Puntuados Y Ordenada Correctamente:\n%s"%m111[1])

m12 = m11[1][0:num_poblacion,:]
#print("Población Después de una Generación:\n%s"%m12)
#m112 = np.around(m12, decimals=3)
#print("Población Después de una Generación:\n%s"%m112)

i=1

m13 = muerteAleatoria(m12,0.8,3)
#print("Población Después de una Pandemia:\n%s"%m13)
#print("Población Final",i,"º Iteración:\n%s"%m13)
m113 = np.around(m13, decimals=3)
#print("Población Después de una Pandemia:\n%s"%m113)
print("Población Final",i,"º Iteración:\n%s"%m113)

m_sustitucion = m13

i=2

while i <= 20: # Mi condición de parada es cuando lleve x iteraciones.
    '''
    m2 = sustitucionIndividuo(m_sustitucion[:,0:largo]) 
    print("Población Inicial Corregida:\n%s"%m2)
    '''
    m3 = calcularFitness(m_sustitucion[:,0:largo])
    #print("Población Inicial Puntuada:\n%s"%m3)
    #m103 = np.around(m3, decimals=3)
    #print("Población Inicial Puntuada:\n%s"%m103)
    
    m4 = ordenarMatriz(m3)
    #print("Poblacion Incial Puntuada Y Ordenada:\n%s"%m4[0])
    #print("Poblacion Incial Puntuada Y Ordenada Correctamente:\n%s"%m4[1])
    #m104 = np.around(m4, decimals=3)
    #print("Poblacion Incial Puntuada Y Ordenada:\n%s"%m104[0])
    #print("Poblacion Incial Puntuada Y Ordenada Correctamente:\n%s"%m104[1]) 
    
    m5 = seleccion(m4[1])
    #print("Padres para la siguiente generación:\n%s"%m5)
    
    m6 = cruzamiento(m5)
    #print("Padres ordenados para la siguiente generación:\n%s"%m6[0])
    #print("Hijos:\n%s"%m6[1])
    
    m7 = mutacion(m2, 0.4)
    #print("Posibles Mutados:\n%s"%m7[0])
    #print("Mutados:\n%s"%m7[1])
    
    m8 = np.vstack((m6[1],m7[1]))
    #print("Nuevos Descendientes Añadidos:\n%s"%m8)
    
    m9 = calcularFitness(m8)
    #print("Nuevos Descendientes Puntuados:\n%s"%m9)
    #m109 = np.around(m9, decimals=3)
    #print("Nuevos Descendientes Puntuados:\n%s"%m109)
    
    m10 = np.vstack((m3,m9))
    #print("Población Antes de Matar:\n%s"%m10)
    #m110 = np.around(m10, decimals=3)
    #print("Población Antes de Matar:\n%s"%m110)
    
    m11 = ordenarMatriz(m10)
    #print("Nuevos Descendientes Puntuados Y Ordenada:\n%s"%m11[0])
    #print("Nuevos Descendientes Puntuados Y Ordenada Correctamente:\n%s"%m11[1])
    #m111 = np.around(m11, decimals=3)
    #print("Nuevos Descendientes Puntuados Y Ordenada:\n%s"%m111[0])
    #print("Nuevos Descendientes Puntuados Y Ordenada Correctamente:\n%s"%m111[1])
    
    m12 = m11[1][0:num_poblacion,:]
    #print("Población Después de una Generación:\n%s"%m12)
    #m112 = np.around(m12, decimals=3)
    #print("Población Después de una Generación:\n%s"%m112)
    
    m13 = muerteAleatoria(m12,0.8,3)
    #print("Población Después de una Pandemia:\n%s"%m13)
    #print("Población Final",i,"º Iteración:\n%s"%m13)
    m113 = np.around(m13, decimals=3)
    #print("Población Después de una Pandemia:\n%s"%m113)
    print("Población Final",i,"º Iteración:\n%s"%m113)
    
    m_sustitucion = m13
    i+=1
    
for i in m13:
    print(sacarPerfilMaterial(i))