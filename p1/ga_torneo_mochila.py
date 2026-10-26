#Practica 1. Algoritmos Bionspirados 



#5BM1

import random
import numpy as np

productos = ["decoy_detonators" ,"love_potion", "extendable_ears" ,"skiving_snackbox" ,"fever_fudge", "puking_pastilles","nosebleed_nougat"]

pesos = [4, 2, 5, 5, 2, 1.5, 1]
gains = [10, 8, 12, 6, 3, 2, 2]

#P 2*10 + 5*2 = 30

#G 8*10 + 6*2 = 92



n_genes = 7
poblacion_inicial = 10
max_objetos = 10
generaciones = 50
capacidad_maxima = 30

def generar_poblacion_inicial():
    poblacion = []
    for _ in range(poblacion_inicial):
        
        for j in range(n_genes):
            k = (i+j)%7
            max_obj = int(capacidad // pesos[k])
            max_obj = min(max_obj, max_objetos)
            if max_obj > 0:
                individuo[k] += random.randint(0,max_obj)
                capacidad -= individuo[k]*pesos[k]
                if k == 1: capacidad += 6 # 
                if k == 3: capacidad += 10 #
        poblacion.append(individuo)
    return poblacion

def mutacion(ind):
    if random.random() <= 0.1:
        j = random.randint(0, n_genes -1)
        capacidad = int(max(0, (30 - np.dot(ind, pesos)) // pesos[j]))
        if j == 1:
            ind[j] = random.randint(3,capacidad) if capacidad >= 4 else 3
        elif j == 3:
            ind[j] = random.randint(2,capacidad) if capacidad >= 4 else 2
        else:
            ind[j] = random.randint(0,capacidad) if capacidad >= 1 else 0

def calcular_fitness(cromosoma):
    if cromosoma[1] < 3 or cromosoma[3] < 2 or np.dot(pesos,cromosoma) > 30:
        return 1
    return np.dot(gains,cromosoma)

def get_ruleta(generacion):
    ruleta = []
    fit = 0
    for individuo in generacion:
        fit += calcular_fitness(individuo)
        ruleta.append(fit)
    return ruleta

def girar_ruleta(ruleta):
    total = ruleta[-1]
    r = random.randint(1, total)
    i = 0
    while i < len(ruleta) and ruleta[i] < r: 
        i += 1
    return i

def reproducir(padre1, padre2):
    hijo1 = padre1.copy()
    hijo2 = padre2.copy()
    for i in range(7):
        if random.random() < 0.5:
            hijo1[i], hijo2[i] = hijo2[i], hijo1[i]
    return hijo1, hijo2

def reemplazar_debil(poblacion, nuevos_hijos):
    # Encontrar los 2 peores individuos
    fitness_poblacion = [(calcular_fitness(ind), idx) for idx, ind in enumerate(poblacion)]
    fitness_poblacion.sort()  # Ordenar de menor a mayor fitness
    
    # Los 2 peores serán reemplazados
    peores_indices = [fitness_poblacion[0][1], fitness_poblacion[1][1]]
    
    # Reempalzar los peores con los nuevos hijos
    nueva_poblacion = poblacion.copy()
    nueva_poblacion[peores_indices[0]] = nuevos_hijos[0]
    nueva_poblacion[peores_indices[1]] = nuevos_hijos[1]
    
    return nueva_poblacion

if __name__ == '__main__':
    poblacion = generar_poblacion_inicial()
    

    
    mejor_global = []
    mejor_fitness = 0

    print("\nEVOLUCION DE LAS GENERACIONES:")
    print("--" * 20)
    
    for gen in range(generaciones):
        ruleta = get_ruleta(poblacion)
        nuevos_hijos = []
        
        # Generar 2 nuevos hijos por cruza
        for _ in range(1):  # solo 2 hijos por generación
            i = girar_ruleta(ruleta)
            j = i
            while j == i:
                j = girar_ruleta(ruleta)
            
            if random.random() < 0.85:
                hijo1, hijo2 = reproducir(poblacion[i], poblacion[j])
                mutacion(hijo1)
                mutacion(hijo2)
            else:
                hijo1, hijo2 = poblacion[i], poblacion[j]
            
            nuevos_hijos.extend([hijo1, hijo2])
        
        # reeemplazo del más debil
        poblacion = reemplazar_debil(poblacion, nuevos_hijos)
        
        # encontrar el mejor de la generación actual
        mejor_generacion = 0
        for individuo in poblacion:
            fit_actual = calcular_fitness(individuo)
            if fit_actual > mejor_generacion:
                mejor_generacion = fit_actual
            if fit_actual > mejor_fitness and np.dot(individuo, pesos) <= 30:
                mejor_global = individuo.copy()
                mejor_fitness = fit_actual
        
        # fitness promedio
        fitness_promedio = sum(calcular_fitness(ind) for ind in poblacion) / len(poblacion)
        
        
        print(f"Generacion {gen+1:2d}: Mejor = {mejor_generacion:3d}, Promedio = {fitness_promedio:6.1f}, Global = {mejor_fitness:3d}")

        for idx, ind in enumerate(poblacion):
            print(f"  Individuo {idx+1:2d}: {ind} -> Fitness: {calcular_fitness(ind)}")

    print("\n")
    print("="*25)
    print("RESULTADOS")
    print("="*25)
    print("\n")

    print(f"Mejor Fitness Llegado: {mejor_fitness}")
    print(f"Peso total: {np.dot(mejor_global, pesos):.1f} kg")
    
    print("\nCOMPOSICION DE LA MEJOR SOLUCION:")
    

    for i in range(len(mejor_global)):
        if mejor_global[i] > 0:
            print(f"  {productos[i]:<20} : {mejor_global[i]:>2} unidades")
    
    print("\n")
    print("="*45)
    if mejor_fitness >= 92:
        print(" OBJETIVO ALCANZADO!!! (Fitness >= 92)")
    else:
        print(f"Objetivo no fue alcanzado. Fitness maximo posible: 92")
    print("="*45)