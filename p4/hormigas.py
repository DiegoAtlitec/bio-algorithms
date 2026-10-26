#Gonzalez Hernandez Diego (2024630297)
#Tamayo Escalante Eduardo Osmar

import random

matriz_distancias = [
    [0, 6, 9, 17, 13, 21],
    [6, 0, 19, 21, 12, 18],
    [9, 19, 0, 20, 23, 11],
    [17, 21, 20, 0, 15, 10],
    [13, 12, 23, 15, 0, 21],
    [21, 18, 11, 10, 21, 0]
]

tasa_evaporacion = 0.2
constante_feromona = 1
alpha = 1.5
beta = 0.8
num_ciudades = len(matriz_distancias)
iteraciones_maximas = 50

def calcularMatrizReciproca(matriz, constante):
    matriz_reciproca = []
    for fila in matriz:
        nueva_fila = []
        for valor in fila:
            if valor == 0:
                nueva_fila.append(0)
            else:
                nueva_fila.append(constante / valor)
        matriz_reciproca.append(nueva_fila)
    return matriz_reciproca

def reciprocos(lista_valores, constante):
    return [constante/valor if valor != 0 else 0 for valor in lista_valores]

def calcular_suma_visibilidad(feromona_mat, visibilidad_mat, alpha, beta, num_nodos):
    sumatorias = [0] * num_nodos
    for i in range(num_nodos):
        for j in range(num_nodos):
            sumatorias[i] += (feromona_mat[i][j] ** alpha) * (visibilidad_mat[i][j] ** beta)
    return sumatorias

def evaluar_rutas(rutas_visitadas, matriz_distancias):
    evaluaciones = []
    for ruta in rutas_visitadas:
        distancia_total = 0
        for i in range(len(ruta) - 1):
            inicio = ruta[i]
            fin = ruta[i + 1]
            distancia_total += matriz_distancias[inicio][fin]
        evaluaciones.append(distancia_total)
    return evaluaciones

def actualizar_matriz_feromonas(feromona_mat, rutas_visitadas, evaluaciones, alpha, beta, tasa_evaporacion):
    num_nodos = len(feromona_mat)
    
    for i in range(num_nodos):
        for j in range(num_nodos):
            feromona_mat[i][j] = (1 - tasa_evaporacion) * feromona_mat[i][j]
    
    for idx, ruta in enumerate(rutas_visitadas):
        cantidad_feromona = evaluaciones[idx]
        
        for i in range(len(ruta) - 1):
            nodo_actual = ruta[i]
            nodo_siguiente = ruta[i + 1]
            
            feromona_mat[nodo_actual][nodo_siguiente] += cantidad_feromona
            feromona_mat[nodo_siguiente][nodo_actual] += cantidad_feromona
    
    return feromona_mat



def ruleta(probsDisponibles, nodosVisitados):
    nodosCandidatos = []
    probabilidades = []
    
    suma_probas = 0.0
    for nodo, prob in enumerate(probsDisponibles):
        if nodo not in nodosVisitados:
            nodosCandidatos.append(nodo)
            probabilidades.append(prob)
            suma_probas += prob
    
    if suma_probas == 0:
        return None

    probabilidadesNormalizadas = [p / suma_probas for p in probabilidades]

    probabilidades_acumuladas = []
    acumulador = 0.0
    for p in probabilidadesNormalizadas:
        acumulador += p
        probabilidades_acumuladas.append(acumulador)

    valor_aleatorio = random.uniform(0, probabilidades_acumuladas[-1])

    indice = 0
    while indice < len(probabilidades_acumuladas) and probabilidades_acumuladas[indice] < valor_aleatorio:
        indice += 1

    return nodosCandidatos[indice]


def construir_rutas(suma_visibilidad, num_nodos, feromona_mat, visibilidad_mat, alpha, beta):
    matriz_probabilidades = [[0.0] * num_nodos for _ in range(num_nodos)]
    
    for i in range(num_nodos):
        if suma_visibilidad[i] == 0:
            continue
        for j in range(num_nodos):
            matriz_probabilidades[i][j] = ((feromona_mat[i][j] ** alpha) * 
                                         (visibilidad_mat[i][j] ** beta) / 
                                         suma_visibilidad[i])
    
    rutas_completas = [[nodo_inicio] for nodo_inicio in range(num_nodos)]
    
    for idx_hormiga in range(num_nodos):
        nodo_actual = idx_hormiga
        nodosVisitados = [nodo_actual]
        
        while len(nodosVisitados) < num_nodos:
            probsDisponibles = [0.0] * num_nodos
            suma_probas = 0.0
                    
            for nodo_candidato in range(num_nodos):
                if nodo_candidato not in nodosVisitados:
                    probsDisponibles[nodo_candidato] = matriz_probabilidades[nodo_actual][nodo_candidato]
                    suma_probas += probsDisponibles[nodo_candidato]

            if suma_probas == 0:
                break

            siguiente_nodo = ruleta(probsDisponibles, nodosVisitados)

            rutas_completas[idx_hormiga].append(siguiente_nodo)
            nodosVisitados.append(siguiente_nodo)
            nodo_actual = siguiente_nodo

        
        if len(rutas_completas[idx_hormiga]) == num_nodos:
            rutas_completas[idx_hormiga].append(idx_hormiga)
    
    return rutas_completas

def encontrar_mejor_solucion(lista_evaluaciones):
    mejor_indice = 0
    mejor_valor = lista_evaluaciones[0]
    
    for i in range(1, len(lista_evaluaciones)):
        if lista_evaluaciones[i] < mejor_valor:
            mejor_indice = i
            mejor_valor = lista_evaluaciones[i]
    
    return mejor_indice, mejor_valor

if __name__ == '__main__':
    print("\nGonzalez Hernandez Diego (2024630297)\nTamayo Escalante Eduardo Osmar")
    print("\nPractica: ALGORITMO DE COLONIA DE HORMIGAS\n")



    print(f"Parametros: ρ={tasa_evaporacion}, Q={constante_feromona}, α={alpha}, β={beta}")
    print(f"Numero de ciudades: {num_ciudades}")
    print(f"Iteraciones: {iteraciones_maximas}")
    print()
    
    matriz_feromonas = [[0.1 for _ in range(num_ciudades)] for _ in range(num_ciudades)]
    matriz_visibilidad = calcularMatrizReciproca(matriz_distancias, constante_feromona)

    mejor_ruta_global = None
    mejor_fit_global = float('inf')
    
    todas_rutas = []
    todos_fits = []
    
    for iteracion in range(iteraciones_maximas):
        sumas_visibilidad = calcular_suma_visibilidad(
            matriz_feromonas, matriz_visibilidad, alpha, beta, num_ciudades
        )
        
        rutas_hormigas = construir_rutas(
            sumas_visibilidad, num_ciudades, matriz_feromonas, 
            matriz_visibilidad, alpha, beta
        )
        
        evaluaciones_rutas = evaluar_rutas(rutas_hormigas, matriz_distancias)
        
        indice_mejor, fit_mejor = encontrar_mejor_solucion(evaluaciones_rutas)
        ruta_mejor_iteracion = rutas_hormigas[indice_mejor]
        
        todas_rutas.append(ruta_mejor_iteracion)
        todos_fits.append(fit_mejor)
        
        print(f"Iteracion {iteracion + 1}: Mejor ruta = {ruta_mejor_iteracion}, Fit = {fit_mejor}")
        
        if fit_mejor < mejor_fit_global:
            mejor_fit_global = fit_mejor
            mejor_ruta_global = ruta_mejor_iteracion.copy()

        evaluaciones_reciprocas = reciprocos(evaluaciones_rutas, constante_feromona)
        
        matriz_feromonas = actualizar_matriz_feromonas(
            matriz_feromonas, rutas_hormigas, evaluaciones_reciprocas, 
            alpha, beta, tasa_evaporacion
        )
    
    print("\n- RESULTADO FINAL:")
    print(f"La mejor ruta encontrada es: {mejor_ruta_global}")
    print(f"Fit total: {mejor_fit_global}")
    
    