#Gonzalez Hernandez Diego 2024630297
#5BM1

import numpy as np
import math


def funcion_objetivo(particula):
    x = particula[0]
    y = particula[1]
    return x**2 + y**2 + 25 * (np.sin(x) + np.sin(y))

# parametros 
N_PARTICULAS = 20
ITERACIONES = 50
W = 0.8  
C1 = 0.7 
C2 = 1.2 

# parametros del problema
DIMENSIONES = 2  # (x, y)
LIMITE_MIN = -5.0
LIMITE_MAX = 5.0

#inicializacion del enjambre ---
posiciones = (LIMITE_MAX - LIMITE_MIN) * np.random.rand(N_PARTICULAS, DIMENSIONES) + LIMITE_MIN

velocidades = np.random.uniform(-1, 1, (N_PARTICULAS, DIMENSIONES))

pbest_pos = np.copy(posiciones)
pbest_val = np.array([funcion_objetivo(p) for p in pbest_pos])

gbest_idx = np.argmin(pbest_val)
gbest_pos = np.copy(pbest_pos[gbest_idx])
gbest_val = pbest_val[gbest_idx]

print("INICIALIZACION")
print(f"Mejor valor global inicial (gbest_val): {gbest_val}")
print(f"Mejor posicion global inicial (gbest_pos): {gbest_pos}\n")
print("\n")

#Bucle del PSO
for t in range(ITERACIONES):
    print(f"ITERACION {t + 1}/{ITERACIONES}")

    for i in range(N_PARTICULAS):
        
        r1 = np.random.rand(DIMENSIONES)
        r2 = np.random.rand(DIMENSIONES)

        velocidad_inercial = W * velocidades[i]
        componente_cognitivo = C1 * r1 * (pbest_pos[i] - posiciones[i])
        componente_social = C2 * r2 * (gbest_pos - posiciones[i])
        
        velocidades[i] = velocidad_inercial + componente_cognitivo + componente_social

        posiciones[i] = posiciones[i] + velocidades[i]

        posiciones[i] = np.clip(posiciones[i], LIMITE_MIN, LIMITE_MAX)

        fitness_actual = funcion_objetivo(posiciones[i])

        if fitness_actual < pbest_val[i]:
            pbest_val[i] = fitness_actual
            pbest_pos[i] = np.copy(posiciones[i])


            if fitness_actual < gbest_val:
                gbest_val = fitness_actual
                gbest_pos = np.copy(posiciones[i])

    gbest_val = nuevo_gbest_val
    gbest_pos = np.copy(nuevo_gbest_pos)
    
    print("\n[Posiciones de las particulas (x, y)]")
    print(posiciones)
    
    print("\n[Velocidades de las particulas (vx, vy)]")
    print(velocidades)
    
    print("\n[mejores posiciones personales (pbest)]")
    print(pbest_pos)
    
    print("\n[mejor posicion global (gbest)]")
    print(f"  posicion: {gbest_pos}")
    print(f"  valor (fitness): {gbest_val}")
    print("\n")


print("\n")
print("OPTIMIZACION FINALIZADA")
print(f"Iteraciones totales: {ITERACIONES}")
print(f"Numero de particulas: {N_PARTICULAS}")
print("\nMejor posicion global encontrada (gbest):")
print(f"  Posicion (x, y): {gbest_pos}")
print(f"  Valor minimo (Fitness): {gbest_val}")
