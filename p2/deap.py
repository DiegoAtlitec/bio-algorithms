import operator
import math
import random
import numpy
from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from deap import gp
from functools import partial

# --- 1. FUNCIÓN OBJETIVO ---
# f(x, y) = 5x^3 * y^2 + 0.5x
def target_function(x, y):
    return (5.0 * (x**3) * (y**2)) + (x / 2.0)

# --- 2. OPERACIONES PROTEGIDAS Y ESPECÍFICAS ---
def protectedDiv(left, right):
    try:
        return left / right
    except ZeroDivisionError:
        return 1.0

# Primitivas específicas que son clave (sustituyen a protectedPow)
def cube(x):
    return x**3.0
def square(x):
    return x**2.0

# --- 3. GENERACIÓN DE DATOS ---
LOW = -1.0
HIGH = 1.0
N_POINTS = 100 
points_x = numpy.random.uniform(LOW, HIGH, N_POINTS)
points_y = numpy.random.uniform(LOW, HIGH, N_POINTS)
POINTS = list(zip(points_x, points_y))
TARGET_Z = [target_function(x, y) for x, y in POINTS]

# --- 4. CONJUNTO DE PRIMITIVAS (RESTRINGIDO) ---
pset = gp.PrimitiveSet("MAIN", 2)
pset.renameArguments(ARG0='x', ARG1='y')
pset.addPrimitive(operator.add, 2)
pset.addPrimitive(operator.sub, 2)
pset.addPrimitive(operator.mul, 2)
pset.addPrimitive(protectedDiv, 2)
pset.addPrimitive(operator.neg, 1)

# Primitivas de potencia específicas
pset.addPrimitive(cube, 1)    
pset.addPrimitive(square, 1)  

# Constantes efímeras (el rango es clave)
pset.addEphemeralConstant("rand_const", partial(random.uniform, -5.0, 5.0)) 

# --- 5. CONFIGURACIÓN DEL ALGORITMO (DOBLE OBJETIVO) ---
# Se necesita que el fitness sea de dos objetivos: (Error, Tamaño)
creator.create("FitnessMin", base.Fitness, weights=(-1.0, -0.05)) 
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=4)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)

# --- 6. FUNCIÓN DE EVALUACIÓN (DOBLE OBJETIVO Y ROBUSTA) ---
def evalSymbReg(individual, points, target_z):
    func = toolbox.compile(expr=individual)
    sqerrors = []
    LARGE_PENALTY = 1e10 

    for (x, y), z_target in zip(points, target_z):
        try:
            z_pred = func(x, y) 
            if numpy.isnan(z_pred) or numpy.isinf(z_pred) or isinstance(z_pred, complex):
                sqerrors.append(LARGE_PENALTY)
                continue
            sqerrors.append((z_pred - z_target)**2)
        except Exception:
            sqerrors.append(LARGE_PENALTY)
            
    rmse = numpy.sqrt(numpy.mean(sqerrors))
    size = len(individual) 
    return rmse, size 

toolbox.register("evaluate", evalSymbReg, points=POINTS, target_z=TARGET_Z)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("mate", gp.cxOnePoint)
# Mutación estructural
toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
toolbox.register("mutate_structural", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)
# Mutación de constantes (clave para la precisión)
toolbox.register("mutate_ephemeral", gp.mutEphemeral, mode='all')

# Definir el operador de mutación como una combinación de mutación estructural y de constantes
# Probabilidad 70% mutación estructural, 30% mutación de constantes
def custom_mutation(individual):
    if random.random() < 0.7:
        individual, = toolbox.mutate_structural(individual)
    else:
        individual, = toolbox.mutate_ephemeral(individual)
    return individual,

toolbox.register("mutate", custom_mutation)

toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))

# --- 7. EJECUCIÓN PRINCIPAL ---
def main():
    random.seed(44) 

    pop = toolbox.population(n=600) 
    hof = tools.HallOfFame(1)
    
    stats_fit = tools.Statistics(lambda ind: ind.fitness.values[0])
    stats_size = tools.Statistics(len)
    mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
    mstats.register("avg", numpy.mean)
    mstats.register("min", numpy.min)
    
    # Aumentamos a 300 generaciones para asegurar la optimización de constantes
    pop, log = algorithms.eaSimple(pop, toolbox, cxpb=0.7, mutpb=0.3, ngen=125, 
                                   stats=mstats, halloffame=hof, verbose=True)

    best_individual = hof[0]
    print("\n--- Resultado Final del Algoritmo ---")
    print(f"Mejor Individuo (Árbol):\n{best_individual}")
    print(f"Fitness (RMSE, Size): {best_individual.fitness.values}")
    
    # --- SECCIÓN DE EVALUACIÓN TABULAR ---
    final_func = toolbox.compile(expr=best_individual)
    
    TEST_POINTS_N = 5
    test_x = numpy.random.uniform(LOW, HIGH, TEST_POINTS_N)
    test_y = numpy.random.uniform(LOW, HIGH, TEST_POINTS_N)
    test_data = list(zip(test_x, test_y))
    
    print("\n--- EJEMPLO DE EVALUACIÓN TABULAR ---")
    print("---------------------------------------------------------------------")
    print(f"{'X':<8}{'Y':<8}{'TARGET Z (Real)':<18}{'PREDICHO Z (Fórmula)':<18}{'ERROR ABS':<10}")
    print("---------------------------------------------------------------------")
    
    total_error = 0
    for x, y in test_data:
        target_z = target_function(x, y)
        
        try:
            pred_z = final_func(x, y)
        except:
            pred_z = float('nan')
        
        error_abs = abs(pred_z - target_z) if not numpy.isnan(pred_z) else float('inf')
        total_error += error_abs

        print(f"{x:.4f}{y:.4f}{target_z:.6f}{pred_z:.6f}{error_abs:.6f}")

    print("---------------------------------------------------------------------")
    print(f"Error Absoluto Promedio en Muestra: {total_error / TEST_POINTS_N:.6f}")
    
    return pop, log, hof

if __name__ == "__main__":
    main()