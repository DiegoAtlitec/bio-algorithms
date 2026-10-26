#Gonzalez Hernandez Diego           2024630297
#Tamayo Escalante Eduardo Osmar     2024630589

import pandas as pd
import numpy as np
import random
import time
import matplotlib.pyplot as plt
import seaborn as sns
from copy import deepcopy

from sklearn.datasets import load_breast_cancer, fetch_openml
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

class SelectorGenetico:
    def __init__(self, datos, target, n_pop=30, n_gen=20, alpha=0.75):
        self.df = datos
        self.y = target
        self.n_features = datos.shape[1]
        self.tam_pob = n_pop
        self.generaciones = n_gen
        self.alpha = alpha 
        
        # historial
        self.hist_fit = []
        self.hist_prom = []
        self.frecuencias = np.zeros(self.n_features)
        
        self.clf = RandomForestClassifier(n_estimators=10, n_jobs=-1, random_state=42)

    def crear_poblacion(self):
        pob = []
        for _ in range(self.tam_pob):
            cromosoma = [random.randint(0, 1) for _ in range(self.n_features)]
            if sum(cromosoma) == 0:
                cromosoma[random.randint(0, self.n_features-1)] = 1
            pob.append(cromosoma)
        return pob

    def fitness(self, cromosoma):
        cols = [i for i, x in enumerate(cromosoma) if x == 1]
        if len(cols) == 0: return 0.0
        
        X_sub = self.df.iloc[:, cols]
        
        # validacion cruzada
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        #scores = cross_val_score(self.clf, X_sub, self.y, cv=cv, scoring='accuracy')
        #scores = cross_val_score(self.clf, X_sub, self.y, cv=cv, scoring='f1_macro')
        scores = cross_val_score(self.clf, X_sub, self.y, cv=cv, scoring='roc_auc')
        acc = scores.mean()

        reduccion = 1 - (len(cols) / self.n_features)
        return (self.alpha * acc) + ((1 - self.alpha) * reduccion)

    def ruletaF(self, lista_fitness):
        #mi tan amada ruleta
        suma_total = sum(lista_fitness)
        
        if suma_total == 0:
            return random.randint(0, len(lista_fitness)-1)
        
        valor_aleatorio = random.uniform(0, suma_total)
        
        acumulado = 0.0
        for i, fit_val in enumerate(lista_fitness):
            acumulado += fit_val
            if acumulado >= valor_aleatorio:
                return i
        
        return len(lista_fitness) - 1

    def mutar(self, individuo, gen_actual):
        tasa = 0.3 - (0.25 * (gen_actual / self.generaciones))
        mutado = deepcopy(individuo)
        
        for i in range(len(mutado)):
            if random.random() < tasa:
                mutado[i] = 1 - mutado[i]
        
        if sum(mutado) == 0:
            mutado[random.randint(0, self.n_features-1)] = 1  
        return mutado

    def ejecutar(self):
        print(f"\n--> procesando {self.n_features} columnas.")

        poblacion = self.crear_poblacion()
        
        mejor_ind = None
        mejor_fit = 0.0

        for gen in range(self.generaciones):
            fits = []
            
            # evaluar
            for ind in poblacion:
                f = self.fitness(ind)
                fits.append(f)
                self.frecuencias += np.array(ind)
            
            max_f = max(fits)
            prom_f = sum(fits) / len(fits)
            self.hist_fit.append(max_f)
            self.hist_prom.append(prom_f)
            
            # guardar mejor global
            if max_f > mejor_fit:
                mejor_fit = max_f
                mejor_ind = poblacion[fits.index(max_f)]
            
            print(f"Gen {gen+1}: Mejor -> {max_f:.4f} | Promedio -> {prom_f:.4f}")

            # elitismo (guardamos al mejor sin pasarlo por ruleta)
            nueva_pob = [mejor_ind]
            
            # rellenar con ruleta
            while len(nueva_pob) < self.tam_pob:
                idx_p1 = self.ruletaF(fits)
                idx_p2 = self.ruletaF(fits)
                
                padre1 = poblacion[idx_p1]
                padre2 = poblacion[idx_p2]
                
                # cruza
                punto = random.randint(1, self.n_features-1)
                hijo = padre1[:punto] + padre2[punto:]
                
                # mutacion y agregar
                nueva_pob.append(self.mutar(hijo, gen))
            
            poblacion = nueva_pob

        return mejor_ind, mejor_fit

    def graficar(self):
        plt.figure(figsize=(14, 6))
        
        plt.subplot(1, 2, 1)
        plt.plot(self.hist_fit, label='Mejor', color='green')
        plt.plot(self.hist_prom, label='Promedio', color='blue', linestyle='--')
        plt.title('convergencia')
        plt.xlabel('gen')
        plt.ylabel('titness')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        norm = self.frecuencias / (self.generaciones * self.tam_pob)
        sns.barplot(x=np.arange(self.n_features), y=norm, color='purple')
        plt.title('Frecuencia de uso')
        plt.xlabel('Columna')
        
        plt.tight_layout()
        plt.show()

def cargar_datos():
    print("\n----- MENU DATASETS -----")
    print("1. Breast Cancer (30 cols)")
    print("2. Parkinsons (22 cols)")
    print("3. German Credit (24 cols)")
    print("4. Dermatology (33 cols)")
    
    
    op = input("\nIngresa tu opcion: ")
    X, y = None, None

    try:
        if op == '1':
            d = load_breast_cancer()
            X = pd.DataFrame(d.data, columns=d.feature_names)
            y = pd.Series(d.target)
        elif op == '2':
            print("Bajando Parkinsons...")
            d = fetch_openml(name='parkinsons', version=1, as_frame=True, parser='auto')
            X, y = d.data, d.target
        elif op == '3':
            print("Bajando German Credit...")
            d = fetch_openml(name='german_credit', version=1, as_frame=True, parser='auto')
            X, y = d.data, d.target
            for col in X.select_dtypes(include=['category', 'object']).columns:
                X[col] = X[col].astype('category').cat.codes
        elif op == '4':
            print("Bajando Dermatology...")
            d = fetch_openml(name='dermatology', version=1, as_frame=True, parser='auto')
            X, y = d.data, d.target

        else:
            print("Opcion invalida. Usando Breast Cancer.")
            d = load_breast_cancer()
            X = pd.DataFrame(d.data, columns=d.feature_names)
            y = pd.Series(d.target)

        if X is not None:
            if y.dtype == 'object' or y.dtype.name == 'category':
                le = LabelEncoder()
                y = pd.Series(le.fit_transform(y))
            X = X.dropna()
            y = y[X.index]
            return X, y

    except Exception as e:
        print(f"Error: {e}")
        return None, None

if __name__ == "__main__":
    df, y = cargar_datos()
    
    if df is not None:
        ag = SelectorGenetico(df, y, n_pop=30, n_gen=20, alpha=0.75)

        inicio = time.time()
        mejor_sol, mejor_score = ag.ejecutar()
        fin = time.time()
        print(f"\ntiempo: {fin - inicio:.2f} s")
        
        #print("\nGenerando graficas...")
        ag.graficar()
        
        indices = [i for i, x in enumerate(mejor_sol) if x == 1]
        cols_sel = df.columns[indices]
        
        print("\nRESULTADOS FINALES")
        print(f"Fitness: {mejor_score:.3f}")
        print(f"La reduccion fue a: {len(indices)} de {df.shape[1]} columnas")
        
        print("\nColumnas seleccionadas:")
        print("INDICE | NOMBRE")
        print("-----" * 7)
        for i, nom in zip(indices, cols_sel):
            print(f"{i:<6} | {nom}")