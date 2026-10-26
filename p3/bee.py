# Gonzalez Hernandez Diego (2024630297)
# Tamayo Escalante Eduardo Osmar

import random
import numpy as np

nombres = ["decoy_detonators" ,"love_potion", "extendable_ears" ,"skiving_snackbox" ,"fever_fudge", "puking_pastilles","nosebleed_nougat"]
pesos = np.array([4,2,5,5,2,1.5,1])
precios = np.array([10,8,12,6,3,2,2])

# Limites
minimos = [0,3,0,2,0,0,0]
maximo = 10              
capacidad = 30

# Parametros
iteraciones = 50
n_obreras = 20
n_observ = 20
limite = 6
dim = len(pesos)

def peso_total(sol):
    return np.sum(sol * pesos)

def fitness(sol):
    p = peso_total(sol)
    # Penalización si se pasa de peso o viola minimos
    if p > capacidad or sol[1] < 3 or sol[3] < 2:
        return 1
    return np.sum(sol * precios)



def mochilaValida():
    s = np.zeros(dim)

    s[1] = 3
    s[3] = 2

    peso_act = peso_total(s)
    resto = capacidad - peso_act
    orden = list(range(dim))
    random.shuffle(orden)

    for i in orden:
        if resto <= 0: break
        
        max_por_peso = int(resto // pesos[i])
        max_disp = maximo - s[i]

        limite_real = min(max_por_peso, max_disp)
        if limite_real > 0:
            cant = random.randint(0, limite_real)
            s[i] += cant
            resto -= cant * pesos[i]
    return s

def redondo(num):
    return int(num)

def generar_explorador():
    
    s = np.zeros(dim)

    for j in range(dim):
        l_j = minimos[j] #limite inferior 
        u_j = maximo     #limite superior
        r1 = random.uniform(0,1)
        
        val = l_j + r1 * (u_j - l_j)
        
        #redondeo de prueba
        s[j] = redondo(val)
        #s[j] = int(round(val))
        
    return s

def ruleta(fits):
    total = sum(fits)
    if total == 0:
        return random.randint(0, len(fits)-1)
    r = random.uniform(0,total)
    acum = 0
    for i,f in enumerate(fits):
        acum += f
        if acum >= r:
            return i
    return len(fits)-1

class Colonia:
    def __init__(self):
        self.sol = []
        self.fit = []
        self.cont = []
        self.mejor = None
        self.mejor_fit = 0

    def inicio(self):
        for _ in range(n_obreras):
            
            s = mochilaValida()
            f = fitness(s)
            self.sol.append(s)

            self.fit.append(f)
            self.cont.append(0)
            if f > self.mejor_fit:
                self.mejor_fit = f
                self.mejor = s.copy()

    def mutar(self, i):
        k = random.randint(0, n_obreras-1)
        while k == i:
            k = random.randint(0, n_obreras-1)

        j = random.randint(0, dim-1)
        r2 = random.uniform(-1,1)

        nueva = self.sol[i].copy()
        

        nueva[j] = int(round(self.sol[i][j] + r2*(self.sol[i][j] - self.sol[k][j])))

        
        if nueva[j] < minimos[j]: nueva[j] = minimos[j]
        if nueva[j] > maximo: nueva[j] = maximo

        return nueva

    def correr(self):
        self.inicio()

        for it in range(iteraciones):

            # obreras
            for i in range(n_obreras):
                nsol = self.mutar(i)
                f2 = fitness(nsol)

                if f2 > self.fit[i]:
                    self.sol[i] = nsol
                    self.fit[i] = f2
                    self.cont[i] = 0
                else:
                    self.cont[i] += 1

            #observadoras
            for _ in range(n_observ):
                i = ruleta(self.fit)
                nsol = self.mutar(i)
                f2 = fitness(nsol)

                if f2 > self.fit[i]:
                    self.sol[i] = nsol
                    self.fit[i] = f2
                    self.cont[i] = 0
                else:
                    self.cont[i] += 1

            #exploradoras 
            for i in range(n_obreras):
                if self.cont[i] >= limite:
                    
                    self.sol[i] = generar_explorador() 
                    self.fit[i] = fitness(self.sol[i])
                    self.cont[i] = 0 # Reiniciamos contador

            
            idx = np.argmax(self.fit)
            if self.fit[idx] > self.mejor_fit:
                self.mejor_fit = self.fit[idx]
                self.mejor = self.sol[idx].copy()

            print("Iter", it+1, "Mejor fitness: ", int(self.mejor_fit))

        return self.mejor, self.mejor_fit


if __name__ == "__main__":
    colonia = Colonia()
    sol, fit = colonia.correr()

    print("\nla mejor abeja fue: ", sol.astype(int).tolist(), "con un fitness de: ", int(fit), "y un peso:", peso_total(sol))