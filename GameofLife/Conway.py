import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
  


ON = 255
OFF = 0

detenido = False


def inicializar_random(N):

    grid = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            
            if random.random() < 0.17:
                grid[i, j] = ON
            else:
                grid[i, j] = OFF
    return grid

def inicializar_diehard(grid, r=10, c=10):
    diehard = [
        [OFF, ON,  OFF, OFF],
        [ON,  ON,  OFF, OFF],
        [OFF, ON,  ON,  ON ]
    ]

    for i in range(len(diehard)):
        for j in range(len(diehard[i])):
            if diehard[i][j] == ON:
                grid[r + i, c + j] = ON


def john(grid, r=25, c=25):
    pepe = [
        [ON, ON, OFF],
        [OFF, ON, ON],
        [OFF, ON, OFF]
    ]
    for i in range(len(pepe)):
        for j in range(len(pepe[i])):
            if pepe[i][j] == ON:
                grid[r + i, c + j] = ON


def barra(grid, r=10, c=10):
    barra = [
        [ON],
        [ON],
        [ON]
    ]

    for i in range(3):
        grid[r+i, c] = ON


def contar_vecinos(grid, N, i, j):

    dcc= 255

    total = int((grid[i, (j-1)%N] + grid[i, (j+1)%N] +
                 grid[(i-1)%N, j] + grid[(i+1)%N, j] +
                 grid[(i-1)%N, (j-1)%N] + grid[(i-1)%N, (j+1)%N] +
                 grid[(i+1)%N, (j-1)%N] + grid[(i+1)%N, (j+1)%N]) / dcc)
    return total


def actu(frameNum, img, grid, N):
    global detenido

    if detenido:
        return img

    newGrid = grid.copy()

    for i in range(N):
        for j in range(N):
            total = contar_vecinos(grid, N, i, j)

            #reglas
            if grid[i, j] == ON:
                if (total < 2) or (total > 3):
                    newGrid[i, j] = OFF
            else:
                if total == 3:
                    newGrid[i, j] = ON

    if np.array_equal(grid, newGrid):
        print(f"-> Ya convergió en la generacion {frameNum}")
        detenido = True
        return img

    img.set_data(newGrid)
    grid[:] = newGrid[:]
    return img


def main():
    
    N = 50  
    velocidad = 50 

    grid = np.zeros((N, N))

    print("Selecciona opcion inicial:")
    print("1. aleatoria: ")
    print("2. diehard: ")
    print("3. barra: ")
    print("4. john: ")


    opcion = input("Opcion: ")

    if opcion == '1':
        grid = inicializar_random(N)
    elif opcion == '2':
        inicializar_diehard(grid)
    elif opcion == '3':
        barra(grid)
    elif opcion == '4':
        john(grid)

        

    fig, ax = plt.subplots()
    img = ax.imshow(grid, interpolation='nearest', cmap='gray')
    plt.title("Juego de la Vida")

    
    ani = animation.FuncAnimation(fig, actu, fargs=(img, grid, N),
                                  frames=1000,
                                  interval=velocidad,
                                  save_count=50)
    plt.show()

if __name__ == '__main__':
    main()