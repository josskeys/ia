#!pip install deap
import random
import numpy as np
from deap import base, creator, tools, algorithms
import matplotlib.pyplot as plt

# Número de ciudades
NUM_CIUDADES = 15

# Coordenadas aleatorias para las ciudades
ciudades = np.random.randint(1, 30, size=(NUM_CIUDADES, 2))
print("Coordenadas de las ciudades:")
print(ciudades)
print("----------------------------------------------")

# Función de distancia entre dos ciudades
def distancia(ciudad1, ciudad2):
    return np.linalg.norm(ciudades[ciudad1] - ciudades[ciudad2])

def evaluar_ruta(individuo):
    distancia_total = 0
    for i in range(len(individuo)):
        distancia_total += distancia(individuo[i], individuo[(i+1) % NUM_CIUDADES])
    return (distancia_total,)

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individuo", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("indices", random.sample, range(NUM_CIUDADES), NUM_CIUDADES)
toolbox.register("individuo", tools.initIterate, creator.Individuo, toolbox.indices)
toolbox.register("poblacion", tools.initRepeat, list, toolbox.individuo)

toolbox.register("evaluate", evaluar_ruta)
toolbox.register("mate", tools.cxPartialyMatched)
toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)

def graficar_ruta(ciudades, ruta):
    for i in range(len(ciudades)):
        for j in range(i + 1, len(ciudades)):
            x_a, y_a = ciudades[i]
            x_b, y_b = ciudades[j]
            dist = np.linalg.norm(ciudades[i] - ciudades[j])
            plt.plot([x_a, x_b], [y_a, y_b], color='lightgray', linewidth=0.5, zorder=1)
            plt.text((x_a + x_b)/2, (y_a + y_b)/2, f"{dist:.1f}", color='gray', fontsize=6, zorder=2)

    x = [ciudades[ciudad][0] for ciudad in ruta] + [ciudades[ruta[0]][0]]
    y = [ciudades[ciudad][1] for ciudad in ruta] + [ciudades[ruta[0]][1]]
    plt.plot(x, y, 'o-', color='blue', label='Ruta óptima', zorder=3)
    plt.scatter(ciudades[:,0], ciudades[:,1], color='red', zorder=4)
    for i, (x_c, y_c) in enumerate(ciudades):
        plt.text(x_c, y_c, str(i), fontsize=12, color='black', zorder=5)

    for i in range(len(ruta)):
        ciudad_a = ruta[i]
        ciudad_b = ruta[(i+1) % len(ruta)]
        x_a, y_a = ciudades[ciudad_a]
        x_b, y_b = ciudades[ciudad_b]
        dist = np.linalg.norm(ciudades[ciudad_a] - ciudades[ciudad_b])
        plt.text((x_a + x_b)/2, (y_a + y_b)/2, f"{dist:.1f}", color='green', fontsize=12, zorder=6)
    plt.title("Mejor ruta del agente viajero")
    plt.xlabel("Coordenada X")
    plt.ylabel("Coordenada Y")
    plt.legend()
    plt.grid(True)
    plt.show()


def main():
    random.seed(42)
    poblacion = toolbox.poblacion(n=300)
    hof = tools.HallOfFame(1)
    estadisticas = tools.Statistics(lambda ind: ind.fitness.values)
    estadisticas.register("avg", np.mean)
    estadisticas.register("min", np.min)
    estadisticas.register("max", np.max)

    algorithms.eaSimple(poblacion, toolbox, cxpb=0.7, mutpb=0.2, ngen=100,
                        stats=estadisticas, halloffame=hof, verbose=True)
    print("Mejor ruta:", hof[0])
    print("Distancia mínima:", hof[0].fitness.values[0])
    graficar_ruta(ciudades, hof[0])

if __name__ == "__main__":
    main()