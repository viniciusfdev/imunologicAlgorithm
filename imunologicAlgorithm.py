# python imunologic algortihm  

### Algoritmos Imunológicos  
##### Autor: Vinícius França Lima Vilaça  

# a) Max_it=50  
# b) n1=N=50  
# c) n2=0  
# d) beta=0,1  
# e) Nc=beta*N – Define o número de clones a ser gerado para cada anticorpo  
# f) alpha - parâmetro da equação de mutação = 5  



import matplotlib.pyplot as plt
import numpy as np
import random
import math
import time

class Individual:
    def __init__(self, x, y, fitness = None):
        self.x = x
        self.y = y
        self.fitness = fitness

    def __repr__(self):
        return "x: {}, y: {}, z: {}\n".format(self.x, self.y, self.fitness)

class Population:
    MIN = -10
    MAX = 10
    GLOBAL_MIN = -106.764537
    def __init__(self, max_it, pop_size, n1 = 50, n2 = 0, b = 0.1, ro = 5):
        self.b = b
        self.n1 = n1
        self.n2 = n2
        self.ro = ro
        self.reset_path()
        self.max_it = max_it
        self.pop_size = pop_size
        
    def reset_path(self):
        self.path = { "best": [], "average": [], "worst": []}

    def create_rand_pop(self):
        return [Individual(
            random.uniform(self.MIN, self.MAX),
            random.uniform(self.MIN, self.MAX)) for i in range(self.pop_size)]

    def assess_fitness(self, population):
        """
            add fitness attribute to all of the items
        """
        for i, idv in enumerate(population):
            population[i].fitness = self.fitness(idv.x, idv.y)

        return population
    
    def mutate(self, idv, best) -> Individual:
        """
            Mutar de maneira inversamente proporcional ao fitness do idv
        """
        tx_mut = math.exp(-self.ro * abs(idv.fitness / best.fitness))
        
        return Individual(
            idv.x + random.uniform(-5, 5) if random.uniform(0, 1) < tx_mut else idv.x,
            idv.y + random.uniform(-5, 5) if random.uniform(0, 1) < tx_mut else idv.y)

    def clone_by_fitness(self, population) -> list:
      """
          Generates clones according to fitness
      """
      return [[idv for i in range(round(self.n1 * self.b))] for idv in population]

    def select_by_fitness(self, population):
        """
            Seleção por elitismo, a seleção randomica é feita sob
            n1 dos melhores (nao é utilizada na pratica, foi feita a 
            titulo de curiosidade e testes)
        """
        return self.sort_by_fitness(population)[:self.n1]
    
    def sort_by_fitness(self, population) -> list:
        """
            Ordenação pelo melhor fitness
        """
        return sorted(population, key=lambda idv: idv.fitness)

    @staticmethod
    def fitness(x, y) -> float:
        """
            Função fitness para avaliar o individuo
        """
        return (math.sin(x) * math.exp((1-math.cos(y))**2) + 
                math.cos(y) * math.exp((1-math.sin(x))**2) + 
                (x-y)**2)
      
    def get_best(self, population) -> Individual:
        """
            Retorna o melhor individuo da população
        """
        return self.sort_by_fitness(population)[0]

    def get_average(self, population):
        """
            Recupera o fitness medio da população corrente
        """
        return sum([idv.fitness for idv in population]) / len(population)

    def get_worst(self, population):
        return self.sort_by_fitness(population)[len(population)-1]

    def save_all(self, population):
        """
            Salva todos os dados da população atual
        """
        self.path["best"].append(self.get_best(population).fitness)
        self.path["average"].append(self.get_average(population))
        self.path["worst"].append(self.get_worst(population).fitness)

    def run(self) -> int:
        
        # create first random population
        pop = self.create_rand_pop()
        for i in range(self.max_it):

            # eval -> determine the fitness
            pop = self.sort_by_fitness(self.assess_fitness(pop))
            
            # save all the data to register path
            self.save_all(pop)
            
            # select the bests
            ff = self.select_by_fitness(pop)

            # select the n1 highest idv and clone
            mult_clones = self.clone_by_fitness(ff)
            
            clones = []
            best = self.get_best(pop)
            for sub_clones in mult_clones:
                m_sub_clones = [self.mutate(c, best) for c in sub_clones]
                m_sub_clones = self.assess_fitness(m_sub_clones)
                clones.append(self.get_best(m_sub_clones))

            pop = pop[:-self.n1] + clones
        
        return pop
      

if __name__ == "__main__":
    pop = Population(50, 100, n1 = 50, ro=2)
    init_time = time.time()
    final_result = pop.run()

    print("População final:\n {}".format(final_result))
    print("Execution time: {}".format(time.time() - init_time))

    gen_dist = range(len(pop.path["best"]))
    best_line = np.linspace(0, len(pop.path["best"]))

    # second poly fit to plot the aproximation
    best_model = np.poly1d(np.polyfit(gen_dist, pop.path["best"], 3))
    worst_model = np.poly1d(np.polyfit(gen_dist, pop.path["worst"], 3))
    average_model = np.poly1d(np.polyfit(gen_dist, pop.path["average"], 3))
    
    plt.plot(best_line, best_model(best_line), 'g', label='Best')
    plt.plot(best_line, worst_model(best_line), 'r', label='Worst')
    plt.plot(best_line, average_model(best_line), 'b', label='Average')

    plt.legend(['Best', 'Worst', 'Average'])

    plt.xlabel('generations')
    plt.ylabel('fitness')
    plt.show()
