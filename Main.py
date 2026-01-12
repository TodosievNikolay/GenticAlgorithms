
"""
Genetic Algorithm for Traveling Salesman Problem (TSP)
The program:
- loads TSP files
- runs random, greedy and genetic algorithm
- prints results for comparison
"""
import random
import math

def load_tsp(filename):
    cities = []
    with open(filename, "r") as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) == 3 and parts[0].isdigit():
                cities.append((int(parts[0]), float(parts[1]), float(parts[2])))
    return cities

def distance(c1, c2):
    return math.sqrt((c1[1] - c2[1])**2 + (c1[2] - c2[2])**2)

def fitness(route, cities):
    total = 0
    for i in range(len(route)):
        a = cities[route[i] - 1]
        b = cities[route[(i + 1) % len(route)] - 1]
        total += distance(a, b)
    return total

def random_solution(cities):
    route = [c[0] for c in cities]
    random.shuffle(route)
    return route

def greedy_solution(cities, start):
    unvisited = cities[:]
    current = unvisited.pop(start)
    route = [current[0]]
    while unvisited:
        next_city = min(unvisited, key=lambda c: distance(current, c))
        unvisited.remove(next_city)
        route.append(next_city[0])
        current = next_city
    return route

def tournament_selection(population, cities, k=5):
    group = random.sample(population, k)
    return min(group, key=lambda r: fitness(r, cities))

def ordered_crossover(p1, p2):
    size = len(p1)
    a, b = sorted(random.sample(range(size), 2))
    child = [None] * size
    child[a:b] = p1[a:b]
    rest = [c for c in p2 if c not in child]
    idx = 0
    for i in range(size):
        if child[i] is None:
            child[i] = rest[idx]
            idx += 1
    return child

def inversion_mutation(route, prob=0.02):
    if random.random() < prob:
        a, b = sorted(random.sample(range(len(route)), 2))
        route[a:b] = reversed(route[a:b])
    return route

def genetic_algorithm(cities, pop_size=100, generations=500):
    population = [random_solution(cities) for _ in range(pop_size)]
    best_route = None
    best_score = float("inf")

    for _ in range(generations):
        population.sort(key=lambda r: fitness(r, cities))
        new_population = population[:5]

        while len(new_population) < pop_size:
            p1 = tournament_selection(population, cities)
            p2 = tournament_selection(population, cities)
            child = ordered_crossover(p1, p2)
            child = inversion_mutation(child)
            new_population.append(child)

        population = new_population

        score = fitness(population[0], cities)
        if score < best_score:
            best_score = score
            best_route = population[0]

    return best_route, best_score

def run_experiment(filename):
    print("\nFile:", filename)
    cities = load_tsp(filename)

    random_results = [fitness(random_solution(cities), cities) for _ in range(100)]
    print("Random best:", min(random_results))

    greedy_results = []
    for i in range(len(cities)):
        greedy_results.append(fitness(greedy_solution(cities, i), cities))
    print("Greedy best:", min(greedy_results))

    _, ga_best = genetic_algorithm(cities)
    print("GA best:", ga_best)

if __name__ == "__main__":
    files = [
        "berlin11_modified.tsp",
        "berlin52.tsp",
        "kroA100.tsp",
        "kroA150.tsp"
    ]
    for f in files:
        run_experiment(f)
