import sys
import random
import numpy as np
import matplotlib.pyplot as plt

# =========================
# PART 1 — TSP PARSER
# =========================

def load_tsp(path):
    cities = []
    with open(path, "r") as f:
        read = False
        for line in f:
            line = line.strip()
            if line == "NODE_COORD_SECTION":
                read = True
                continue
            if line == "EOF":
                break
            if read:
                i, x, y = line.split()
                cities.append((int(i), float(x), float(y)))
    return np.array(cities)


def distance_matrix(cities):
    n = len(cities)
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            dx = cities[i][1] - cities[j][1]
            dy = cities[i][2] - cities[j][2]
            dist[i, j] = np.hypot(dx, dy)
    return dist


# =========================
# PART 2 — SOLUTIONS & FITNESS
# =========================

def fitness(tour, dist):
    return sum(
        dist[tour[i], tour[(i + 1) % len(tour)]]
        for i in range(len(tour))
    )


def print_solution(tour, score):
    print("Tour:", " ".join(map(str, tour)))
    print(f"Score: {score:.2f}")


def random_tour(n):
    tour = list(range(n))
    random.shuffle(tour)
    return tour


# =========================
# PART 2 — GREEDY
# =========================

def greedy(start, dist):
    n = len(dist)
    visited = {start}
    tour = [start]

    while len(tour) < n:
        last = tour[-1]
        next_city = min(
            (i for i in range(n) if i not in visited),
            key=lambda x: dist[last][x]
        )
        visited.add(next_city)
        tour.append(next_city)

    return tour


# =========================
# PART 3 — POPULATION
# =========================

def initial_population(size, dist, greedy_seeds=0):
    pop = []
    n = len(dist)

    for i in range(greedy_seeds):
        tour = greedy(i % n, dist)
        pop.append({
            "tour": tour,
            "fitness": fitness(tour, dist)
        })

    while len(pop) < size:
        tour = random_tour(n)
        pop.append({
            "tour": tour,
            "fitness": fitness(tour, dist)
        })

    return pop


def population_info(pop):
    scores = sorted(ind["fitness"] for ind in pop)
    print(
        f"Size={len(pop)} | "
        f"Best={scores[0]:.2f} | "
        f"Median={scores[len(scores)//2]:.2f} | "
        f"Worst={scores[-1]:.2f}"
    )


# =========================
# PART 3 — SELECTION
# =========================

def tournament_selection(pop, k=3):
    return min(random.sample(pop, k), key=lambda x: x["fitness"])


# =========================
# PART 3 — CROSSOVER (OX)
# =========================

def ordered_crossover(p1, p2):
    n = len(p1)
    a, b = sorted(random.sample(range(n), 2))
    child = [-1] * n
    child[a:b] = p1[a:b]

    fill = [x for x in p2 if x not in child]
    idx = 0
    for i in range(n):
        if child[i] == -1:
            child[i] = fill[idx]
            idx += 1
    return child


# =========================
# PART 4 — MUTATION
# =========================

def inversion_mutation(tour, prob):
    if random.random() < prob:
        a, b = sorted(random.sample(range(len(tour)), 2))
        tour[a:b] = reversed(tour[a:b])
    return tour


# =========================
# PART 4 — EPOCH
# =========================

def next_generation(pop, dist, mutation_rate=0.02, elite=1):
    new_pop = sorted(pop, key=lambda x: x["fitness"])[:elite]

    while len(new_pop) < len(pop):
        p1 = tournament_selection(pop)
        p2 = tournament_selection(pop)

        child = ordered_crossover(p1["tour"], p2["tour"])
        child = inversion_mutation(child, mutation_rate)

        new_pop.append({
            "tour": child,
            "fitness": fitness(child, dist)
        })

    return new_pop


# =========================
# PART 5 — FULL RUN + PLOT
# =========================

def run_ga(tsp_file):
    cities = load_tsp(tsp_file)
    dist = distance_matrix(cities)

    # === PARSER VISUALIZATION ===
    plot_cities(cities, "Parsed cities (parser output)")

    # === GREEDY ===
    greedy_tour_best, greedy_score = greedy_analysis(cities, dist)

    # === RANDOM ===
    random_analysis(cities, dist)

    # === GA ===
    pop = initial_population(
        size=100,
        dist=dist,
        greedy_seeds=5
    )

    best_scores = []
    best_solution = None

    for epoch in range(300):
        pop = next_generation(pop, dist)
        best = min(pop, key=lambda x: x["fitness"])
        best_scores.append(best["fitness"])
        best_solution = best

        if epoch % 20 == 0:
            print(f"Epoch {epoch}", end=" → ")
            population_info(pop)

    print("\n=== BEST GA SOLUTION ===")
    print_solution(best_solution["tour"], best_solution["fitness"])

    plot_tour(cities, best_solution["tour"], "Best GA Tour")

    # === CONVERGENCE PLOT ===
    plt.figure(figsize=(7, 4))
    plt.plot(best_scores, label="Best GA fitness")
    plt.axhline(greedy_score, color="red", linestyle="--", label="Best Greedy")
    plt.xlabel("Epoch")
    plt.ylabel("Tour length")
    plt.title("GA convergence vs Greedy")
    plt.legend()
    plt.grid(True)
    plt.show()
# =========================
# ENTRY POINT
# =========================

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python tsp_ga_all_parts.py file.tsp")
        sys.exit(1)

    run_ga(sys.argv[1])
