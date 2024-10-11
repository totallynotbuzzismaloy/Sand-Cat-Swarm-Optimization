import numpy as np
import time
import pandas as pd
import matplotlib.pyplot as plt


class SearchAgent:
    def __init__(self, dimension, lower_bound, upper_bound):
        self.position = np.random.uniform(lower_bound, upper_bound, dimension)
        self.fitness = None


def fitness_function(x):
    # return 0.5 * np.sum(x**4 - 16 * x**2 + 5 * x) # Стыбинского-Танга
    term1 = np.sum(x ** 2) / 400
    term2 = np.prod(np.cos(x / np.sqrt(np.arange(1, len(x) + 1))))
    return term1 - term2 + 1


def initialize_population(pop_size, dimension, lower_bound, upper_bound):
    population = [SearchAgent(dimension, lower_bound, upper_bound) for _ in range(pop_size)]
    return population


def calculate_r(iter_c, iter_max, sM):
    rG = sM - (sM * iter_c / iter_max)
    return rG * np.random.rand()


def enforce_bounds(position, lower_bound, upper_bound):
    return np.clip(position, lower_bound, upper_bound)


def update_position(agent, best_position, current_position, r, theta, Xrnd, R, lower_bound, upper_bound):
    if abs(R) > 1:
        new_position = r * (best_position - np.random.rand() * current_position)
    else:
        new_position = best_position - r * Xrnd * np.cos(theta)

    agent.position = enforce_bounds(new_position, lower_bound, upper_bound)


def sand_cat_swarm_optimization(pop_size, dimension, iter_max, lower_bound, upper_bound, sM):
    population = initialize_population(pop_size, dimension, lower_bound, upper_bound)
    best_position = min(population, key=lambda x: fitness_function(x.position)).position
    iter_c = 0

    while iter_c <= iter_max:
        r = calculate_r(iter_c, iter_max, sM)
        for agent in population:
            random_angle = np.random.uniform(0, 360)
            theta = np.deg2rad(random_angle)
            Xrnd = np.abs(np.random.rand() * best_position - agent.position)
            R = 2 * r * np.random.rand() - r

            update_position(agent, best_position, agent.position, r, theta, Xrnd, R, lower_bound, upper_bound)
            agent.fitness = fitness_function(agent.position)

            if agent.fitness < fitness_function(best_position):
                best_position = agent.position

        iter_c += 1

    return best_position


def run_experiments(algorithm, pop_size_range, dimension_range, iter_max_range, sM_range):
    results = []

    for pop_size in pop_size_range:
        for dimension in dimension_range:
            for iter_max in iter_max_range:
                start_time = time.time()
                best_solution = algorithm(pop_size, dimension, iter_max, lower_bound, upper_bound, sM_range)
                elapsed_time = time.time() - start_time
                number_of_calls = pop_size * iter_max

                result = {
                    'pop_size': pop_size,
                    'dimension': dimension,
                    'iter_max': iter_max,
                    'number_of_calls': number_of_calls,
                    'elapsed_time': elapsed_time,
                    'best_solution': best_solution,
                    'best_fitness': fitness_function(best_solution),
                }
                results.append(result)

    return results


pop_size_range = [5]  # Диапазон значений для размера популяции
dimension_range = [2, 5, 10, 15, 20]  # Диапазон значений для размерности задачи
iter_max_range = [750]  # Диапазон значений для максимального числа итераций
sM_range = 2  # sM

# Parameters
lower_bound = -20  # Lower bound of search space
upper_bound = 20  # Upper bound of search space

# Запускаем серию экспериментов
experiment_results = run_experiments(sand_cat_swarm_optimization, pop_size_range, dimension_range, iter_max_range,
                                     sM_range)

# Преобразуем результаты в DataFrame для удобства анализа
df_results = pd.DataFrame(experiment_results)

# Отобразим результаты в виде графиков и таблицэ# Выводим таблицу с результатами
print(df_results)

# Например, можно построить графики зависимости времени выполнения, лучшей фитнесс-функции и т.д. от параметров алгоритма и размерности задачи
# Создаем график зависимости времени выполнения от параметров алгоритма и размерности задачи
plt.figure(figsize=(10, 6))
for pop_size in pop_size_range:
    for iter_max in iter_max_range:
        df_subset = df_results[(df_results['pop_size'] == pop_size) & (df_results['iter_max'] == iter_max)]
        plt.plot(df_subset['dimension'], df_subset['elapsed_time'], label=f'pop_size={pop_size}, iter_max={iter_max}')
plt.xlabel('Dimension')
plt.ylabel('Elapsed Time (s)')
plt.title('Dependency of Elapsed Time on Algorithm Parameters and Problem Dimension')
plt.legend()
plt.grid(True)
plt.show()


# Создаем график зависимости лучшей фитнесс-функции от параметров алгоритма и размерности задачи
plt.figure(figsize=(10, 6))
for pop_size in pop_size_range:
    for iter_max in iter_max_range:
        df_subset = df_results[(df_results['pop_size'] == pop_size) & (df_results['iter_max'] == iter_max)]
        plt.plot(df_subset['dimension'], df_subset['best_fitness'], label=f'pop_size={pop_size}, iter_max={iter_max}')
plt.xlabel('Dimension')
plt.ylabel('Best Fitness')
plt.title('Dependency of Best Fitness on Algorithm Parameters and Problem Dimension')
plt.legend()
plt.grid(True)
plt.show()
