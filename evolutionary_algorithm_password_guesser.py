import numpy as np
import matplotlib.pyplot as plt
import sys
import matplotlib
import string

plt.style.use('seaborn')

libraries = (('Matplotlib', matplotlib), ('Numpy', np))
print('Python Version:', sys.version, '\n')
for lib in libraries:
    print('{0} Version: {1}'.format(lib[0], lib[1].__version__))

allowed_characters = list(string.printable[:62]) + [' ']
print(allowed_characters)

np.random.seed(42)
def create_guess(length):
    guess = [np.random.choice(allowed_characters) for _ in range(length)]
    return ''.join(guess)
create_guess(10)

def create_generation(population = 100, length = 10):
    generation = [create_guess(length) for _ in range(population)]
    return generation
test_generation = create_generation(population = 10, length = 10)
print(test_generation)

def fitness_score(guess, actual):
    score = 0
    for i, j in zip(guess, actual):
        if i == j:
            score += 1
    return score

def check_fitness(guesses, actual):
    fitness_indicator = []
    for guess in guesses:
        fitness_indicator.append((guess, fitness_score(guess, actual)))
    return fitness_indicator
print(check_fitness(test_generation, 'imbobandhi'))

def get_breeders_from_generation(guesses, actual, take_best_N = 10, take_random_N = 5, verbose = False, mutation_rate = 0.1):
    fit_scores = check_fitness(guesses, actual)
    sorted_guesses = sorted(fit_scores, key = lambda x: x[1], reverse = True)
    new_generation = [x[0] for x in sorted_guesses[:take_best_N]]
    best_guess = new_generation[0]
    if verbose:
        print(best_guess)
    for _ in range(take_random_N):
        new_generation.append(np.random.choice(guesses))
    guess_indexes = list(range(len(actual)))
    for guess in new_generation:
        if np.random.uniform() < mutation_rate:
            gs = list(guess)
            gs[np.random.choice(guess_indexes)] = np.random.choice(allowed_characters)
            guess = ''.join(gs)
    np.random.shuffle(new_generation)
    return new_generation, best_guess

def make_child(parent1, parent2):
    child = list(parent1)
    parent2 = list(parent2)
    for ix, gene in enumerate(child):
        if np.random.uniform() >= 0.5:
            child[ix] = parent2[ix]
    return ''.join(child)

def make_children(old_generation, children_per_couple = 1):
    mid_point = len(old_generation) // 2
    next_generation = []
    for ix, parent in enumerate(old_generation[:mid_point]):
        for _ in range(children_per_couple):
            next_generation.append(make_child(parent, old_generation[-ix - 1]))
    return next_generation

make_child('steveisfish', 'lauranobird')

breeders, _ = get_breeders_from_generation(test_generation, 'imbobandhi')
print(breeders)

print(make_children(breeders, children_per_couple = 2))

current_generation = create_generation(population = 500, length = 10)
actual_password = 'passwordhi'
print_every_n_generations = 5
for i in range(1000):
    if not i % print_every_n_generations:
        print("Generation %i: "%i, end = '')
        print(len(current_generation))
        is_verbose = True
    else:
        is_verbose = False
    breeders, best_guess = get_breeders_from_generation(current_generation, actual_password,
                                                        take_best_N = 250, take_random_N = 100,
                                                        verbose = is_verbose)
    if best_guess == actual_password:
        print("Got the password in %i generation. It's %s"%(i, best_guess))
        break
    current_generation = make_children(breeders, children_per_couple = 3)

def evolve_to_solve(current_generation, actual_password, max_generations, take_best_N, take_random_N,
                    mutation_rate, children_per_couple, print_every_n_generations, verbose = False):
    fitness_tracking = []
    for i in range(max_generations):
        if verbose and not i % print_every_n_generations:
            print("Generation %i: "%i, end = '')
            print(len(current_generation))
            is_verbose = True
        else:
            is_verbose = False
        breeders, best_guess = get_breeders_from_generation(current_generation, actual_password,
                                                            take_best_N = take_best_N, take_random_N = take_random_N,
                                                            verbose = is_verbose, mutation_rate = mutation_rate)
        fitness_tracking.append(fitness_score(best_guess, actual_password))
        if best_guess == actual_password:
            print("Got the password in %i generations. It's '%s'"%(i, best_guess))
            return fitness_tracking
        current_generation = make_children(breeders, children_per_couple = children_per_couple)
    print("Couldn't get the password. My best guess is '%s'"%(best_guess))
    return fitness_tracking
current_generation = create_generation(population = 500, length = 10)
fitness_tracking = evolve_to_solve(current_generation, 'imbobandhi', 1000, 150, 70, 0.5, 3, 5, verbose = True)

def make_fitness_tracking_plot(fitness_tracking, actual_pass):
    plt.figure(dpi = 150)
    plt.plot(range(len(fitness_tracking)), [x/len(actual_pass) for x in fitness_tracking])
    plt.ylabel("Fitness Score (% of max)")
    plt.xlabel("Generation")
    plt.title("Fitness Evolution")
make_fitness_tracking_plot(fitness_tracking, 'imbobandhi')

actual_pass = 'lets see if it can do a whole sentence'
current_generation = create_generation(population = 1000, length = len(actual_pass))
fitness_tracking2 = evolve_to_solve(current_generation, actual_pass, 1000, 350, 100, 0.5, 3, 5, verbose = True)

make_fitness_tracking_plot(fitness_tracking2, actual_pass)
