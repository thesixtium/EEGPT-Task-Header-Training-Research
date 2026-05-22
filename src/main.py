import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import random
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold


# =========================================================
# LEARNING RATE SCHEDULE
# Modify this function however you want
# =========================================================
def learning_rate_schedule(initial_lr, generation, decay=0.97):
    """
    Exponential decay learning rate schedule.

    Example:
        generation 0  -> initial_lr
        generation 10 -> smaller step size
    """
    return initial_lr * (decay ** generation)


# =========================================================
# HYPERPARAMETER SPACE
#
# LIST  -> categorical/discrete values (same behavior)
# DICT  -> continuous/int search space using LR decay
# =========================================================

k = 0.1
# k  	Behavior
# 0.5	huge jumps
# 0.25	aggressive
# 0.1	good default
# 0.05	conservative
# 0.01	tiny local search

# =========================================================
# HYPERPARAMETER SPACE
#
# LIST  -> categorical/discrete values
# DICT  -> numeric search space
#
# Learning rate is automatically computed as:
#     (max - min) * k
# =========================================================

k = 0.1

# k behavior:
# 0.5   huge jumps
# 0.25  aggressive
# 0.1   good default
# 0.05  conservative
# 0.01  tiny local search

hyperparameter_space = {

    # =====================================================
    # CATEGORICAL PARAMETERS
    # =====================================================
    'criterion': ['gini', 'entropy'],
    'splitter': ['best', 'random'],
    'max_features': ['sqrt', 'log2', None],

    # =====================================================
    # INTEGER PARAMETERS
    # =====================================================
    'max_depth': {
        'type': 'int',
        'min': 1,
        'max': 30
    },

    'min_samples_split': {
        'type': 'int',
        'min': 2,
        'max': 20
    },

    'min_samples_leaf': {
        'type': 'int',
        'min': 1,
        'max': 10
    },

    'max_leaf_nodes': {
        'type': 'int',
        'min': 5,
        'max': 100
    },

    # =====================================================
    # FLOAT PARAMETERS
    # =====================================================
    'min_impurity_decrease': {
        'type': 'float',
        'min': 0.0,
        'max': 0.5
    },

    'ccp_alpha': {
        'type': 'float',
        'min': 0.0,
        'max': 0.5
    }
}


# =========================================================
# INITIALIZATION
# =========================================================
def initialize_population(population_size):

    population = []

    for _ in range(population_size):

        individual = {}

        for key, values in hyperparameter_space.items():

            # LIST -> original behavior
            if isinstance(values, list):

                individual[key] = random.choice(values)

            # DICT -> use start value
            elif isinstance(values, dict):

                if values['type'] == 'int':
                    individual[key] = random.randint(
                        values['min'],
                        values['max']
                    )

                elif values['type'] == 'float':
                    individual[key] = random.uniform(
                        values['min'],
                        values['max']
                    )

        population.append(individual)

    return population


# =========================================================
# CROSSOVER
# =========================================================
def crossover(parent1, parent2):

    crossover_point = np.random.randint(1, len(parent1))

    child1 = dict(
        list(parent1.items())[:crossover_point]
        + list(parent2.items())[crossover_point:]
    )

    child2 = dict(
        list(parent2.items())[:crossover_point]
        + list(parent1.items())[crossover_point:]
    )

    return child1, child2


# =========================================================
# MUTATION
# =========================================================
def mutate(individual, mutation_rate, generation):

    for param in individual.keys():

        if random.random() < mutation_rate:

            config = hyperparameter_space[param]

            # =================================================
            # LIST PARAMETERS
            # =================================================
            if isinstance(config, list):

                individual[param] = random.choice(config)

            # =================================================
            # NUMERIC PARAMETERS
            # =================================================
            elif isinstance(config, dict):

                # Automatically calculate LR
                range_size = config['max'] - config['min']

                base_lr = range_size * k

                lr = learning_rate_schedule(
                    base_lr,
                    generation
                )

                # Integer mutation
                if config['type'] == 'int':

                    delta = random.randint(
                        -max(1, int(round(lr))),
                        max(1, int(round(lr)))
                    )

                # Float mutation
                elif config['type'] == 'float':

                    delta = np.random.normal(0, lr)

                new_value = individual[param] + delta

                # Clamp to range
                new_value = max(config['min'], new_value)
                new_value = min(config['max'], new_value)

                # Convert ints
                if config['type'] == 'int':
                    new_value = int(round(new_value))

                individual[param] = new_value

    return individual


# =========================================================
# FITNESS USING K-FOLD CROSS VALIDATION
# =========================================================
def calculate_fitness(X, y, parameters, k_folds=5):

    dt_model = DecisionTreeClassifier(
        random_state=42,
        **parameters
    )

    # Stratified keeps class balance consistent
    kfold = StratifiedKFold(
        n_splits=k_folds,
        shuffle=True,
        random_state=42
    )

    # Compute accuracy across folds
    scores = cross_val_score(
        dt_model,
        X,
        y,
        cv=kfold,
        scoring='accuracy'
    )

    # Return mean accuracy
    return scores.mean()


# =========================================================
# GENETIC ALGORITHM
# =========================================================
def genetic_algorithm(
    X,
    y,
    population_size=13,
    generations=100,
    mutation_rate=0.1
):

    results = []
    seen_combinations = set()

    population = initialize_population(population_size)

    for generation in range(generations):
        print(generation)

        fitness_scores = []

        for parameters in population:

            param_tuple = tuple(sorted(parameters.items()))

            score = calculate_fitness(X, y, parameters)

            # Save only unique combinations
            if param_tuple not in seen_combinations:

                seen_combinations.add(param_tuple)

                result_row = parameters.copy()
                result_row["fitness_score"] = score
                result_row["generation"] = generation

                results.append(result_row)

            fitness_scores.append(score)

        # Best parents
        idx_best_2 = np.argsort(fitness_scores)[::-1][:2]

        new_population = [population[i] for i in idx_best_2]

        # Create offspring
        for _ in range(int((len(population) / 2) - 1)):

            parent1 = new_population[0]
            parent2 = new_population[1]

            child1, child2 = crossover(parent1, parent2)

            child1 = mutate(
                child1,
                mutation_rate,
                generation
            )

            child2 = mutate(
                child2,
                mutation_rate,
                generation
            )

            new_population.extend([child1, child2])

        population = np.array(new_population)

    # Final evaluation
    final_scores = [
        calculate_fitness(X, y, parameters)
        for parameters in population
    ]

    best_parameters = population[np.argmax(final_scores)]
    best_score = max(final_scores)

    # Create dataframe
    df = pd.DataFrame(results)

    # Save CSV
    df.to_csv("df.csv", index=False)

    # =========================================================
    # GENERATION PERFORMANCE GRAPH
    # =========================================================

    # Average fitness per generation
    generation_avg = (
        df.groupby("generation")["fitness_score"]
        .mean()
    )

    # Best fitness INSIDE each generation
    generation_best = (
        df.groupby("generation")["fitness_score"]
        .max()
    )

    # Best fitness found OVERALL up to that generation
    generation_best_overall = generation_best.cummax()

    plt.figure(figsize=(12, 7))

    # ---------------------------------------------------------
    # Average Fitness
    # ---------------------------------------------------------
    plt.plot(
        generation_avg.index,
        generation_avg.values,
        label="Average Fitness"
    )

    # ---------------------------------------------------------
    # Best Fitness Per Generation
    # ---------------------------------------------------------
    plt.plot(
        generation_best.index,
        generation_best.values,
        label="Best Fitness (Generation)"
    )

    # ---------------------------------------------------------
    # Best Overall Fitness So Far
    # ---------------------------------------------------------
    plt.plot(
        generation_best_overall.index,
        generation_best_overall.values,
        linewidth=3,
        linestyle="--",
        label="Best Overall Fitness"
    )

    plt.xlabel("Generation")
    plt.ylabel("Accuracy")

    plt.title("Genetic Algorithm Accuracy Over Generations")

    plt.legend()

    plt.grid(True)

    # Save graph
    plt.savefig(
        "generation_accuracy.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    return best_parameters, best_score, df


# =========================================================
# LOAD DATA
# =========================================================
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"

titanic_df = pd.read_csv(url)

titanic_df = titanic_df.drop(
    ['Name', 'Ticket', 'Cabin', 'PassengerId'],
    axis=1
)

titanic_df = titanic_df.dropna()

titanic_df = pd.get_dummies(
    titanic_df,
    columns=['Sex', 'Embarked'],
    drop_first=True
)

X = titanic_df.drop('Survived', axis=1)
y = titanic_df['Survived']


# =========================================================
# RUN GA
# =========================================================
best_parameters, best_score, df = genetic_algorithm(
    X,
    y,
    population_size=10,
    generations=100,
    mutation_rate=0.25
)

print("Best Parameters:")
print(best_parameters)

print("\nBest Score:")
print(best_score)

print("\nSaved unique combinations to df.csv")
print(df.head())