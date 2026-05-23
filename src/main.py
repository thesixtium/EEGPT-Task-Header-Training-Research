import os
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt

from src.train_new_eegpt_models.datasets.BNCI2015_001 import DatasetBNCI2015_001
from src.train_new_eegpt_models.moabbMotorImageryDataLoader import MoabbMotorImageryDataLoader
from src.core.getLibPaths import GetLibPaths
from src.core.generic_eegpt_model_lib.modelMethods import seed_torch
from src.core.generic_eegpt_model_lib.metricMethods import metrics_display, get_latest_metrics_csv

# Import the training function from your existing trainer module
from src.train_new_eegpt_models.trainEEGPTModelFromDataset import train_EEGPT_model_from_dataset


# =========================================================
# LEARNING RATE SCHEDULE
# =========================================================
def learning_rate_schedule(initial_lr, generation, decay=0.97):
    """Exponential decay: shrinks mutation step size over generations."""
    return initial_lr * (decay ** generation)


# =========================================================
# HYPERPARAMETER SPACE
#
# LIST  -> categorical / discrete  (random choice)
# DICT  -> numeric                 (mutated with LR-scaled delta)
# =========================================================

k = 0.1  # mutation scale factor

hyperparameter_space = {

    # ---- categorical ----
    'lr_scheduler_name': [
        "OneCycleLR",
        "StepLR",
        "ReduceLROnPlateau",
        "ExponentialLR",
        "CosineAnnealingLR",
    ],
    'target_name': ["f1score", "loss", "mcc"],

    # ---- integer ----
    'max_epochs': {
        'type': 'int',
        'min': 1,
        'max': 100,
    },

    # ---- float ----
    'max_learning_rate': {
        'type': 'float',
        'min': 1e-6,
        'max': 1e-2,
    },
    'gamma': {
        'type': 'float',
        'min': 0.001,
        'max': 0.999,
    },
}


# =========================================================
# INITIALIZATION
# =========================================================
def initialize_population(population_size):
    population = []
    for _ in range(population_size):
        individual = {}
        for key, values in hyperparameter_space.items():
            if isinstance(values, list):
                individual[key] = random.choice(values)
            elif isinstance(values, dict):
                if values['type'] == 'int':
                    individual[key] = random.randint(values['min'], values['max'])
                elif values['type'] == 'float':
                    individual[key] = random.uniform(values['min'], values['max'])
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

            if isinstance(config, list):
                individual[param] = random.choice(config)

            elif isinstance(config, dict):
                range_size = config['max'] - config['min']
                base_lr = range_size * k
                lr = learning_rate_schedule(base_lr, generation)

                if config['type'] == 'int':
                    delta = random.randint(
                        -max(1, int(round(lr))),
                         max(1, int(round(lr)))
                    )
                elif config['type'] == 'float':
                    delta = np.random.normal(0, lr)

                new_value = individual[param] + delta
                new_value = max(config['min'], min(config['max'], new_value))

                if config['type'] == 'int':
                    new_value = int(round(new_value))

                individual[param] = new_value

    return individual


# =========================================================
# FITNESS  –  trains one EEGPT model and returns a scalar
# =========================================================
def calculate_fitness(loaded_data, dataset, parameters, glp, base_model, run_id):
    """
    Train a GenericEEGPTModel with the given hyperparameters and return
    the best validation accuracy as the fitness score.

    Parameters
    ----------
    loaded_data : MoabbMotorImageryDataLoader
    parameters  : dict  – keys from hyperparameter_space
    glp         : GetLibPaths
    base_model  : Path  – pretrained EEGPT checkpoint
    run_id      : str   – unique name so every run saves its own checkpoint
    """

    model_name = f"ga_run_{run_id}"

    train_EEGPT_model_from_dataset(
        model_name=model_name,
        data=loaded_data,
        use_channels_names=dataset.get_use_channels_names(),
        base_model=base_model,
        max_epochs=parameters['max_epochs'],
        max_lr=parameters['max_learning_rate'],
        output_classes=dataset.get_n_classes(),
        glp=glp,
        lr_scheduler_name=parameters['lr_scheduler_name'],
        gamma=parameters['gamma'],
        target_name=parameters['target_name'],
    )

    # Read the best valid_acc that was logged to CSV
    metrics_csv = get_latest_metrics_csv(glp.get_logs_path(), model_name)
    df_metrics = pd.read_csv(metrics_csv)

    # valid_acc is logged per epoch; grab the best value
    valid_acc_col = df_metrics['valid_acc'].dropna()
    fitness = float(valid_acc_col.max()) if len(valid_acc_col) > 0 else 0.0

    return fitness

# =========================================================
# STATUS FILE
# =========================================================
STATUS_FILE = "ga_eegpt_status.txt"

def log_status(msg: str, filepath: str = STATUS_FILE):
    """Print msg and append it to the status file."""
    print(msg)
    with open(filepath, "a") as f:
        f.write(msg + "\n")

# =========================================================
# GENETIC ALGORITHM
# =========================================================
def genetic_algorithm(
    loaded_data,
    dataset,
    glp,
    base_model,
    population_size=10,
    generations=20,
    mutation_rate=0.25,
):
    results = []
    seen_combinations = set()

    open(STATUS_FILE, "w").close()

    population = initialize_population(population_size)

    for generation in range(generations):
        print(f"\n{'='*50}")
        log_status(f"  Generation {generation + 1} / {generations}")
        print(f"  Generation {generation + 1} / {generations}")
        print(f"{'='*50}")

        fitness_scores = []

        for idx, parameters in enumerate(population):
            param_tuple = tuple(sorted(parameters.items()))

            run_id = f"gen{generation:03d}_ind{idx:03d}"
            log_status(f"\n  Individual {idx} | {parameters}")
            print(f"\n  Individual {idx} | {parameters}")

            score = calculate_fitness(loaded_data, dataset, parameters, glp, base_model, run_id)
            print(f"  Fitness (valid_acc): {score:.4f}")

            fitness_scores.append(score)

            # Record only unique hyperparameter combinations
            if param_tuple not in seen_combinations:
                seen_combinations.add(param_tuple)
                result_row = parameters.copy()
                result_row["fitness_score"] = score
                result_row["generation"] = generation
                results.append(result_row)

        # ---- elitism: keep the 2 best parents ----
        idx_best_2 = np.argsort(fitness_scores)[::-1][:2]
        new_population = [population[i] for i in idx_best_2]

        print(f"\n  Best this generation: {max(fitness_scores):.4f}")
        print(f"  Best params: {new_population[0]}")

        # ---- breed offspring to refill population ----
        while len(new_population) < population_size:
            parent1 = new_population[0]
            parent2 = new_population[1]
            child1, child2 = crossover(parent1, parent2)
            child1 = mutate(child1, mutation_rate, generation)
            child2 = mutate(child2, mutation_rate, generation)
            new_population.extend([child1, child2])

        # Trim to exact population size (crossover may overshoot by 1)
        population = new_population[:population_size]

    # ---- final evaluation on surviving population ----
    print("\nRunning final evaluation on last population...")
    final_scores = [
        calculate_fitness(
            loaded_data, dataset, params, glp, base_model,
            run_id=f"final_{i:03d}"
        )
        for i, params in enumerate(population)
    ]

    best_idx = int(np.argmax(final_scores))
    best_parameters = population[best_idx]
    best_score = final_scores[best_idx]

    # ---- save results ----
    df = pd.DataFrame(results)
    df.to_csv("ga_eegpt_results.csv", index=False)

    # ---- plot ----
    generation_avg = df.groupby("generation")["fitness_score"].mean()
    generation_best = df.groupby("generation")["fitness_score"].max()
    generation_best_overall = generation_best.cummax()

    plt.figure(figsize=(12, 7))
    plt.plot(generation_avg.index, generation_avg.values, label="Average Fitness")
    plt.plot(generation_best.index, generation_best.values, label="Best Fitness (Generation)")
    plt.plot(
        generation_best_overall.index,
        generation_best_overall.values,
        linewidth=3,
        linestyle="--",
        label="Best Overall Fitness",
    )
    plt.xlabel("Generation")
    plt.ylabel("Validation Accuracy")
    plt.title("Genetic Algorithm – EEGPT Accuracy Over Generations")
    plt.legend()
    plt.grid(True)
    plt.savefig("ga_eegpt_accuracy.png", dpi=300, bbox_inches="tight")
    plt.close()

    return best_parameters, best_score, df


# =========================================================
# ENTRY POINT
# =========================================================
if __name__ == '__main__':

    seed_torch(7_11_2002)

    glp = GetLibPaths()
    base_model = glp.get_checkpoints_path() / "eegpt_mcae_58chs_4s_large4E.ckpt"

    dataset = DatasetBNCI2015_001()
    loaded_data = MoabbMotorImageryDataLoader(dataset)

    best_parameters, best_score, df = genetic_algorithm(
        loaded_data,
        dataset,
        glp,
        base_model,
        population_size=100,
        generations=100,
        mutation_rate=0.25,
    )

    print("\nBest Parameters:")
    print(best_parameters)
    print("\nBest Validation Accuracy:")
    print(best_score)
    print("\nAll unique combinations saved to ga_eegpt_results.csv")
    print(df.head())