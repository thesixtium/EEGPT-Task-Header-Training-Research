import os
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt

from src.train_new_eegpt_models.gdfToCSV import convert_bciciv2b_directory_to_single_csv
from src.train_new_eegpt_models.csvDataLoader import CsvEegDataLoader
from src.train_new_eegpt_models.datasets.dataset import Dataset
from src.train_new_eegpt_models.moabbMotorImageryDataLoader import MoabbMotorImageryDataLoader
from src.core.getLibPaths import GetLibPaths
from src.core.generic_eegpt_model_lib.modelMethods import seed_torch
from src.core.generic_eegpt_model_lib.metricMethods import get_latest_metrics_csv

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
        'min': 10,
        'max': 30,
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
# TOURNAMENT SELECTION
# Replaces always-breed-from-top-2, which collapses diversity
# by generation 3-4. Tournament selection keeps a healthy spread
# while still favouring high-fitness individuals.
# =========================================================
def tournament_select(population, fitness_scores, k=3):
    """Pick k random individuals, return the best one."""
    indices = random.sample(range(len(population)), min(k, len(population)))
    best = max(indices, key=lambda i: fitness_scores[i])
    return population[best]


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
# FITNESS CACHE
# Avoids re-training identical hyperparameter combinations,
# which happen frequently once the population converges.
# =========================================================
_fitness_cache: dict[tuple, float] = {}


def _params_key(parameters: dict) -> tuple:
    return tuple(sorted(parameters.items()))


# =========================================================
# CORE TRAINING CALL
# Separated so proxy and full evals share the same logic.
# =========================================================
def _run_training(loaded_data, dataset, parameters, glp, base_model, run_id, epochs_override=None):
    """
    Train for `epochs_override` epochs (or parameters['max_epochs'] if None)
    and return the best valid_acc seen.
    """
    epochs = epochs_override if epochs_override is not None else parameters['max_epochs']

    model_name = f"ga_run_{run_id}"

    train_EEGPT_model_from_dataset(
        model_name=model_name,
        data=loaded_data,
        use_channels_names=dataset.get_use_channels_names(),
        base_model=base_model,
        max_epochs=epochs,
        max_lr=parameters['max_learning_rate'],
        output_classes=dataset.get_n_classes(),
        glp=glp,
        lr_scheduler_name=parameters['lr_scheduler_name'],
        gamma=parameters['gamma'],
        target_name=parameters['target_name'],
    )

    metrics_csv = get_latest_metrics_csv(glp.get_logs_path(), model_name)
    df_metrics = pd.read_csv(metrics_csv)
    valid_acc_col = df_metrics['valid_acc'].dropna()
    return float(valid_acc_col.max()) if len(valid_acc_col) > 0 else 0.0


# =========================================================
# FITNESS  –  proxy + full evaluation with caching
#
# Strategy
# --------
# 1. Check the cache — if this exact config was already trained,
#    return the stored score immediately (no GPU time spent).
# 2. Run a cheap 3-epoch proxy to rank individuals quickly.
# 3. Only fully train the top `full_eval_frac` fraction; the
#    rest keep their proxy score.  The proxy score is a noisy
#    but directionally correct signal — good enough to discard
#    clearly bad configs without wasting 20-30 epochs on them.
#
# full_eval_frac=0.2 means the bottom 80 % of the population
# each generation costs only 3 epochs instead of up to 30,
# giving roughly a 5-8x wall-time reduction per generation.
# =========================================================
def calculate_fitness(
    loaded_data,
    dataset,
    parameters,
    glp,
    base_model,
    run_id,
    proxy_epochs=3,
    is_full_eval=False,
):
    """
    Return a fitness score for `parameters`.

    Parameters
    ----------
    proxy_epochs   : epochs used for the cheap initial ranking pass
    is_full_eval   : if True, skip the proxy and run max_epochs directly
                     (used for the top candidates and the final evaluation)
    """
    key = _params_key(parameters)

    # -- cache hit --
    if key in _fitness_cache:
        log_status(f"    [cache hit] skipping training for {parameters}")
        return _fitness_cache[key]

    if is_full_eval:
        score = _run_training(loaded_data, dataset, parameters, glp, base_model, run_id)
    else:
        score = _run_training(
            loaded_data, dataset, parameters, glp, base_model,
            run_id=f"{run_id}_proxy", epochs_override=proxy_epochs,
        )

    _fitness_cache[key] = score
    return score


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
    population_size: int = 20,
    generations: int = 30,
    mutation_rate: float = 0.30,
    proxy_epochs: int = 3,
    full_eval_frac: float = 0.20,
    elitism_n: int = 2,
    tournament_k: int = 3,
):
    """
    Parameters
    ----------
    population_size  : number of individuals per generation
    generations      : number of generations to run
    mutation_rate    : probability of mutating each parameter
    proxy_epochs     : epochs for the cheap ranking pass
    full_eval_frac   : fraction of population that gets a full-epoch eval
                       each generation (the rest keep their proxy score)
    elitism_n        : number of top individuals carried over unchanged
    tournament_k     : candidates drawn per tournament selection
    """
    results = []
    seen_combinations = set()

    open(STATUS_FILE, "w").close()

    population = initialize_population(population_size)
    n_full = max(elitism_n, int(round(population_size * full_eval_frac)))

    for generation in range(generations):
        log_status(f"\n{'='*54}")
        log_status(f"  Generation {generation + 1} / {generations}")
        log_status(f"{'='*54}")

        # --------------------------------------------------
        # Phase 1 — cheap proxy scoring for the whole pop
        # --------------------------------------------------
        proxy_scores = []
        for idx, parameters in enumerate(population):
            run_id = f"gen{generation:03d}_ind{idx:03d}"
            log_status(f"  [proxy] individual {idx} | {parameters}")

            score = calculate_fitness(
                loaded_data, dataset, parameters, glp, base_model,
                run_id=run_id,
                proxy_epochs=proxy_epochs,
                is_full_eval=False,
            )
            proxy_scores.append(score)
            log_status(f"    proxy score: {score:.4f}")

        # --------------------------------------------------
        # Phase 2 — full eval on top n_full individuals only
        # --------------------------------------------------
        top_indices = np.argsort(proxy_scores)[::-1][:n_full]
        fitness_scores = proxy_scores.copy()

        log_status(f"\n  Full eval on top {n_full} individuals: {list(top_indices)}")
        for idx in top_indices:
            parameters = population[idx]
            run_id = f"gen{generation:03d}_ind{idx:03d}_full"
            log_status(f"  [full ] individual {idx} | {parameters}")

            score = calculate_fitness(
                loaded_data, dataset, parameters, glp, base_model,
                run_id=run_id,
                is_full_eval=True,
            )
            fitness_scores[idx] = score
            log_status(f"    full score:  {score:.4f}")

        # --------------------------------------------------
        # Logging
        # --------------------------------------------------
        best_gen_score = max(fitness_scores)
        best_gen_idx   = int(np.argmax(fitness_scores))
        log_status(f"\n  Best this generation: {best_gen_score:.4f} (individual {best_gen_idx})")
        log_status(f"  Best params: {population[best_gen_idx]}")

        for idx, (parameters, score) in enumerate(zip(population, fitness_scores)):
            param_tuple = _params_key(parameters)
            if param_tuple not in seen_combinations:
                seen_combinations.add(param_tuple)
                result_row = parameters.copy()
                result_row["fitness_score"] = score
                result_row["generation"]    = generation
                results.append(result_row)

        # --------------------------------------------------
        # Selection — elitism + tournament breeding
        # --------------------------------------------------
        elite_indices  = np.argsort(fitness_scores)[::-1][:elitism_n]
        new_population = [population[i] for i in elite_indices]

        while len(new_population) < population_size:
            parent1 = tournament_select(population, fitness_scores, k=tournament_k)
            parent2 = tournament_select(population, fitness_scores, k=tournament_k)
            child1, child2 = crossover(parent1, parent2)
            child1 = mutate(child1, mutation_rate, generation)
            child2 = mutate(child2, mutation_rate, generation)
            new_population.extend([child1, child2])

        population = new_population[:population_size]

    # --------------------------------------------------
    # Final evaluation — full epochs on surviving population
    # --------------------------------------------------
    log_status("\nRunning final full evaluation on last population...")
    final_scores = []
    for i, params in enumerate(population):
        score = calculate_fitness(
            loaded_data, dataset, params, glp, base_model,
            run_id=f"final_{i:03d}",
            is_full_eval=True,
        )
        final_scores.append(score)

    best_idx        = int(np.argmax(final_scores))
    best_parameters = population[best_idx]
    best_score      = final_scores[best_idx]

    # --------------------------------------------------
    # Save results
    # --------------------------------------------------
    df = pd.DataFrame(results)
    df.to_csv("ga_eegpt_results.csv", index=False)

    generation_avg          = df.groupby("generation")["fitness_score"].mean()
    generation_best         = df.groupby("generation")["fitness_score"].max()
    generation_best_overall = generation_best.cummax()

    plt.figure(figsize=(12, 7))
    plt.plot(generation_avg.index,          generation_avg.values,          label="Average Fitness")
    plt.plot(generation_best.index,         generation_best.values,         label="Best Fitness (Generation)")
    plt.plot(generation_best_overall.index, generation_best_overall.values,
             linewidth=3, linestyle="--",   label="Best Overall Fitness")
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

    glp        = GetLibPaths()
    base_model = glp.get_checkpoints_path() / "eegpt_mcae_58chs_4s_large4E.ckpt"

    _, trial_length = convert_bciciv2b_directory_to_single_csv(
        gdf_dir=r"C:\Users\ajrbe\Downloads\BCICIV_2b_gdf",
        csv_path="bciciv2b_all.csv",
        verbose=True,
    )

    loaded_data = CsvEegDataLoader(
        csv_path="bciciv2b_all.csv",
        class_names=["left_hand", "right_hand"],
        trial_length=trial_length,
        batch_size=32,
        target_sample=1000,   # match trial_length exactly — avoids the 93x resample artefact
        timestamp_col=None,
        label_col="y",
    )

    dataset = Dataset(
        name="BCICIV_2b",
        dataset=None,
        n_classes=2,
        fmin=0.5,
        fmax=100.0,
        tmax=4.0,
        sample_rate=250,
        use_channels_names=["C3", "Cz", "C4"],
    )

    best_parameters, best_score, df = genetic_algorithm(
        loaded_data,
        dataset,
        glp,
        base_model,
        population_size=20,      # was 100 — diversity maintained via tournament selection
        generations=30,          # was 100 — net evaluations are similar due to proxy + cache
        mutation_rate=0.30,      # slightly higher than 0.25 to compensate for smaller pop
        proxy_epochs=3,          # cheap ranking pass — bad configs die here
        full_eval_frac=0.20,     # only top 20% (4 individuals) get full max_epochs training
        elitism_n=2,             # top 2 always survive unchanged
        tournament_k=3,          # tournament pool size for parent selection
    )

    log_status("\nBest Parameters:")
    log_status(str(best_parameters))
    log_status(f"\nBest Validation Accuracy: {best_score:.4f}")
    log_status("\nAll unique combinations saved to ga_eegpt_results.csv")
    print(df.head())