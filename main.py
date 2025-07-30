import cProfile
import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import threading
import zlib
import time
import traceback
import json
import argparse
import warnings

from model_evaluation import evaluate_model
from model_training import preprocess_and_train

from sklearn.exceptions import ConvergenceWarning
from data_loader import load_data, create_dir_path_results

from utils import is_valid_filename, save_pkl

warnings.filterwarnings("ignore", category=SyntaxWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)


completed_csv_lock = threading.Lock()


def main():
    run_with_exceptions_path: str = 'run_with_exceptions/'

    if not os.path.isdir(run_with_exceptions_path):
        os.makedirs(run_with_exceptions_path, exist_ok=True)

    parser = argparse.ArgumentParser(description="Train a model on air pollution data.")

    # Command-line arguments
    parser.add_argument("--data", type=str, required=True, help="Dataset name.")
    parser.add_argument("--scaling", type=str, required=True, help="Scaling strategy for numerical features.")
    parser.add_argument("--model", type=str, required=True, help="Model name (e.g., 'random_forest', 'linear_regression').")
    parser.add_argument("--test_size", type=float, required=True, help="Proportion of the dataset to use for testing.")
    parser.add_argument("--func_set", type=str, required=True, help="Comma-separated list of functions.")
    parser.add_argument("--linear_scaling", type=int, required=True, help="Whether or not performing linear scaling.")
    parser.add_argument("--pop_size", type=int, required=True, help="Population size.")
    parser.add_argument("--num_gen", type=int, required=True, help="Number of generations.")
    parser.add_argument("--tournament_size", type=int, required=True, help="Tournament size.")
    parser.add_argument("--perc_train_records", type=float, required=True, help="Percentage of records to use for training.")
    parser.add_argument("--seed_index", type=int, required=True, help="Random seed_index for reproducibility.")
    parser.add_argument("--run_id", type=str, default='default', help="The run id, used for logging purposes of successful runs.")
    parser.add_argument("--verbose", required=False, action="store_true", help="Verbose flag.")
    parser.add_argument("--profile", required=False, action="store_true", help="Whether to run and log profiling of code or not.")

    args = parser.parse_args()

    data = args.data
    scaling = args.scaling
    model = args.model
    test_size = float(args.test_size)
    func_set = args.func_set
    linear_scaling = int(args.linear_scaling)
    pop_size = int(args.pop_size)
    num_gen = int(args.num_gen)
    tournament_size = int(args.tournament_size)
    perc_train_records = float(args.perc_train_records)
    seed_index = int(args.seed_index)
    run_id: str = args.run_id

    verbose: int = int(args.verbose)
    profiling: int = int(args.profile)

    args_string = ";".join(f"{key};{vars(args)[key]}" for key in sorted(list(vars(args).keys())) if key not in ('profile', 'verbose'))
    all_items_string = ";".join(f"{key}={value}" for key, value in vars(args).items())

    pr = None
    if profiling != 0:
        pr = cProfile.Profile()
        pr.enable()

    try:
        if not is_valid_filename(run_id):
            raise ValueError(f'run_id {run_id} is not a valid filename.')

        if seed_index < 1:
            raise AttributeError(f'seed_index does not start from 1, it is {seed_index}.')

        with open('random_seeds.txt', 'r') as f:
            # THE ACTUAL SEED TO BE USED IS LOCATED AT POSITION seed_index - 1 SINCE seed_index IS AN INDEX THAT STARTS FROM 1
            all_actual_seeds = [int(curr_actual_seed_as_str) for curr_actual_seed_as_str in f.readlines()]
        seed = all_actual_seeds[seed_index - 1]

        # Load datasets
        df = load_data(f"datasets/{data}/data.csv")
        functions = func_set.split(',')

        # Call training function
        start_time = time.time()
        estimator, X_train, X_test, y_train, y_test = preprocess_and_train(
            df=df,
            scaling_strategy=scaling,
            model_name=model,
            test_size=test_size,
            seed=seed,
            functions=functions,
            pop_size=pop_size,
            num_gen=num_gen,
            tournament_size=tournament_size,
            perc_train_records=perc_train_records,
            linear_scaling=linear_scaling != 0,
            verbose=verbose
        )
        end_time = time.time()
        execution_time_in_minutes = (end_time - start_time) * (1 / 60)

        metrics = ['r2', 'mae', 'mse', 'rmse']
        score_dict = {k: evaluate_model(estimator, X_train, X_test, y_train, y_test, k) for k in metrics}

        if model not in ('symbolic_regression', 'genetic_programming'):
            functions = []
            pop_size = 0
            num_gen = 0
            tournament_size = 0

        final_path = create_dir_path_results(
            'results/',
            dataset=data,
            scaling=scaling,
            model=model,
            functions=functions,
            test_size=test_size,
            linear_scaling=linear_scaling,
            pop_size=pop_size,
            num_gen=num_gen,
            tournament_size=tournament_size,
            perc_train_records=perc_train_records,
        )
        if not os.path.isdir(final_path):
            os.makedirs(final_path, exist_ok=True)

        result_dict = {'seed': seed, 'seed_index': seed_index,
                       'functions': functions,
                       'data': data, 'data_bench': data.split('_')[0], 'run_id': run_id,
                       'execution_time_in_minutes': execution_time_in_minutes}

        for k in metrics:
            train_score, test_score = score_dict[k]['train_score'], score_dict[k]['test_score']

            result_dict[k] = {'train_score': train_score, 'test_score': test_score}

        with open(os.path.join(final_path, f'result{seed_index}.json'), 'w') as f:
            json.dump(result_dict, f, indent=4)

        save_pkl(estimator, os.path.join(final_path, f'estimator{seed_index}.pkl'))

        if model == 'symbolic_regression':
            sr_model = estimator.pipeline['regressor']
            equations = sr_model.equations
            equations.to_csv(os.path.join(final_path, f'equations{seed_index}.tsv'), sep='\t', index=False)

        if model == 'genetic_programming':
            gp_model = estimator.pipeline['regressor']
            gp_dict_gen = {'model': gp_model.get_model(), 'n_nodes': gp_model.get_n_nodes(),
                           'n_evals': gp_model.get_evaluations(), 'progress': gp_model.get_progress_log()}
            with open(os.path.join(final_path, f'equations{seed_index}.json'), 'w') as f:
                json.dump(gp_dict_gen, f, indent=4)

        with completed_csv_lock:
            with open(os.path.join('results/', f'completed_{run_id}.txt'), 'a+') as terminal_std_out:
                terminal_std_out.write(args_string)
                terminal_std_out.write('\n')
            print(f'Completed run: {all_items_string}.')
    except Exception as e:
        try:
            error_string = str(traceback.format_exc())
            with open(os.path.join(run_with_exceptions_path, f'error_{zlib.adler32(bytes(args_string, "utf-8"))}.txt'), 'w') as f:
                f.write(args_string + '\n\n' + all_items_string + '\n\n' + error_string)
            print(f'\nException in run: {all_items_string}.\n\n{str(e)}\n\n')
        except Exception as ee:
            with open(os.path.join(run_with_exceptions_path, 'error_in_error.txt'), 'w') as f:
                f.write(str(traceback.format_exc()) + '\n\n')
            print(str(ee))

    if profiling != 0:
        pr.disable()
        pr.print_stats(sort='tottime')


if __name__ == '__main__':
    main()
