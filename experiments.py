import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression, Lasso, Lars, LassoLars
import ioh 

from helpers import *
from model import Model
from walshbasis import WalshBasis
from optimizer import HillClimber

# Generate MAE dataframe for a given n, k 
# Repeats for confidence interval: 5
def mae_exp_nk(regression_type, n, k, problem_name, interactions_nk, walsh_basis, csv_name=None, repeats=5):
    rows = []

    problem = ioh.get_problem(
        problem_name,
        instance=4,
        dimension=n,
        problem_class=ioh.ProblemClass.PBO
    )

    min_train_size, interval, max_train_size = gen_sample_vars(n, k)

    x_test = gen_test_inputs(n, test_size=200)
    y_true = gen_outputs(x_test, problem)

    x_train = gen_train_inputs(n, max_train_size, x_test)
    y_train = gen_outputs(x_train, problem)

    x_train = np.array(x_train)
    y_train = np.array(y_train)

    for r in range(1, repeats + 1):
        print(f'Repeat {r}')   

        indices = np.random.permutation(len(x_train))

        shuffled_x_train = x_train[indices]
        shuffled_y_train = y_train[indices]

        model = Model(regression=regression_type, n=n, k=k, interactions_nk=interactions_nk, walsh_basis=walsh_basis)

        prev_train_size = 0

        for train_size in range(min_train_size, max_train_size + 1, interval):
            cur_x_train = shuffled_x_train[prev_train_size:train_size]
            cur_y_train = shuffled_y_train[prev_train_size:train_size]

            model.add_train_sample(cur_x_train, cur_y_train)

            predictions = model.predicting(x_test)

            mae = mean_absolute_error(y_true, predictions)

            model_call = model.regression

            row = {
                'sample_size': train_size,
                'n': n,
                'k': k,
                'repeat': r,
                'mae': mae,
                'alphas': getattr(model_call, 'alphas_', None),
                'coef_path': getattr(model_call, 'coef_path_', None),
                'coefficients': getattr(model_call, 'coef_', None),
                'non_zero_coef': np.count_nonzero(model_call.coef_),
                'num_iter': getattr(model_call, 'n_iter_', None)
            }

            rows.append(row)

            prev_train_size = train_size

    df = pd.DataFrame(rows)

    if csv_name:
        df.to_csv(csv_name, index=False)
        print(f"Stored results as CSV at: {csv_name}")
            
    # Return full MAE dataframe
    return df


# Runs optimization 
# Repeats for confidence interval: 5
def optimization_exp_nk(regression_type, n, k, problem_name, maximize, interactions_nk, walsh_basis, csv_name=None, repeats=5):
    rows = []

    best_found_all = -np.inf
    best_x_all = None

    problem = ioh.get_problem(
                problem_name,
                instance=4,
                dimension=n,
                problem_class=ioh.ProblemClass.PBO)
    
    min_train_size, interval, max_train_size = gen_sample_vars(n, k)

    x_test = gen_test_inputs(n, test_size=200)
    y_true = gen_outputs(x_test, problem)

    x_train = gen_train_inputs(n, max_train_size, x_test)
    y_train = gen_outputs(x_train, problem)

    x_train = np.array(x_train)
    y_train = np.array(y_train)

    for r in range(1, repeats + 1):
        print(f'Repeat {r}')    
        indices = np.random.permutation(len(x_train))

        shuffled_x_train = x_train[indices]
        shuffled_y_train = y_train[indices]

        model = Model(regression=regression_type, n=n, k=k, interactions_nk=interactions_nk, walsh_basis=walsh_basis)

        prev_train_size = 0

        for train_size in range(min_train_size, max_train_size, interval):                    
            cur_x_train = shuffled_x_train[prev_train_size:train_size]
            cur_y_train = shuffled_y_train[prev_train_size:train_size]

            model.add_train_sample(cur_x_train, cur_y_train)

            best_x, best_found = HillClimber(model, maximize=maximize).optimization()

            if (maximize and best_found > best_found_all) or (not maximize and best_found < best_found_all):
                best_found_all = best_found
                best_x_all = best_x

            row = {
                'sample_size': train_size,
                'n': n,
                'k': k,
                'repeat': r,
                'best_x': tuple(best_x_all),
                'predicted_best': float(best_found_all)
            }

            rows.append(row)

            prev_train_size = train_size

    df = pd.DataFrame(rows)

    if csv_name:
        df.to_csv(csv_name, index=False)
        print(f"Stored results as CSV at: {csv_name}")

    # Return list as dataframe
    return df


# Runs approximation and optimziation experiments at once
def experiment_nk(regression_type, n, k, 
                  problem_name, interactions_nk, 
                  walsh_basis, csv_name, 
                  maximize=True, repeats=5):
    
    rows_mae = []
    rows_opt = []

    best_found_all = -np.inf if maximize else np.inf
    best_x_all = None

    problem = ioh.get_problem(
                problem_name,
                instance=1,
                dimension=n,
                problem_class=ioh.ProblemClass.PBO)
    
    min_train_size, interval, max_train_size = gen_sample_vars(n, k)

    x_test = gen_test_inputs(n, test_size=200)
    y_true = gen_outputs(x_test, problem)

    x_train = gen_train_inputs(n, max_train_size, x_test)
    y_train = gen_outputs(x_train, problem)

    x_train = np.array(x_train)
    y_train = np.array(y_train)

    for r in range(1, repeats+1):
        print(f"Repeat {r}")

        indices = np.random.permutation(len(x_train))

        shuffled_x_train = x_train[indices]
        shuffled_y_train = y_train[indices]

        model = Model(
            regression=regression_type,
            n=n,
            k=k,
            interactions_nk=interactions_nk,
            walsh_basis=walsh_basis,
        )

        prev_train_size = 0

        train_interval = 0
        for train_size in range(min_train_size, max_train_size+1, interval):     
            train_interval += 1

            print(f"On train size {train_interval}/21")     

            cur_x_train = shuffled_x_train[prev_train_size:train_size]
            cur_y_train = shuffled_y_train[prev_train_size:train_size]

            model.add_train_sample(cur_x_train, cur_y_train)

            predictions = model.predicting(x_test)

            mae = mean_absolute_error(y_true, predictions)

            model_call = model.regression

            row_mae = {
                'sample_size': train_size,
                'n': n,
                'k': k,
                'repeat': r,
                'mae': mae,
                'coefficients': getattr(model_call, 'coef_', None),
                'non_zero_coef': np.count_nonzero(model_call.coef_),
                'num_iter': getattr(model_call, 'n_iter_', None)
            }

            rows_mae.append(row_mae)

            best_x, best_found = HillClimber(model, maximize=maximize).optimization()

            if (maximize and best_found > best_found_all) or (not maximize and best_found < best_found_all):
                best_found_all = best_found
                best_x_all = best_x

            row_opt = {
                'sample_size': train_size,
                'n': n,
                'k': k,
                'repeat': r,
                'best_x': tuple(best_x_all),
                'predicted_best': float(best_found_all)
            }

            rows_opt.append(row_opt)

            prev_train_size = train_size

    mae_df = pd.DataFrame(rows_mae)
    opt_df = pd.DataFrame(rows_opt)

    combined_df = mae_df.merge(opt_df, on=['sample_size', 'n', 'k', 'repeat'])

    if csv_name is not None:
        combined_df.to_csv(csv_name, index=False)
        print(f"Stored results as CSV at: {csv_name}")

    return combined_df