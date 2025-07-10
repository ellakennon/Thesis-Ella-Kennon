import numpy as np
from itertools import combinations

# Sample size intervals for given n, k
# To be used with range()
def gen_sample_vars(n, k):
    if n==10:
        min_sample = 5
        if k==1:
            sample_interval = 10
        else:
            sample_interval = 20
    elif n==25:
        min_sample = 10
        if k==1:
            sample_interval = 20
        else:
            sample_interval = 40
    elif n==50:
        min_sample = 25
        if k==1:
            sample_interval = 50
        else:
            sample_interval = 100
    else:
        min_sample = 50
        if k==1:
            sample_interval = 75
        else:
            sample_interval = 250

    max_sample = min_sample + sample_interval * 20

    return min_sample, sample_interval, max_sample

# Randomly generates unique binary inputs
# Used only for test set
def gen_test_inputs(n, test_size):
    # Initialize the x as a set to prevent repeat values
    inputs = set()

    # While the samples size hasn't reached the expected size
    while len(inputs) < test_size:
        # Generate random binary value as a tuple (for set)
        x = tuple(np.random.randint(0, 2, size=n))
        # Add x to set
        inputs.add(x)

    # Convert x from tuple to list
    inputs = [list(x) for x in inputs]

    # Return inputs
    return inputs

# Generates training inputs with regards to the test set
def gen_train_inputs(n, train_size, test_set):
    '''Gemerates training samples.
    
    Returns:
    np.array: Training data set'''
    
    # Determine the maximum number of inputs possible
    max_size = 2 ** n

    # If the requested size exceeds the maximum exit
    if train_size + len(test_set) > max_size:
        raise ValueError(f"Asked for {train_size} but only {max_size} unique vectors exist.")
    
    # Initialize the x as a set to prevent repeat values
    inputs = set()

    # Set test set as tuples
    test_tuple = {tuple(row) for row in test_set}

    # While the samples size hasn't reached the expected size
    while len(inputs) < train_size:
        # Generate random binary value as a tuple (for set)
        x = tuple(np.random.randint(0, 2, size=n))
        
        if x not in test_tuple:
            # Add x to set
            inputs.add(x)

    # Convert x from tuple to list
    inputs = [np.asarray(x) for x in inputs]

    # Return inputs
    return inputs

# Retrieves the outputs of the generated input data 
def gen_outputs(inputs, problem):
    # Put inputs through the problem to get a list of outputs
    outputs = [problem(x) for x in inputs]

    # Return outputs
    return outputs

# Gets the interactions indices to be used later
def get_interactions_idx(n, k):
    # Initialize list of interactions
    interactions_idx = []

    # Loop through each interaction combination based on n and k
    for i in range(1, k+1):
        for j in combinations(range(n), i):
            # Add the interaction index to the list
            interactions_idx.append(j)
    
    # Return list of interactions indices
    return interactions_idx

# Gets a matrix of the interactions as binary vectors
def get_interactions_vec(interactions, n):
    # Initialize interaction matrix
    # Will have a row for each interaction and n columns, one for each variable
    inter_matrix = np.zeros((len(interactions), n))

    # Loop through interactions
    for i, idx in enumerate(interactions):
        # At the current position i, set each index in the interactions to 1
        # Creates the interactions indices list corresponding interactions matrix
        # Used with an input x to find the Walsh basis 
        inter_matrix[i, list(idx)] = 1

    # Return the interactions matrix
    return np.array(inter_matrix)  

# Remove largest MAE
# Designed for Lars
def remove_outliers(df, num_outliers):
    # Retrieves indices of largest num_outliers mean MAE values (over 5 repeats) by sample_size
    outlier_idxs = (df.groupby('sample_size')['mae'].mean().nlargest(num_outliers).index)

    # Copy and set outlier(s) indices to NaN
    no_outliers_df = df.copy()
    no_outliers_df.loc[no_outliers_df['sample_size'].isin(outlier_idxs), 'mae'] = np.nan

    return no_outliers_df