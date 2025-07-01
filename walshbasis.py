import numpy as np

class WalshBasis:
    def __init__(self, n, k, interactions_nk): 
        self.n = n
        self.k = k

        # Interactions as binary vectors
        self.interactions_nk = interactions_nk

    # Returns the Walsh basis for a given x 
    def get_basis(self, x):  
        # Use multiplication @ with interactions matrix and x
        prod   = self.interactions_nk @ x
        # Find parity of multiplication results
        # Parity is whether the amount of 1s in a vector is even (0) or odd (1)
        # Mod 2 to get parity
        # Parity is a vector with the same number of columns as number of interactions
        parity = prod % 2

        # Multiply each value of parity by 2 
        # Subtract result from 1 (getting +1 or -1)
        # Returns Walsh basis vector 
        return 1-2*parity