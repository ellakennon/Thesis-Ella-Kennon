import numpy as np
from operator import itemgetter 

from helpers import *
from model import Model

# Optimizer
class HillClimber():
    def __init__(self, model, maximize, max_iters=1000, restarts=5):
        # Paramteres
        self.model = model
        self.n = model.n
        self.k = model.k
        self.maximize = maximize
        
        # Dictionary of scores for each move
        self.scores = {}

        # Indices of interactions
        self.interactions_idx = get_interactions_idx(self.n, self.k)

        # Restarts and iterations
        self.max_iters = max_iters
        self.restarts = restarts

        # Store intial value for best x and y 
        self.best_found = -np.inf if maximize else np.inf
        self.best_x = None

    def compute_scores(self, x):
        # Initial y
        base_val = self.model.predicting(x)
        # Dictionary of scores
        deltas = {}

        for v in self.interactions_idx:
            x_flip = x.copy()
            
            x_flip[list(v)] ^= 1
            
            x_flip = tuple(x_flip)

            deltas[v] = self.model.predicting(x_flip) - base_val

        return deltas
    
    # Returns best move and it's score
    def pick_best_delta(self, deltas):
        if self.maximize:
            return max(deltas.items(), key=itemgetter(1))
        else:
            return min(deltas.items(), key=itemgetter(1))
        
    def is_improving(self, x, y):
        if (self.maximize and y > self.best_found) or (not self.maximize and y < self.best_found):
            self.best_found = y
            self.best_x = x.copy()

    def optimization(self):
        for r in range(self.restarts):

            x = np.random.randint(0, 2, self.n)
            y = self.model.predicting(x)

            deltas = self.compute_scores(x)

            for _ in range(self.max_iters):
                move, delta = self.pick_best_delta(deltas)

                # Restart if no improving move (local optima)
                if (self.maximize and delta <= 0) or (not self.maximize and delta >= 0):
                    break
                
                x[list(move)] ^= 1
                y += delta
                self.is_improving(x, y)
                
                deltas = self.compute_scores(x)

        return self.best_x, self.best_found
