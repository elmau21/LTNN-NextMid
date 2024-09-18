
import torch

class LogicalMask:
    def __init__(self, logic_fn):
        self.logic_fn = logic_fn

    def __call__(self, Q, K):
        return self.logic_fn(Q, K)
