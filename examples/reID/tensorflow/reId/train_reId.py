import math

import numpy as np
import tensorflow as tf

Batch_size = 64
MAX_EPOCHES = 120000
learning_rate = 0.01
learning_policy = 'MultiStep'
MultiStepValue = [20000, 100000, 300000]
CurrentAdjustStep = 0


def adjust_learning_rate_by_policy(base_lr_rate, gamma, iter, step, policy):
    if policy == 'MultiStep':
        global CurrentAdjustStep
        if CurrentAdjustStep < len(MultiStepValue) and iter > MultiStepValue[CurrentAdjustStep]:
            CurrentAdjustStep += 1
        return base_lr_rate * math.pow(gamma, CurrentAdjustStep)
    elif policy == 'fixed':
        return base_lr_rate
    elif policy == 'step':
        return base_lr_rate *math.pow(gamma, math.floor(iter/step))
    elif policy == 'exp':
        return base_lr_rate*math.pow(gamma, iter)


def train()




