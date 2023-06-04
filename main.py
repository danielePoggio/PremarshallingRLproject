#!/usr/bin/python3
# -*- coding: utf-8 -*-
from experiment import decide_next_steps

num_experiment = 1
experiment_trials = []
for i in range(0, num_experiment):
    experiment_trials.append(decide_next_steps(True, True, True, True))
# decide_next_steps(True, True, True, True)
print("Finito")
