import os
import time

import numpy as np
import pygad
import pandas as pd
from DeployAutoEncoder import generate_random
from LSR_comm import LSR_comm
from SpectraWizSaver import save_curve

def readAndCurateCurve(file):
    with open(file, 'rb') as f2:
        curve = pd.read_csv(f2, delimiter=" ", skiprows=1, names=['nm', 'ignore', 'value'])
        curve = curve.loc[(curve['nm'] >= 350) & (curve['nm'] <= 850)]
        curve = curve.groupby(np.arange(len(curve)) // 5).agg({"nm": 'mean', 'value': 'mean'})
        curve[curve < 0] = 0
        #curve['value'] = transformToLog10(curve['value'] + EPS)
        return curve

function_inputs = generate_random()
desired_output = readAndCurateCurve("ZAGREB071022/Akrozprozor.ssm").value

def fitness_func(solution, soulution_idx):

    # lsr = LSR_comm("COM3")
    # time.sleep(1)
    # # Start LSR with params
    # lsr.set_column_data(1, solution)
    # lsr.set_column_data(2, lsr.compute_column_based_on_first(0.7))
    # lsr.set_column_data(3, lsr.compute_column_based_on_first(0.5))
    # lsr.set_column_data(4, lsr.compute_column_based_on_first(0.3))
    # lsr.run()
    #
    # # Spectra has to point to example_database folder before starting
    # save_curve("{}".format("recreated.ssm"))
    # print("Waiting for recreated file to be saved...")
    # time.sleep(2)
    # while not os.path.exists("example_database/{}".format("recreated.ssm")):
    #     time.sleep(1)
    #
    # print("\t Reading new HyperOCR data...")
    # # Read HYperOCR (Current Curve)
    # sensor_reading = readAndCurateCurve("example_database/recreated.ssm")

    sensor_reading = np.random.randint(65000, size=201)

    mse = (np.square(desired_output - sensor_reading)).mean(axis=0)
    return mse

fitness_function = fitness_func

num_generations = 500
num_parents_mating = 8

sol_per_pop = 8
num_genes = len(function_inputs)

init_range_low = 0
init_range_high = 1000

parent_selection_type = "rank"
keep_parents = 1

crossover_type = "two_points"

mutation_type = "random"
mutation_percent_genes = 30

ga_instance = pygad.GA(num_generations=num_generations,
                       num_parents_mating=num_parents_mating,
                       fitness_func=fitness_function,
                       sol_per_pop=sol_per_pop,
                       num_genes=num_genes,
                       init_range_low=init_range_low,
                       init_range_high=init_range_high,
                       parent_selection_type=parent_selection_type,
                       keep_parents=keep_parents,
                       crossover_type=crossover_type,
                       mutation_type=mutation_type,
                       mutation_percent_genes=mutation_percent_genes)

ga_instance.run()

solution, solution_fitness, solution_idx = ga_instance.best_solution()
print("Parameters of the best solution : {solution}".format(solution=solution))
print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))

prediction = np.sum(np.array(function_inputs)*solution)
print("Predicted output based on the best solution : {prediction}".format(prediction=prediction))