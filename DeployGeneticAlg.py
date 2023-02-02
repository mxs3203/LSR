import os
import random
import time

import numpy as np
import pygad
import pandas as pd
import torch
import scipy.stats as st

import admin
from LSR_comm import LSR_comm
from SpectraWizSaver import save_curve
import matplotlib.pyplot as plt

from modeling.Predict10 import Predict10

# if not admin.isUserAdmin():
#     admin.runAsAdmin()
def generate_random():
    max = [100, 100, 100, 100, 100, 100, 100, 100, 100, 100]
    return np.array(random.sample(range(1, 1000), 10), dtype="int")

def readAndCurateCurve(file, EPS=0.0001):
    with open(file, 'rb') as f2:
        curve = pd.read_csv(f2, delimiter=" ", skiprows=1, names=['nm', 'ignore', 'value'])
        curve = curve.loc[(curve['nm'] >= 350) & (curve['nm'] <= 850)]
        curve = curve.groupby(np.arange(len(curve)) // 5).agg({"nm": 'mean', 'value': 'mean'})
        curve[curve < 0] = 0
        log10_curve = np.log10(curve['value'] + EPS)
        return curve,log10_curve

input_curve_file = "modeling/mlj15_1001_1158_uW.IRR"#r"C:\Users\Korisnik\Desktop\mlj15_1001_1158_uW.IRR"
file_name = input_curve_file.split("\\")[-1]
function_inputs = generate_random()
desired_output_all, desired_log10_curve = readAndCurateCurve(input_curve_file)
desired_output = desired_output_all['value'].values

def scale_curve(x_in):
    mn, mx = 0, 500
    x_scaled = (x_in - mn) / (mx - mn)
    return x_scaled

def fitness_func(solution, soulution_idx):

    solution = [int(ele) for ele in solution]
    lsr = LSR_comm("COM3")
    # Start LSR with params
    lsr.set_column_data(1, solution)
    lsr.set_column_data(2, lsr.compute_column_based_on_first(0.7))
    lsr.set_column_data(3, lsr.compute_column_based_on_first(0.5))
    lsr.set_column_data(4, lsr.compute_column_based_on_first(0.3))
    lsr.run()

    # Spectra has to point to example_database folder before starting
    save_curve("{}".format("recreated.ssm"))
    print("Waiting for recreated file to be saved...")
    time.sleep(0.5)
    while not os.path.exists("example_database/{}".format("recreated.ssm")):
        time.sleep(1)

    print("\t Reading new HyperOCR data...")
    # Read HYperOCR (Current Curve)
    sensor_reading,_ = readAndCurateCurve("example_database/recreated.ssm")

    #sensor_reading = np.random.randint(65000, size=201)

    mse = (np.abs(scale_curve(desired_output) - scale_curve(sensor_reading['value'].values))).mean(axis=0)
    print("MSE= ",mse)
    print("Fitness: ", 1.0/mse)
    return 1.0/mse

def findLSRTenNumberRange(log_10_curve):
    device = "cpu"#torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = Predict10(curve_size=201)
    model.to(device)
    model.load_state_dict(torch.load("modeling/best_model.pth"))
    model.eval()

    solutions = []
    for i in range(0, 1000):
        sampl = np.random.uniform(low=-2, high=2, size=(201,))
        noisy = np.array(log_10_curve + sampl, dtype="float")
        predicted_ten_nums = model(torch.FloatTensor(noisy))
        predicted_ten_nums = [int(10 ** (item - 0.0001)) for item in predicted_ten_nums]
        solutions.append(predicted_ten_nums)
    return pd.DataFrame(solutions)

def findOptimalNumberForSplit(diff):
    print("Diff:",diff)
    if diff < 50:
        return 2
    if diff >= 50 and diff < 100:
        return 3
    else:
        return 5


def computeRange(ten_num_range_):
    m = ten_num_range_.mean()
    sd = ten_num_range_.std()
    n = len(ten_num_range_)
    Zstar = 1.96
    lcb = m - Zstar * sd
    ucb = m + Zstar * sd

    my_min = 1000
    my_max = -1
    total = []
    for i in range(0, 10):
        if lcb[i] < 0:
            lcb[i] = 0
        if ucb[i] > 200:
            ucb[i] = 200
        if lcb[i] < my_min:
            my_min = lcb[i]
        if ucb[i] > my_max:
            my_max = ucb[i]
        print(lcb[i], ucb[i])
        tmp_range = np.arange(int(lcb[i]), int(ucb[i]), findOptimalNumberForSplit(abs(int(lcb[i]) - int(ucb[i]))))
        print(tmp_range)
        total.append(tmp_range)
    return total, my_min, my_max

fitness_function = fitness_func

num_generations = 20
num_parents_mating = 4

sol_per_pop = 8
num_genes = 10

parent_selection_type = "sss"
keep_parents = 2

crossover_type = "single_point"

mutation_type = "random"
mutation_percent_genes = 20

day_time_range = [np.arange(0,300,20).tolist(),np.arange(0,350,20).tolist(), np.arange(0,250,20).tolist(),
                                   np.arange(0,400,20).tolist(), np.arange(0,300,20).tolist(), np.arange(0,500,20).tolist(),
                                   np.arange(0,200,20).tolist(), np.arange(0,200,20).tolist(), np.arange(0,350,20).tolist(),
                                   np.arange(0,400,20).tolist()]
morning_range = [np.arange(0,10,2).tolist(),np.arange(0,10,2).tolist(), np.arange(0,50,10).tolist(),
                                   np.arange(0,100,5).tolist(), np.arange(0,100,5).tolist(), np.arange(0,100,5).tolist(),
                                   np.arange(0,50,5).tolist(), np.arange(0,100,5).tolist(), np.arange(0,100,5).tolist(),
                                   np.arange(0,100,5).tolist()]

simulated_range = findLSRTenNumberRange(desired_log10_curve)

ten_num_range,init_range_low,init_range_high = computeRange(simulated_range)


ga_instance = pygad.GA(num_generations=num_generations,
                       num_parents_mating=num_parents_mating,
                       fitness_func=fitness_function,
                       sol_per_pop=sol_per_pop,
                       num_genes=num_genes,
                       gene_type=int,
                       gene_space=day_time_range,
                       init_range_low=init_range_low,
                       init_range_high=init_range_high,
                       parent_selection_type=parent_selection_type,
                       keep_parents=keep_parents,
                       crossover_type=crossover_type,
                       mutation_type=mutation_type,
                       mutation_percent_genes=mutation_percent_genes,
                       keep_elitism=1,
                       save_best_solutions=True,
                       stop_criteria=["reach_400"])

ga_instance.run()
print(ga_instance.best_solutions)
solution, solution_fitness, solution_idx = ga_instance.best_solution()
print("Parameters of the best solution : {solution}".format(solution=solution))
print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))

prediction = np.sum(np.array(function_inputs)*solution)
print("Predicted output based on the best solution : {prediction}".format(prediction=prediction))

lsr = LSR_comm("COM3")
time.sleep(0.5)
# Start LSR with params
lsr.set_column_data(1, solution)
lsr.set_column_data(2, lsr.compute_column_based_on_first(0.7))
lsr.set_column_data(3, lsr.compute_column_based_on_first(0.5))
lsr.set_column_data(4, lsr.compute_column_based_on_first(0.3))
lsr.run()

save_curve("{}".format("recreated.ssm"))
print("Waiting for recreated file to be saved...")
time.sleep(0.5)
while not os.path.exists("example_database/{}".format("recreated.ssm")):
    time.sleep(1)

print("\t Reading new HyperOCR data...")
# Read HYperOCR (Current Curve)
sensor_reading = readAndCurateCurve("example_database/recreated.ssm")
# Compare the two curves
plt.plot(sensor_reading['nm'].values, sensor_reading['value'].values)
plt.plot(desired_output_all['nm'].values, desired_output_all['value'].values)
plt.legend(['Recreated by {} gen'.format(num_generations), file_name])
plt.title("The best solution(MSE)")
plt.show()



# Go through top 3 solutions
cnt = 1
for s in ga_instance.best_solutions:
    lsr.set_column_data(1, s)
    lsr.set_column_data(2, lsr.compute_column_based_on_first(0.7))
    lsr.set_column_data(3, lsr.compute_column_based_on_first(0.5))
    lsr.set_column_data(4, lsr.compute_column_based_on_first(0.3))
    lsr.run()

    save_curve("{}".format("recreated.ssm"))
    print("Waiting for recreated file to be saved...")
    time.sleep(0.5)
    while not os.path.exists("example_database/{}".format("recreated.ssm")):
        time.sleep(1)

    print("\t Reading new HyperOCR data...")
    # Read HYperOCR (Curreynt Curve)
    sensor_reading = readAndCurateCurve("example_database/recreated.ssm")
    # Compare the two curves
    plt.plot(sensor_reading['nm'].values, sensor_reading['value'].values)
    plt.plot(desired_output_all['nm'].values, desired_output_all['value'].values)
    plt.legend(['Recreated by {} gen'.format(num_generations), file_name])
    plt.title("Alternative Solution {}".format(cnt))
    plt.show()
    cnt = cnt + 1
    if cnt > 3:
        break






# wait for 100 sec
time.sleep(100)