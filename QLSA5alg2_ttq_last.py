import pandas as pd
from datetime import datetime
import numpy as np
from compute import compute_distance, softmax, generate_tsp, epsilon_greedy, double_bridge_kick_cy
from copy import deepcopy
import tsplib95
import multiprocessing
import plotly.graph_objects as go
import plotly.io as pio
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv, json, os
import time
from pathlib import Path
from plotly.subplots import make_subplots


class SimulatedAnnealing_TSP_Logging:
    def __init__(self, TestsFilePath, problem, initial_solution, temperature=1.0, cooling_rate=0.99, tempmin=0.01,
                 epsilon=0.1, alpha=0.1, gamma=0.9, des=0.001, gamma1=0.9, rp=0.4,eps=0.05,best_known=0):
        self.problem = tsplib95.load(TestsFilePath + problem + '.tsp')
        self.has_node_coords = (self.problem.node_coords != {} or self.problem.display_data != {})
        self.solution = deepcopy(initial_solution)
        self.gbest = deepcopy(initial_solution)
        self.Fbest = compute_distance(initial_solution, self.problem)
        self.iter_log = []   # list of dicts
        self.log_enabled = True

        #self.pbest = deepcopy(initial_solution)
        #self.Fpbest = compute_distance(initial_solution, self.problem)
        ###print ('Fbest',self.Fbest)
        self.reached = False
        self.ittq = 0
        self.tttq = 0
        self.target_quality = int(round((1.0 + eps) * best_known))
        self.state = 0
        self.stateAlgo = 0
        self.nextstate = 0
        self.temperature = self.Fbest / 2
        self.temperature_max = self.Fbest / 2
        # self.temperature = temperature
        # self.temperature_max = temperature
        self.cooling_rate = cooling_rate
        self.tempmin = tempmin
        self.fitness_history = []
        random_sol = generate_tsp(1, len(self.solution), self.has_node_coords)[0]
        double_bridge_sol = double_bridge_kick_cy(np.array(self.solution, dtype=np.int32)).tolist()
        self.setcandidat = [self.solution, self.gbest, random_sol, double_bridge_sol]  # ,random_sol
        # self.setcandidat=[self.solution,self.gbest,self.pbest,random_sol]
        self.selection_percent={0:0,1:0,2:0,3:0}
        self.q_table = np.zeros((2, (len(self.setcandidat))))
        self.leader_count = np.zeros((2, (len(self.setcandidat))), dtype=int)
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.gamma1 = gamma1
        self.r = rp
        self.rp = rp
        self.fitness_evolution = []
        self.temperature_evolution = []
        self.des = des
        self.nbrville = len(self.solution)
        self.i=0

    def _log_iter_nostate(self, leader_idx, q_values, probs, reward):
        """
        Iteration-level logging for QLSA WITHOUT states.
        Frozen schema – do not modify during experiments.
        """

        # Q-dispersion metrics
        q_sorted = np.sort(q_values)
        q_gap = float(q_sorted[-1] - q_sorted[0])
        q_margin = float(q_sorted[-1] - q_sorted[-2]) if len(q_sorted) >= 2 else 0.0

        # Policy entropy
        entropy = float(-(probs * np.log(probs + 1e-12)).sum())

        log = {
            "iter": int(self.i),
            "leader": int(leader_idx),
            "reward": float(reward),
            "entropy": entropy,
            "p_selected": float(probs[leader_idx]),
            "q_gap": q_gap,
            "q_margin": q_margin,
            #"iter_time_sec": float(iter_time_sec),
        }

        self.iter_log.append(log)

    def _log_iter_state(self, leader_idx, q_values, probs, reward):
            """
            Iteration-level logging for QLSA WITH states.
            Frozen schema – do not modify during experiments.
            """

            # Q-dispersion for active state
            q_sorted = np.sort(q_values)
            q_gap = float(q_sorted[-1] - q_sorted[0])
            q_margin = float(q_sorted[-1] - q_sorted[-2]) if len(q_sorted) >= 2 else 0.0

            # Policy entropy
            entropy = float(-(probs * np.log(probs + 1e-12)).sum())

            # Per-state Q-gaps (global diagnostic)
            q_gap_state0 = float(np.max(self.q_table[0]) - np.min(self.q_table[0]))
            q_gap_state1 = float(np.max(self.q_table[1]) - np.min(self.q_table[1]))

            log = {
                "iter": int(self.i),
                "state": int(self.state),
                "nextstate": int(self.nextstate),
                "leader": int(leader_idx),
                "reward": float(reward),
                "entropy": entropy,
                "p_selected": float(probs[leader_idx]),
                "q_gap": q_gap,
                "q_margin": q_margin,
                "q_gap_state0": q_gap_state0,
                "q_gap_state1": q_gap_state1,
                #"iter_time_sec": float(iter_time_sec),
            }

            self.iter_log.append(log)



    def update_setcandidat(self):
        random_sol = generate_tsp(1, len(self.solution), self.has_node_coords)[0]
        double_bridge_sol = double_bridge_kick_cy(np.array(self.solution, dtype=np.int32)).tolist()
        self.setcandidat = [self.solution, self.gbest, random_sol, double_bridge_sol]  # ,random_sol
        # self.setcandidat=[self.solution,self.gbest,self.pbest,random_sol]#,random_sol

    def reset_q_table(self, nbr_states):
        self.q_table = np.zeros((nbr_states, (len(self.setcandidat))))
        self.update_setcandidat()

    def two_opt(self, solution, leader):  # x une solution x=[(ci1,ci2),(),...]
        x = deepcopy(solution)
        h = max(1, self.Hamming_dist(solution, self.gbest))
        ##print('ham',h)
        v = max(1, np.random.choice(range(h)))
        # print('v',v, )
        # print('self.solution',solution==self.gbest)
        ##print('self.gbest',self.gbest)
        for k in range(v):
            #   #print('v,k',v,k)
            # p=np.random.choice(range(len(x)))
            p = 1
            for i in range(p, len(x) - 1):
                for j in range(i + 2, len(x) - 1):
                    delta = (self.problem.get_weight(x[i], x[j]) + self.problem.get_weight(x[i + 1], x[j + 1])) - (
                            self.problem.get_weight(x[i], x[i + 1]) + self.problem.get_weight(x[j], x[j + 1]))
                    rnd = np.random.random_sample()
                    if delta < 0:
                        # print('2_opt_delta<0')
                        x[i + 1:j + 1] = reversed(x[i + 1:j + 1])
                    elif rnd < np.exp(-delta / (self.temperature / 50)):
                        # print('2_opt_metropolis')
                        x[i + 1:j + 1] = reversed(x[i + 1:j + 1])
        #   if compute_distance(x,self.problem) != compute_distance(leader,self.problem):
        #       break
        return x, compute_distance(x, self.problem)

    def Hamming_dist(self, x1, x2):
        n = 0
        for i, j in zip(x1, x2):
            if i != j:
                n = n + 1
        return n

    def two_opt_metropolis(self, leader):  # x une solution x=[(ci1,ci2),(),...]
        x = deepcopy(leader)
        h = max(1, self.Hamming_dist(self.solution, self.gbest))
        ###print('ham',h)
        v = max(1, np.random.choice(range(h)))
        ##print('v',v, )
        # print('self.solution',self.solution==self.gbest)
        # ##print('self.gbest',self.gbest)
        for k in range(v):
            # print('v,k',v,k)
            # p=np.random.choice(range(len(x)))
            p = 1
            for i in range(p, len(x) - 1):
                for j in range(i + 2, len(x) - 1):
                    delta = (self.problem.get_weight(x[i], x[j]) + self.problem.get_weight(x[i + 1], x[j + 1])) - (
                            self.problem.get_weight(x[i], x[i + 1]) + self.problem.get_weight(x[j], x[j + 1]))
                    rnd = np.random.random_sample()
                    if delta < 0:
                        # print('2_opt_delta<0')
                        x[i + 1:j + 1] = reversed(x[i + 1:j + 1])
                    elif rnd < np.exp(-delta / (self.temperature / 50)):
                        # print('2_opt_metropolis')
                        x[i + 1:j + 1] = reversed(x[i + 1:j + 1])
        #   if compute_distance(x,self.problem) != compute_distance(leader,self.problem):
        #       break
        return x  # ,compute_distance(x,self.problem)

    def select_leader(self):
        candidates = [j for j in range(len(self.setcandidat))]
        q_values = np.array([self.q_table[0][j] for j in candidates])
        probs = softmax(q_values, self.temperature)
        leader = np.random.choice(candidates, p=probs)
        # leader= epsilon_greedy(q_values,self.epsilon)
        self.leader_count[0][leader] += 1
        return leader, q_values, probs

    def select_leader_epsilon(self):
        candidates = [j for j in range(len(self.setcandidat))]
        q_values = np.array([self.q_table[0][j] for j in candidates])
        self.epsilon = self.epsilon * (1 - self.des)

        leader = epsilon_greedy(q_values, self.epsilon)
        self.leader_count[0][leader] += 1
        return leader

    def select_leader_states(self):
        candidates = [j for j in range(len(self.setcandidat))]
        q_values = np.array([self.q_table[self.state][j] for j in candidates])
        probs = softmax(q_values, self.temperature)
        leader = np.random.choice(candidates, p=probs)
        # leader= epsilon_greedy(q_values,self.epsilon)
        self.leader_count[self.state][leader] += 1
        return leader, q_values, probs

    def select_leader_epsilon_states(self):
        candidates = [j for j in range(len(self.setcandidat))]
        q_values = np.array([self.q_table[self.state][j] for j in candidates])
        self.epsilon = self.epsilon * (1 - self.des)

        leader = epsilon_greedy(q_values, self.epsilon)
        self.leader_count[self.state][leader] += 1
        #print(self.state)
        return leader

    def update_q_table(self, i, leader_idx, reward):
        max_future_q = np.max(self.q_table[self.nextstate])
        self.q_table[self.state][leader_idx] += self.alpha * (
                    reward + self.gamma * max_future_q - self.q_table[self.state][leader_idx])
        self.state = self.nextstate

    def update_q_table2(self, i, leader_idx, reward):
        # max_future_q = np.max(self.q_table[i])
        self.q_table[i][leader_idx] += self.alpha * (reward - self.q_table[i][leader_idx])

    def step_SA(self):
        candidate = self.two_opt_metropolis(self.solution)
        current_score = compute_distance(self.solution, self.problem)
        candidate_score = compute_distance(candidate, self.problem)
        delta = candidate_score - current_score

        if delta < 0 or np.random.rand() < np.exp(-delta / self.temperature):
            self.solution = candidate
            if candidate_score < self.Fbest:
                self.gbest = deepcopy(candidate)
                self.Fbest = candidate_score

        self.fitness_history.append(self.Fbest)
        self.fitness_evolution.append(current_score)
        self.temperature_evolution.append(self.temperature)

    def best(self, X, F):  # X:population de solution and F is the fitness list
        b = min(F)
        n = F.index(b)
        s = X[n]
        return b, s

    def step_uniform(self):
        leader_idx = np.random.choice(range(len(self.setcandidat)))
        self.leader_count[0][leader_idx] += 1
        leader = self.setcandidat[leader_idx]
        ##print('self.setcandidat,leader_idx,leader',self.setcandidat,leader_idx,leader)
        candidate = self.two_opt_metropolis(leader)
        current_score = compute_distance(self.solution, self.problem)
        candidate_score = compute_distance(candidate, self.problem)
        delta = candidate_score - current_score

        if delta < 0 or np.random.rand() < np.exp(-delta / self.temperature):
            self.solution = candidate
            if candidate_score < self.Fbest:
                self.gbest = deepcopy(candidate)
                self.Fbest = candidate_score
            # if candidate_score < self.Fpbest:
            #     self.pbest = deepcopy(candidate)
            #     self.Fpbest = candidate_score

        self.fitness_history.append(self.Fbest)
        self.fitness_evolution.append(current_score)
        self.temperature_evolution.append(self.temperature)
        # self.update_setcandidat()

    def step2(self):
        old_score = compute_distance(self.solution, self.problem)
        leader_idx, q_values, probs = self.select_leader()
        leader = self.setcandidat[leader_idx]
        ##print('self.setcandidat,leader_idx,leader',self.setcandidat,leader_idx,leader)
        candidate = self.two_opt_metropolis(leader)
        current_score = compute_distance(self.solution, self.problem)
        candidate_score = compute_distance(candidate, self.problem)
        delta = candidate_score - current_score

        if delta < 0 or np.random.rand() < np.exp(-delta / self.temperature):
            self.solution = candidate
            if candidate_score < self.Fbest:
                self.gbest = deepcopy(candidate)
                self.Fbest = candidate_score
            # if candidate_score < self.Fpbest:
            #     self.pbest = deepcopy(candidate)
            #     self.Fpbest = candidate_score

        self.fitness_history.append(self.Fbest)
        self.fitness_evolution.append(current_score)
        self.temperature_evolution.append(self.temperature)

        reward = (old_score - candidate_score) / old_score
        self.update_q_table2(0, leader_idx, reward)
        self._log_iter_nostate( leader_idx, q_values, probs, reward)
        # self.update_setcandidat()

    def step_greedy2(self):
        old_score = compute_distance(self.solution, self.problem)
        leader_idx = self.select_leader_epsilon()
        leader = self.setcandidat[leader_idx]
        candidate = self.two_opt_metropolis(leader)
        current_score = compute_distance(self.solution, self.problem)
        candidate_score = compute_distance(candidate, self.problem)
        delta = candidate_score - current_score

        if delta < 0 or np.random.rand() < np.exp(-delta / self.temperature):
            self.solution = candidate
            if candidate_score < self.Fbest:
                self.gbest = deepcopy(candidate)
                self.Fbest = candidate_score
            # if candidate_score < self.Fpbest:
            #     self.pbest = deepcopy(candidate)
            #     self.Fpbest = candidate_score

        self.fitness_history.append(self.Fbest)
        self.fitness_evolution.append(current_score)
        self.temperature_evolution.append(self.temperature)

        reward = (old_score - candidate_score) / old_score
        self.update_q_table2(0, leader_idx, reward)
        # self.update_setcandidat()

    def step2_state(self):
        old_score = compute_distance(self.solution, self.problem)
        leader_idx, q_values, probs = self.select_leader_states()
        leader = self.setcandidat[leader_idx]
        ##print('self.setcandidat,leader_idx,leader',self.setcandidat,leader_idx,leader)
        candidate = self.two_opt_metropolis(leader)
        current_score = compute_distance(self.solution, self.problem)
        candidate_score = compute_distance(candidate, self.problem)
        delta = candidate_score - current_score
        diff = self.Hamming_dist(candidate, self.gbest)
        if diff < self.nbrville / 2:
            self.nextstate = 0
        else:
            self.nextstate = 1

        if delta < 0 or np.random.rand() < np.exp(-delta / self.temperature):
            self.solution = candidate
            if candidate_score < self.Fbest:
                self.gbest = deepcopy(candidate)
                self.Fbest = candidate_score
            # if candidate_score < self.Fpbest:
            #     self.pbest = deepcopy(candidate)
            #     self.Fpbest = candidate_score

        self.fitness_history.append(self.Fbest)
        self.fitness_evolution.append(current_score)
        self.temperature_evolution.append(self.temperature)

        reward = (old_score - candidate_score) / old_score
        self.update_q_table(self.state, leader_idx, reward)
        self._log_iter_state( leader_idx, q_values, probs, reward)
        # self.update_setcandidat()

    def step_greedy2_state(self):
        old_score = compute_distance(self.solution, self.problem)
        leader_idx = self.select_leader_epsilon_states()
        leader = self.setcandidat[leader_idx]
        candidate = self.two_opt_metropolis(leader)
        current_score = compute_distance(self.solution, self.problem)
        candidate_score = compute_distance(candidate, self.problem)
        delta = candidate_score - current_score
        diff = self.Hamming_dist(candidate, self.gbest)
        if diff < self.nbrville / 2:
            self.nextstate = 0
        else:
            self.nextstate = 1

        if delta < 0 or np.random.rand() < np.exp(-delta / self.temperature):
            self.solution = candidate
            if candidate_score < self.Fbest:
                self.gbest = deepcopy(candidate)
                self.Fbest = candidate_score
            # if candidate_score < self.Fpbest:
            #     self.pbest = deepcopy(candidate)
            #     self.Fpbest = candidate_score

        self.fitness_history.append(self.Fbest)
        self.fitness_evolution.append(current_score)
        self.temperature_evolution.append(self.temperature)

        reward = (old_score - candidate_score) / old_score
        self.update_q_table(self.state, leader_idx, reward)
        #print(self.leader_count)
        # self.update_setcandidat()


    def run_greedy2(self, iterations=500, episodes=50, stateAlgo=0):
        # print('sa greedy')
        start_time = time.perf_counter()  # Start timer
        for i in range(iterations):
            self.temperature = (self.temperature_max - ((self.temperature_max - self.tempmin) * ((i + 1))) / iterations)
            if stateAlgo == 0:
                self.step_greedy2()
                nbr_states = 1
            else:
                self.step_greedy2_state()
                nbr_states = 2
            if (i + 1) % episodes == 0:
                self.reset_q_table(nbr_states)
            if (self.target_quality is not None) and (not self.reached) and (self.Fbest <= self.target_quality):
                self.reached = True
                self.ittq = i
                self.tttq = time.perf_counter() - start_time
            # print(f"Iteration {i}, Temp: {self.temperature:.4f}, Best: {self.Fbest:.2f}")
        end_time = time.perf_counter()  # End timer
        execution_time = end_time - start_time
        return self.gbest, self.Fbest, self.fitness_history, self.fitness_evolution, self.temperature_evolution, execution_time,self.reached,self.ittq,self.tttq,self.leader_count,self.iter_log

    def run_SA(self, iterations=500):
        # print('sa ')
        start_time = time.perf_counter()  # Start timer
        for i in range(iterations):
            self.temperature = (self.temperature_max - ((self.temperature_max - self.tempmin) * ((i + 1))) / iterations)
            self.step_SA()
            if (self.target_quality is not None) and (not self.reached) and (self.Fbest <= self.target_quality):
                self.reached = True
                self.ittq = i
                self.tttq = time.perf_counter() - start_time
            ###print(f"Iteration {i}, Temp: {self.temperature:.4f}, Best: {self.Fbest:.2f}")
        end_time = time.perf_counter()  # End timer
        execution_time = end_time - start_time
        return self.gbest, self.Fbest, self.fitness_history, self.fitness_evolution, self.temperature_evolution, execution_time,self.reached,self.ittq,self.tttq,self.leader_count,self.iter_log

    def run2(self, iterations=500, episodes=50, stateAlgo=0):
        # print('sa softmax')
        start_time = time.perf_counter()  # Start timer
        for i in range(iterations):
            self.i=i
            self.temperature = (self.temperature_max - ((self.temperature_max - self.tempmin) * ((i + 1))) / iterations)
            if stateAlgo == 0:
                self.step2()
                nbr_states = 1
            else:
                self.step2_state()
                nbr_states = 2
            if (i + 1) % episodes == 0:
                self.reset_q_table(nbr_states)

            if (self.target_quality is not None) and (not self.reached) and (self.Fbest <= self.target_quality):
                self.reached = True
                self.ittq = i
                self.tttq = time.perf_counter() - start_time
            ##print(f"Iteration {i}, Temp: {self.temperature:.4f}, Best: {self.Fbest:.2f}")
        end_time = time.perf_counter()  # End timer
        execution_time = end_time - start_time
        return self.gbest, self.Fbest, self.fitness_history, self.fitness_evolution, self.temperature_evolution, execution_time,self.reached,self.ittq,self.tttq,self.leader_count,self.iter_log

    def run_uniform(self, iterations=500, episodes=50):
        # print('sa softmax')
        start_time = time.perf_counter()  # Start timer
        for i in range(iterations):
            self.temperature = (self.temperature_max - ((self.temperature_max - self.tempmin) * ((i + 1))) / iterations)
            self.step_uniform()
            if (self.target_quality is not None) and (not self.reached) and (self.Fbest <= self.target_quality):
                self.reached = True
                self.ittq = i
                self.tttq = time.perf_counter() - start_time

        end_time = time.perf_counter()  # End timer
        execution_time = end_time - start_time
        return self.gbest, self.Fbest, self.fitness_history, self.fitness_evolution, self.temperature_evolution, execution_time,self.reached,self.ittq,self.tttq,self.leader_count,self.iter_log

    def run_greedy2_sans_reset(self, iterations=500, episodes=50, stateAlgo=0):
        # print('sa greedy')
        start_time = time.perf_counter()  # Start timer
        for i in range(iterations):
            self.i=i
            self.temperature = (self.temperature_max - ((self.temperature_max - self.tempmin) * ((i + 1))) / iterations)
            if stateAlgo == 0:
                self.step_greedy2()
            else:
                self.step_greedy2_state()
            # if(i+1) % episodes==0:
            # self.reset_q_table(1)
            self.update_setcandidat()
            if (self.target_quality is not None) and (not self.reached) and (self.Fbest <= self.target_quality):
                self.reached = True
                self.ittq = i
                self.tttq = time.perf_counter() - start_time
            # print(f"Iteration {i}, Temp: {self.temperature:.4f}, Best: {self.Fbest:.2f}")
        end_time = time.perf_counter()  # End timer
        execution_time = end_time - start_time
        return self.gbest, self.Fbest, self.fitness_history, self.fitness_evolution, self.temperature_evolution, execution_time,self.reached,self.ittq,self.tttq,self.leader_count,self.iter_log

    def run2_sans_reset(self, iterations=500, episodes=50, stateAlgo=0):
        # print('sa softmax')
        start_time = time.perf_counter()  # Start timer
        for i in range(iterations):
            self.i=i
            self.temperature = (self.temperature_max - ((self.temperature_max - self.tempmin) * ((i + 1))) / iterations)
            if stateAlgo == 0:
                self.step2()
            else:
                self.step2_state()
            # if(i+1) % episodes==0:
            # self.reset_q_table(1)
            self.update_setcandidat()
            if (self.target_quality is not None) and (not self.reached) and (self.Fbest <= self.target_quality):
                self.reached = True
                self.ittq = i
                self.tttq = time.perf_counter() - start_time
            ##print(f"Iteration {i}, Temp: {self.temperature:.4f}, Best: {self.Fbest:.2f}")
        end_time = time.perf_counter()  # End timer
        execution_time = end_time - start_time
        return self.gbest, self.Fbest, self.fitness_history, self.fitness_evolution, self.temperature_evolution, execution_time,self.reached,self.ittq,self.tttq,self.leader_count,self.iter_log

    def run_uniform_sans_reset(self, iterations=500, episodes=50):
        # print('sa softmax')
        start_time = time.perf_counter()  # Start timer
        for i in range(iterations):
            self.temperature = (self.temperature_max - ((self.temperature_max - self.tempmin) * ((i + 1))) / iterations)
            self.step_uniform()
            # if(i+1) % episodes==0:
            # self.reset_q_table(1)
            self.update_setcandidat()
            if (self.target_quality is not None) and (not self.reached) and (self.Fbest <= self.target_quality):
                self.reached = True
                self.ittq = i
                self.tttq = time.perf_counter() - start_time
        end_time = time.perf_counter()  # End timer
        execution_time = end_time - start_time
        return self.gbest, self.Fbest, self.fitness_history, self.fitness_evolution, self.temperature_evolution, execution_time,self.reached,self.ittq,self.tttq,self.leader_count ,self.iter_log


ALGO_NAMES = {
    1: "SA",
    2: "QL-SA_softmax",
    3:"Greedy",
    4: "QL-SA_softmax_state",
    5: "Greedy_state",
    6: "Uniform",
}


def _jsonify(obj):
    try:
        import numpy as np
    except ImportError:
        np = None
    if isinstance(obj, dict):
        return {k: _jsonify(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonify(x) for x in obj]
    if np is not None and isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if np is not None and isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, set):
        return sorted(list(obj))
    return obj


def save_result_csv_full(out_dir, problem_name, algo_id, run_id, iter_used, res):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    algo_name = ALGO_NAMES.get(algo_id, f"algo-{algo_id}")
    fname = f"{problem_name}_algo-{algo_name}_{run_id}.csv"
    fpath = Path(out_dir) / fname

    gbest, Fbest, fitness_history, fitness_evolution, temperature_evolution, execution_time,reached,ittq,tttq,leader_count,iter_log = res
    gbest_j = json.dumps(_jsonify(gbest), ensure_ascii=False)
    fhist_j = json.dumps(_jsonify(fitness_history), ensure_ascii=False)
    fevol_j = json.dumps(_jsonify(fitness_evolution), ensure_ascii=False)
    tevol_j = json.dumps(_jsonify(temperature_evolution), ensure_ascii=False)
    tttq_j = json.dumps(_jsonify([reached,ittq,tttq]), ensure_ascii=False)
    leader_count_j = json.dumps(_jsonify(leader_count), ensure_ascii=False)
    iter_log_j = json.dumps(_jsonify(iter_log), ensure_ascii=False)

    write_header = not fpath.exists()
    with open(fpath, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f,
        delimiter=";",        # <-- semicolon separator
        quoting=csv.QUOTE_MINIMAL)
        if write_header:
            w.writerow([
                "timestamp", "problem", "algorithm_id", "algorithm_name", "run_id", "iterations",
                "gbest", "Fbest", "fitness_history", "fitness_evolution", "temperature_evolution", "execution_time", "tttq_j","leader_count_j","iter_log_j"
            ])
        w.writerow([
            datetime.now().isoformat(timespec="seconds"),
            problem_name, algo_id, algo_name, run_id, iter_used,
            gbest_j, Fbest, fhist_j, fevol_j, tevol_j, execution_time, tttq_j,leader_count_j,iter_log_j
        ])
    return str(fpath)


runs = 10
TestsFilePath = 'inputs/'


def runAlgo(params):
    # ORDER changed to align with your DF_results_parallel unpack:
    # (TestsFilePath, problem_name, initial_solution, param, run_id, Iter, episodes, epsilon, alpha, gamma, des, tempmin)
    (TestsFilePath, problem_name, initial_solution, param, run_id, Iter, episodes,
     epsilon, des, tempmin,best_known) = params

    temperature = 1000.0
    cooling_rate = 0.99
    gamma1 = 0.9
    rp = 0.4
    alpha1 = 0.3
    alpha2 = 0.6
    gamma_1 = 1
    gamma_2 = 0.8
    sa_obj1 = SimulatedAnnealing_TSP_Logging(
        TestsFilePath, problem_name, initial_solution,
        temperature, cooling_rate, tempmin,
        epsilon, alpha1, gamma_1, des, gamma1, rp,best_known=best_known
    )
    sa_obj2 = SimulatedAnnealing_TSP_Logging(
        TestsFilePath, problem_name, initial_solution,
        temperature, cooling_rate, tempmin,
        epsilon, alpha2, gamma_2, des, gamma1, rp, best_known=best_known
    )

    Iter = 1000  # 1000#sa_obj.nbrville * 500
    print('problem_name,Iter', problem_name, Iter)
    # episodes = int(Iter * 0.1)
    episodes = 100

    # if param == 1:
    #     res = sa_obj.run2(iterations=Iter, episodes=episodes)
    if param == 1:
        res = sa_obj1.run_SA(iterations=Iter)
    # elif param == 3:
    #     res = sa_obj.run_greedy2(iterations=Iter, episodes=episodes)
    # elif param == 4:
    #     res = sa_obj.run_uniform(iterations=Iter, episodes=episodes)
    elif param == 2:
        res = sa_obj1.run2_sans_reset(iterations=Iter, episodes=episodes)
    elif param == 3:
        res = sa_obj1.run_greedy2_sans_reset(iterations=Iter, episodes=episodes)
    # elif param == 7:
    #     res = sa_obj.run2(iterations=Iter, episodes=episodes, stateAlgo=1)
    # elif param == 8:
    #     res = sa_obj.run_greedy2(iterations=Iter, episodes=episodes, stateAlgo=1)
    elif param == 4:
        res = sa_obj2.run2_sans_reset(iterations=Iter, episodes=episodes, stateAlgo=1)
    elif param == 5:
        res = sa_obj2.run_greedy2_sans_reset(iterations=Iter, episodes=episodes, stateAlgo=1)
    #elif param == 6:
    #    res = sa_obj.run_uniform_sans_reset(iterations=Iter, episodes=episodes)
    else:
        raise ValueError(f"Invalid param value: {param}")

    # Save full results as-is (CSV), but still return the tuple unchanged
    base_dir = "./New_Results_30_01_2026_analysis/"
    out_dir = os.path.join(base_dir, "resultsanalysis")
    save_result_csv_full(out_dir, problem_name, param, run_id, Iter, res)

    return res


# ===============================
# Task Preparation
# ===============================
def prepare_tasks(ListProb, TestsFilePath, runs, Iter, episodes, epsilon, des, tempmin,best_known):
    tasks = []
    for PROB in ListProb:
        problem = tsplib95.load(TestsFilePath + PROB + '.tsp')
        nbrville = problem.dimension
        best_known_prob=best_known[PROB]
        has_node_coords = (problem.node_coords != {} or problem.display_data != {})
        for k in range(runs):
            initial_solution = generate_tsp(1, nbrville, has_node_coords)[0]
            for p in range(1, 6):
                print('instance', PROB, 'run', k, 'algo', p)
                tasks.append((
                    TestsFilePath, PROB, initial_solution, p, k,  # <-- run_id k is 5th element
                    Iter, episodes, epsilon, des, tempmin,best_known_prob
                ))
    return tasks


# ===============================
# Parallel Execution
# ===============================
def parallel_run(tasks, max_workers=None):
    if max_workers is None:
        max_workers =  int(os.getenv("nb_proc", 10))
    results = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_task = {executor.submit(runAlgo, t): t for t in tasks}
        for future in as_completed(future_to_task):
            task = future_to_task[future]
            try:
                zres = future.result()  # still the 6-tuple
                results.append((task, zres))  # <-- preserves your DF_results_parallel contract
            except Exception as e:
                print(f"Task {task} failed: {e}")
    return results


# ===============================
# FIXED SEED FOR REPRODUCIBILITY
# ===============================


def _load_tsplib_coords(tsp_path):
    coords = {}
    with open(tsp_path, "r", encoding="utf-8") as f:
        lines = f.read().strip().splitlines()
    start = None
    for i, line in enumerate(lines):
        if line.strip().upper().startswith("NODE_COORD_SECTION"):
            start = i + 1
            break
    if start is None:
        raise ValueError("NODE_COORD_SECTION not found")
    for line in lines[start:]:
        s = line.strip()
        if s == "" or s.upper().startswith("EOF"):
            break
        p = s.split()
        if len(p) >= 3:
            coords[int(p[0])] = (float(p[1]), float(p[2]))
    # Return as array ordered by node id (0-based indexing for your gbest)
    ordered = [coords[k] for k in sorted(coords)]
    return np.array(ordered, dtype=float)


def _load_coords(prob, tests_path):
    tsp = Path(tests_path) / f"{prob}.tsp"
    csv = Path(tests_path) / f"{prob}.csv"
    if tsp.exists():
        return _load_tsplib_coords(str(tsp))
    if csv.exists():
        arr = np.loadtxt(str(csv), delimiter=",", dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        assert arr.shape[1] >= 2, "CSV must have at least two columns: x,y"
        return arr[:, :2]
    raise FileNotFoundError(f"No coords file found for {prob} in {tests_path}")


def DF_results_parallel(ListProb, TestsFilePath, runs, best_known):
    # Hyperparameters
    Iter, episodes = 1000, 100
    # Iter, episodes = 300000, 100
    epsilon, gamma_1, gamma_2, alpha1, alpha2, des, tempmin = 1, 1, 0.8, 0.3, 0.6, 0.001, 0.001
    date = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Algorithm names
    MM = ['SA', 'QLSA_s_without_reset', 'QLSA_e_without_reset', 'QLSA_s_state_sans_reset',
          'QLSA_e_state_sans_reset']  # Internal names for CSVs
    pretty_names = {'SA': 'SA',
                    'QLSA_s_without_reset': 'QLSA_s', 'QLSA_e_without_reset': 'QLSA_ε',
                    'QLSA_s_state_sans_reset': 'SB-QLSA_s',
                    'QLSA_e_state_sans_reset': 'SB-QLSA_ε'}  # For plots

    # Output dirs
    base_dir = f"./Last_results"
    os.makedirs(base_dir, exist_ok=True)
    plot_dir = f"{base_dir}/Plots"
    os.makedirs(plot_dir, exist_ok=True)
    gbest_dir = f"{base_dir}/gbest_runs"
    os.makedirs(gbest_dir, exist_ok=True)

    # Master runtime CSV (append)
    runtime_master_csv = f"{base_dir}/runtime_master_{date}.csv"

    # Save parameters for reproducibility
    params_data = {
        "Date": date,
        "Iterations": Iter,
        "Episodes": episodes,
        "Epsilon": epsilon,
        "Alpha1": alpha1,
        "Alpha2": alpha2,
        "Gamma1": gamma_1,
        "Gamma2": gamma_2,
        "Decay": des,
        "TempMin": tempmin,
        "Runs": runs,
        "Instances": ", ".join(ListProb)
    }
    pd.DataFrame([params_data]).to_csv(f"{base_dir}/parameters_{date}.csv", sep=";", index=False)

    all_results_collection = {}

    # Main loop over each problem instance
    for prob in ListProb:
        print(f"--- Running instance: {prob} ---")
        
        # Prepare tasks for THIS problem only
        tasks = prepare_tasks([prob], TestsFilePath, runs, Iter, episodes, epsilon,  des, tempmin,best_known)

        # Parallel execution for this problem
        results = parallel_run(tasks)

        # Storage for the current problem
        all_conv_data = {algo: [] for algo in MM}
        all_accepted_data = {algo: [] for algo in MM}
        all_results_df = pd.DataFrame({algo: pd.Series(dtype=float) for algo in MM})
        runtime_per_algo = {algo: [] for algo in MM}
        best_across = {algo: {"Fbest": float("inf"), "run": None, "gbest": None} for algo in MM}

        # Populate results for the current problem
        for task, zres in results:
            TestsFilePath, PROB, initial_solution, param, run_id, *_ = task
            algo_name = MM[param - 1]

            gbest_route = zres[0]
            run_Fbest = float(zres[1])
            conv_curve = zres[2]
            accepted_curve = zres[3]
            exec_time_s = float(zres[5])
            reached,ittq,tttq=zres[6:9]
            leader_count=zres[9:]

            # 1) Save per-run gbest (for later plotting/analysis)
            gbest_path = os.path.join(gbest_dir, f"{PROB}_{algo_name}_run{run_id}_gbest.txt")
            np.savetxt(gbest_path, np.asarray(gbest_route, dtype=int), fmt="%d")

            # 2) Append per-run runtime row into a master CSV
            row = {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "instance": PROB,
                "algorithm": algo_name,
                "run": run_id,
                "exec_time_s": exec_time_s,
                "Fbest": run_Fbest,
                "reached":reached,
                "ittq":ittq,
                "tttq":tttq,
                "leadercount":leader_count
            }
            mode = "a" if os.path.exists(runtime_master_csv) else "w"
            df_row = pd.DataFrame([row])
            df_row.to_csv(runtime_master_csv, mode=mode, header=not os.path.exists(runtime_master_csv), sep=";", index=False)

            # 3) Update per-instance DataFrame of Fbest
            current_df = all_results_df
            if algo_name not in current_df.columns:
                current_df[algo_name] = np.nan
            current_df.loc[run_id, algo_name] = run_Fbest # Use run_id as index
            all_results_df = current_df

            # 4) Store convergence & accepted solution curves (for mean plots)
            all_conv_data[algo_name].append(conv_curve)
            all_accepted_data[algo_name].append(accepted_curve)

            # 5) Store runtimes for stats
            runtime_per_algo[algo_name].append(exec_time_s)

            # 6) Track best across runs (per prob x algo)
            if run_Fbest < best_across[algo_name]["Fbest"]:
                best_across[algo_name] = {
                    "Fbest": run_Fbest,
                    "run": run_id,
                    "gbest": np.asarray(gbest_route, dtype=int)
                }

        all_results_collection[prob] = all_results_df
        
        # Save per-instance results, runtime stats & plots for the CURRENT problem
        # Save detailed Fbest runs
        file_path = f"{base_dir}/{prob}_runs_{date}.csv"
        all_results_df.to_csv(file_path, sep=";", index_label="run")

        # Save descriptive stats for Fbest
        desc_path = f"{base_dir}/{prob}_stats_{date}.csv"
        all_results_df.describe().to_csv(desc_path, sep=";")

        # Save runtime per algo
        runtime_df = pd.DataFrame({algo: pd.Series(runtime_per_algo[algo]) for algo in MM})
        runtime_runs_csv = f"{base_dir}/{prob}_runtime_runs_{date}.csv"
        runtime_df.to_csv(runtime_runs_csv, sep=";", index=False)

        runtime_stats_csv = f"{base_dir}/{prob}_runtime_stats_{date}.csv"
        runtime_df.describe().to_csv(runtime_stats_csv, sep=";")

        # Save "best across runs" metadata + gbest text
        best_meta_rows = []
        for algo in MM:
            b = best_across[algo]
            best_gbest_path = os.path.join(gbest_dir, f"{prob}_{algo}_BEST_run{b['run']}_gbest.txt")
            if b["gbest"] is not None:
                np.savetxt(best_gbest_path, b["gbest"], fmt="%d")
            best_meta_rows.append({
                "instance": prob,
                "algorithm": algo,
                "best_run": b["run"],
                "best_Fbest": b["Fbest"],
                "best_gbest_path": best_gbest_path if b["gbest"] is not None else ""
            })
        pd.DataFrame(best_meta_rows).to_csv(f"{base_dir}/{prob}_best_across_runs_{date}.csv", sep=";", index=False)

        # =======================
        # Convergence plot (mean)
        # =======================
        if all(len(all_conv_data[a]) > 0 for a in MM):
            min_len = min(min(len(c) for c in all_conv_data[algo]) for algo in MM if all_conv_data[algo])
            mean_conv = {algo: np.mean([np.asarray(c)[:min_len] for c in all_conv_data[algo]], axis=0) for algo in MM}
            iterations = list(range(min_len))

            fig1 = go.Figure()
            for algo in MM:
                fig1.add_trace(go.Scatter(x=iterations, y=mean_conv[algo], name=pretty_names[algo], mode='lines'))

            fig1.update_layout(
                title=f"Convergence Plot (Mean Best Cost Per Iteration) - {prob}",
                xaxis_title="Iteration",
                yaxis_title="Cost",
                template="plotly_white",
                legend=dict(x=0.01, y=0.99)
            )
            pio.write_html(fig1, file=f"{plot_dir}/{prob}_convergence_{date}.html", auto_open=False)
            pio.write_image(fig1, f"{plot_dir}/{prob}_convergence_{date}.png")

        # =======================
        # Accepted Fitness Plot (mean)
        # =======================
        if all(len(all_accepted_data[a]) > 0 for a in MM):
            min_len_acc = min(min(len(c) for c in all_accepted_data[algo]) for algo in MM if all_accepted_data[algo])
            mean_accepted = {algo: np.mean([np.asarray(c)[:min_len_acc] for c in all_accepted_data[algo]], axis=0) for algo in MM}
            iterations_acc = list(range(min_len_acc))

            fig2 = go.Figure()
            for algo in MM:
                fig2.add_trace(go.Scatter(x=iterations_acc, y=mean_accepted[algo], name=pretty_names[algo], mode='lines'))

            fig2.update_layout(
                title=f"Accepted Fitness Plot (Mean Accepted Solutions Per Iteration) - {prob}",
                xaxis_title="Iteration",
                yaxis_title="Cost",
                template="plotly_white",
                legend=dict(x=0.01, y=0.99)
            )
            pio.write_html(fig2, file=f"{plot_dir}/{prob}_accepted_fitness_{date}.html", auto_open=False)
            pio.write_image(fig2, f"{plot_dir}/{prob}_accepted_fitness_{date}.png")

        # =======================
        # FIGURE 3: Best Routes (gbest)
        # =======================
        try:
            coords = _load_coords(prob, TestsFilePath)
            fig3 = make_subplots(rows=1, cols=len(MM), subplot_titles=[pretty_names[a] for a in MM])

            for i_col in range(1, len(MM) + 1):
                fig3.update_xaxes(scaleanchor=f"y{i_col}", scaleratio=1, row=1, col=i_col)

            for col, algo in enumerate(MM, start=1):
                b = best_across[algo]
                route = b["gbest"]
                if route is None or len(route) == 0:
                    fig3.add_annotation(row=1, col=col, text="No route", showarrow=False)
                    continue
                
                route = np.asarray(route, dtype=int).flatten()
                loop = np.r_[route, route[0]]
                xs = coords[loop, 0]
                ys = coords[loop, 1]

                fig3.add_trace(go.Scatter(x=xs, y=ys, mode="lines", showlegend=False), row=1, col=col)
                fig3.add_trace(go.Scatter(x=coords[:, 0], y=coords[:, 1], mode="markers", marker=dict(size=6), showlegend=False), row=1, col=col)
                fig3.add_trace(go.Scatter(x=[coords[route[0], 0]], y=[coords[route[0], 1]], mode="markers+text", text=["start"], textposition="top center", marker=dict(size=9, symbol="star"), showlegend=False), row=1, col=col)

            fig3.update_layout(
                title=f"Best Routes (gbest) — {prob}",
                template="plotly_white",
                height=500,
                width=1200,
                margin=dict(l=30, r=30, t=60, b=30)
            )
            pio.write_html(fig3, file=f"{plot_dir}/{prob}_best_routes_{date}.html", auto_open=False)
            pio.write_image(fig3, f"{plot_dir}/{prob}_best_routes_{date}.png")

        except Exception as e:
            print(f"[WARN] Could not plot best routes for {prob}: {e}")

    return all_results_collection, plot_dir


# ===============================
# MAIN
# ===============================
if __name__ == "__main__":
    TestsFilePath = "inputs/"  # adjust path
    runs = 10
    ListProb = [ "ulysses16", "ulysses22", "bayg29", "bays29", "swiss42", "gr48", "hk48", "eil51" ]#,'st70','eil76','pr76','rat99','kroA100','eil101']  # ,'dantzig42','swiss42','gr48','hk48']  # add more instances
    #ListProb = ['hk48','berlin52','eil101','kroA100']#,'dantzig42','swiss42','gr48','hk48']  # add more instances
    #ListProb = ['st70','pr76','eil76','rat99']#,'kroA100','kroB100','kroC100','kroD100','kroE100','eil101','lin105','pr124','ch150','tsp225']  # add more instances
    #ListProb = ['kroA100']#,'kroA100']#,'kroB100','kroC100','kroD100','kroE100','eil101','lin105','pr124','ch150']#,'lin105','pr124','ch150','tsp225']
#     best_known={
#     "gr17": 2085,
#     "ulysses16": 6859,
#     "ulysses22": 7013,
#     "bayg29": 1610,
#     "bays29": 2020,
#     "dantzig42": 699
# }
#     best_known={
#         "gr17":2085,
#     "hk48": 11461,
#     "berlin52": 7542,
#     "eil101": 629,
#     "ulysses16": 6859,
#     "ulysses22": 7013,
#     "bayg29": 1610,
#     "gr24": 1272,
#     "bays29": 2020,
#     "dantzig42": 699,
#     "kroA100": 21282
# }
    best_known = {
    "gr17": 2085,
    "gr24": 1272,
    "ulysses16": 6859,
    "ulysses22": 7013,
    "bayg29": 1610,
    "bays29": 2020,
    "dantzig42": 699,
    "swiss42": 1273,
    "gr48": 5046,
    "hk48": 11461,
    "eil51": 426,
    "berlin52": 7542,
    "st70": 675,
    "eil76": 538,
    "pr76": 108159,
    "rat99": 1211,
    "kroA100": 21282,
    "eil101": 629
}

    # gr17,bayg29,bays29,oliver30,swiss42,eil51,berlin52,st70,pr76,eil76,rat99,kroA100,kroB100,kroC100,kroD100,kroE100,eil101,lin105,pr124,ch150,tsp225
    results, plots = DF_results_parallel(ListProb, TestsFilePath, runs,best_known)
    print("Saved detailed run results for each instance.")
    print("Plots saved in:", plots)