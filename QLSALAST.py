import pandas as pd
from datetime import datetime
import numpy as np
from compute import compute_distance, softmax,generate_tsp,epsilon_greedy
from copy import deepcopy
import tsplib95
import multiprocessing
import plotly.graph_objects as go
import plotly.io as pio
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
import time
from pathlib import Path
from plotly.subplots import make_subplots

class SimulatedAnnealing_TSP_Logging:
    def __init__(self,TestsFilePath, problem, initial_solution, temperature=1.0, cooling_rate=0.99, tempmin=0.01, epsilon=0.1, alpha=0.1, gamma=0.9,des=0.001,gamma1=0.9,rp=0.4):
        self.problem = tsplib95.load(TestsFilePath + problem + '.tsp')
        self.has_node_coords = (self.problem.node_coords != {} or self.problem.display_data != {})
        self.solution = deepcopy(initial_solution)
        self.gbest = deepcopy(initial_solution)
        self.Fbest = compute_distance(initial_solution, self.problem)
        self.pbest = deepcopy(initial_solution)
        self.Fpbest = compute_distance(initial_solution, self.problem)
        ###print ('Fbest',self.Fbest)
        self.temperature = self.Fbest/2
        self.temperature_max = self.Fbest/2
        # self.temperature = temperature
        # self.temperature_max = temperature
        self.cooling_rate = cooling_rate
        self.tempmin = tempmin
        self.fitness_history = []
        random_sol=generate_tsp(1,len(self.solution), self.has_node_coords)[0]
        self.setcandidat=[self.solution,self.gbest,self.pbest,random_sol]
        self.q_table = np.zeros((1, (len(self.setcandidat))))
        self.leader_count = np.zeros((1, (len(self.setcandidat))), dtype=int)
        self.epsilon=epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.gamma1=gamma1
        self.r=rp
        self.rp=rp
        self.fitness_evolution=[]
        self.temperature_evolution=[]
        self.des=des
        self.nbrville=len(self.solution)

    def update_setcandidat(self):
        random_sol=generate_tsp(1,len(self.solution), self.has_node_coords)[0]
        self.setcandidat=[self.solution,self.gbest,self.pbest,random_sol]#,random_sol

    def reset_q_table(self):
        self.q_table = np.zeros((1, (len(self.setcandidat))))
        self.update_setcandidat()

    def two_opt(self,solution,leader): #x une solution x=[(ci1,ci2),(),...]
        x=deepcopy(solution)
        h=max(1,self.Hamming_dist(solution,self.gbest))
        ##print('ham',h)
        v=max(1,np.random.choice(range(h))  ) 
        #print('v',v, )            
        #print('self.solution',solution==self.gbest)
        ##print('self.gbest',self.gbest)
        for k in range(v):
        #   #print('v,k',v,k)
          p=np.random.choice(range(len(x)))
          #p=1
          for i in range(p,len(x) - 1):
            for j in range(i + 2, len(x) - 1):
                delta=(self.problem.get_weight(x[i], x[j]) + self.problem.get_weight(x[i+1] ,x[j+1]))-(self.problem.get_weight(x[i],x[i+1]) + self.problem.get_weight(x[j],x[j+1]))
                rnd = np.random.random_sample()
                if delta<0:
                    #print('2_opt_delta<0')
                    x[i+1:j+1]= reversed(x[i+1:j+1])
                elif rnd<np.exp(-delta/(self.temperature/50)):
                    #print('2_opt_metropolis')
                    x[i+1:j+1]= reversed(x[i+1:j+1])
        #   if compute_distance(x,self.problem) != compute_distance(leader,self.problem):
        #       break
        return x,compute_distance(x,self.problem)


    def Hamming_dist(self,x1,x2):
        n=0
        for i,j in zip(x1,x2):
            if i!=j:
                n=n+1
        return n
    
    def two_opt_metropolis(self,leader): #x une solution x=[(ci1,ci2),(),...]
        x=deepcopy(leader)
        h=max(1,self.Hamming_dist(self.solution,self.gbest))
        ###print('ham',h)
        v=max(1,np.random.choice(range(h))  ) 
        ##print('v',v, )            
        #print('self.solution',self.solution==self.gbest)
        # ##print('self.gbest',self.gbest)
        for k in range(v):
          #print('v,k',v,k)
          p=np.random.choice(range(len(x)))
          #p=1
          for i in range(p,len(x) - 1):
            for j in range(i + 2, len(x) - 1):
                delta=(self.problem.get_weight(x[i], x[j]) + self.problem.get_weight(x[i+1] ,x[j+1]))-(self.problem.get_weight(x[i],x[i+1]) + self.problem.get_weight(x[j],x[j+1]))
                rnd = np.random.random_sample()
                if delta<0:
                    #print('2_opt_delta<0')
                    x[i+1:j+1]= reversed(x[i+1:j+1])
                elif rnd<np.exp(-delta/(self.temperature/50)):
                    #print('2_opt_metropolis')
                    x[i+1:j+1]= reversed(x[i+1:j+1])
        #   if compute_distance(x,self.problem) != compute_distance(leader,self.problem):
        #       break
        return x#,compute_distance(x,self.problem)
    
    def select_leader(self):
        candidates = [j for j in range(len(self.setcandidat))]
        q_values = np.array([self.q_table[0][j] for j in candidates])
        probs = softmax(q_values, self.temperature)
        leader = np.random.choice(candidates, p=probs)
        #leader= epsilon_greedy(q_values,self.epsilon)
        self.leader_count[0][leader] += 1
        return leader
    
    def select_leader_epsilon(self):
        candidates = [j for j in range(len(self.setcandidat))]
        q_values = np.array([self.q_table[0][j] for j in candidates])
        self.epsilon=self.epsilon*(1-self.des)

        leader= epsilon_greedy(q_values,self.epsilon)
        self.leader_count[0][leader] += 1
        return leader
    
    def update_q_table2(self, i, leader_idx, reward):
        #max_future_q = np.max(self.q_table[i])
        self.q_table[i][leader_idx] += self.alpha * (reward  - self.q_table[i][leader_idx])

    def update_q_table(self, i, leader_idx, reward):
        max_future_q = np.max(self.q_table[i])
        self.q_table[i][leader_idx] += self.alpha * (reward + self.gamma * max_future_q - self.q_table[i][leader_idx])

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

    

    def dataBat(self, NPop):
        v = [0]*NPop  #velocity
        X = generate_tsp(NPop,len(self.solution), self.has_node_coords)
        fitness=[compute_distance(X[i],self.problem) for i in range(NPop)]
        return X,v,fitness
    
    def best(self,X,F): #X:population de solution and F is the fitness list
        b=min(F)
        n=F.index(b)
        s=X[n]
        return b,s

    def step_NISA(self):
        candidate = self.two_opt_metropolis(self.solution)
        current_score = compute_distance(self.solution, self.problem)
        candidate_score = compute_distance(candidate, self.problem)
        delta = candidate_score - current_score

        if delta < 0 or np.random.rand() < np.exp(-delta / self.temperature):
            self.solution = candidate
            if candidate_score < self.Fbest:
                self.gbest = deepcopy(candidate)
                self.Fbest = candidate_score

        rnd = np.random.random_sample()
        if rnd>self.r:
            Sol,v0,Fit=self.dataBat(3)
            Sol.append(self.gbest)
            Fit.append(self.Fbest)
            X=deepcopy(Sol)
            fitness=deepcopy(Fit)
            FbestP,gbestP=self.best(X,fitness)
            for n in range(5):
                    for i in range(4):
                            newfi=compute_distance(self.solution, self.problem)
                            X[i],fitness[i]=self.two_opt(X[i],self.solution)
                            if fitness[i] < FbestP:
                                gbestP=deepcopy(X[i])
                                FbestP=fitness[i]
                            if FbestP<newfi:
                                    self.solution=gbestP
                            if FbestP<self.Fbest:
                                    self.gbest=gbestP
                                    self.Fbest=FbestP
                                    

        self.fitness_history.append(self.Fbest)
        self.fitness_evolution.append(current_score)
        self.temperature_evolution.append(self.temperature)

    def step_NISA_greedy(self):
        old_score = compute_distance(self.solution, self.problem)
        leader_idx = self.select_leader_epsilon()
        leader=self.setcandidat[leader_idx]
        candidate = self.two_opt_metropolis(leader)
        current_score = compute_distance(self.solution, self.problem)
        candidate_score = compute_distance(candidate, self.problem)
        delta = candidate_score - current_score

        if delta < 0 or np.random.rand() < np.exp(-delta / self.temperature):
            self.solution = candidate
            if candidate_score < self.Fbest:
                self.gbest = deepcopy(candidate)
                self.Fbest = candidate_score
            if candidate_score < self.Fpbest:
                self.pbest = deepcopy(candidate)
                self.Fpbest = candidate_score

        reward = (old_score - candidate_score)/old_score
        self.update_q_table(0, leader_idx, reward)

        rnd = np.random.random_sample()
        if rnd>self.r:
            Sol,v0,Fit=self.dataBat(3)
            Sol.append(self.gbest)
            Fit.append(self.Fbest)
            X=deepcopy(Sol)
            fitness=deepcopy(Fit)
            FbestP,gbestP=self.best(X,fitness)
            for n in range(5):
                    for i in range(4):
                            newfi=compute_distance(self.solution, self.problem)
                            X[i],fitness[i]=self.two_opt(X[i],self.solution)
                            if fitness[i] < FbestP:
                                gbestP=deepcopy(X[i])
                                FbestP=fitness[i]
                            if FbestP<newfi:
                                    self.solution=gbestP
                            if FbestP<self.Fbest:
                                    self.gbest=gbestP
                                    self.Fbest=FbestP
                                    

        self.fitness_history.append(self.Fbest)
        self.fitness_evolution.append(current_score)
        self.temperature_evolution.append(self.temperature)

    def step_Soft_NISA(self):
        old_score = compute_distance(self.solution, self.problem)
        leader_idx = self.select_leader()
        leader=self.setcandidat[leader_idx]
        ##print('self.setcandidat,leader_idx,leader',self.setcandidat,leader_idx,leader)
        #print('two_opt metro')
        candidate = self.two_opt_metropolis(leader)
        current_score = compute_distance(self.solution, self.problem)
        candidate_score = compute_distance(candidate, self.problem)
        delta = candidate_score - current_score

        if delta < 0 or np.random.rand() < np.exp(-delta / self.temperature):
            self.solution = candidate
            if candidate_score < self.Fbest:
                self.gbest = deepcopy(candidate)
                self.Fbest = candidate_score
            if candidate_score < self.Fpbest:
                self.pbest = deepcopy(candidate)
                self.Fpbest = candidate_score

        reward = (old_score - candidate_score)/old_score
        self.update_q_table(0, leader_idx, reward)

        rnd = np.random.random_sample()
        if rnd>self.r:
            Sol,v0,Fit=self.dataBat(3)
            Sol.append(self.gbest)
            Fit.append(self.Fbest)
            X=deepcopy(Sol)
            fitness=deepcopy(Fit)
            FbestP,gbestP=self.best(X,fitness)
            for n in range(5):
                    for i in range(4):
                            newfi=compute_distance(self.solution, self.problem)
                            #print('two_opt ')
                            X[i],fitness[i]=self.two_opt(X[i],self.solution)
                            #print('after two_opt ')
                            if fitness[i] < FbestP:
                                gbestP=deepcopy(X[i])
                                FbestP=fitness[i]
                            if FbestP<newfi:
                                    self.solution=gbestP
                            if FbestP<self.Fbest:
                                    self.gbest=gbestP
                                    self.Fbest=FbestP
                                    

        self.fitness_history.append(self.Fbest)
        self.fitness_evolution.append(current_score)
        self.temperature_evolution.append(self.temperature)

    def step(self):
            old_score = compute_distance(self.solution, self.problem)
            leader_idx = self.select_leader()
            leader=self.setcandidat[leader_idx]
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
                if candidate_score < self.Fpbest:
                    self.pbest = deepcopy(candidate)
                    self.Fpbest = candidate_score

            self.fitness_history.append(self.Fbest)
            self.fitness_evolution.append(current_score)
            self.temperature_evolution.append(self.temperature)

            reward = (old_score - candidate_score)/old_score
            self.update_q_table(0, leader_idx, reward)
            #self.update_setcandidat()

    def step_uniform(self):
            leader_idx = np.random.choice(range(len(self.setcandidat)))
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
                if candidate_score < self.Fpbest:
                    self.pbest = deepcopy(candidate)
                    self.Fpbest = candidate_score

            self.fitness_history.append(self.Fbest)
            self.fitness_evolution.append(current_score)
            self.temperature_evolution.append(self.temperature)
            #self.update_setcandidat()


    def step_greedy(self):
            old_score = compute_distance(self.solution, self.problem)
            leader_idx = self.select_leader_epsilon()
            leader=self.setcandidat[leader_idx]
            candidate = self.two_opt_metropolis(leader)
            current_score = compute_distance(self.solution, self.problem)
            candidate_score = compute_distance(candidate, self.problem)
            delta = candidate_score - current_score

            if delta < 0 or np.random.rand() < np.exp(-delta / self.temperature):
                self.solution = candidate
                if candidate_score < self.Fbest:
                    self.gbest = deepcopy(candidate)
                    self.Fbest = candidate_score
                if candidate_score < self.Fpbest:
                    self.pbest = deepcopy(candidate)
                    self.Fpbest = candidate_score

            self.fitness_history.append(self.Fbest)
            self.fitness_evolution.append(current_score)
            self.temperature_evolution.append(self.temperature)

            reward = (old_score - candidate_score)/old_score
            self.update_q_table(0, leader_idx, reward)
            #self.update_setcandidat()

    def step2(self):
            old_score = compute_distance(self.solution, self.problem)
            leader_idx = self.select_leader()
            leader=self.setcandidat[leader_idx]
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
                if candidate_score < self.Fpbest:
                    self.pbest = deepcopy(candidate)
                    self.Fpbest = candidate_score

            self.fitness_history.append(self.Fbest)
            self.fitness_evolution.append(current_score)
            self.temperature_evolution.append(self.temperature)

            reward = (old_score - candidate_score)/old_score
            self.update_q_table2(0, leader_idx, reward)
            #self.update_setcandidat()

    

    def step_greedy2(self):
            old_score = compute_distance(self.solution, self.problem)
            leader_idx = self.select_leader_epsilon()
            leader=self.setcandidat[leader_idx]
            candidate = self.two_opt_metropolis(leader)
            current_score = compute_distance(self.solution, self.problem)
            candidate_score = compute_distance(candidate, self.problem)
            delta = candidate_score - current_score

            if delta < 0 or np.random.rand() < np.exp(-delta / self.temperature):
                self.solution = candidate
                if candidate_score < self.Fbest:
                    self.gbest = deepcopy(candidate)
                    self.Fbest = candidate_score
                if candidate_score < self.Fpbest:
                    self.pbest = deepcopy(candidate)
                    self.Fpbest = candidate_score

            self.fitness_history.append(self.Fbest)
            self.fitness_evolution.append(current_score)
            self.temperature_evolution.append(self.temperature)

            reward = (old_score - candidate_score)/old_score
            self.update_q_table2(0, leader_idx, reward)
            #self.update_setcandidat()

    
    
    def run_Soft_NISA(self, iterations=500,episodes=50):
        #print('nisa softmax')
        start_time = time.time()  # Start timer
        for i in range(iterations):
            #print('i',i)
            self.r = self.rp *(1-np.exp(-self.gamma1*(i+1)))
            self.temperature =  (self.temperature_max-((self.temperature_max-self.tempmin)*((i+1))) /iterations  )
            #print('bef step_Soft_NISA')
            self.step_Soft_NISA()
            #print('after step_Soft_NISA')
            if(i+1) % episodes==0:
                self.reset_q_table()
            ###print(f"Iteration {i}, Temp: {self.temperature:.4f}, Best: {self.Fbest:.2f}")
        #print('nisa soft',self.Fbest)
        end_time = time.time()  # End timer
        execution_time = end_time - start_time
        return self.gbest, self.Fbest, self.fitness_history,self.fitness_evolution, self.temperature_evolution,execution_time
    
    def run_SA(self, iterations=500):
        #print('sa ')
        start_time = time.time()  # Start timer
        for i in range(iterations):
            self.temperature =  (self.temperature_max-((self.temperature_max-self.tempmin)*((i+1))) /iterations  )
            self.step_SA()
            ###print(f"Iteration {i}, Temp: {self.temperature:.4f}, Best: {self.Fbest:.2f}")
        end_time = time.time()  # End timer
        execution_time = end_time - start_time
        return self.gbest, self.Fbest, self.fitness_history,self.fitness_evolution, self.temperature_evolution,execution_time
    
    def run_NISA(self, iterations=500):
        #print('nisa ')
        start_time = time.time()  # Start timer
        for i in range(iterations):
            self.r = self.rp *(1-np.exp(-self.gamma1*(i+1)))
            self.temperature =  (self.temperature_max-((self.temperature_max-self.tempmin)*((i+1))) /iterations  )
            self.step_NISA()
            ###print(f"Iteration {i}, Temp: {self.temperature:.4f}, Best: {self.Fbest:.2f}")
        end_time = time.time()  # End timer
        execution_time = end_time - start_time
        return self.gbest, self.Fbest, self.fitness_history,self.fitness_evolution, self.temperature_evolution,execution_time
    
    def run_greedy(self, iterations=500, episodes=50):
        #print('sa greedy')
        start_time = time.time()  # Start timer
        for i in range(iterations):
            self.temperature =  (self.temperature_max-((self.temperature_max-self.tempmin)*((i+1))) /iterations  )
            self.step_greedy()
            if(i+1) % episodes==0:
                self.reset_q_table()
            #print(f"Iteration {i}, Temp: {self.temperature:.4f}, Best: {self.Fbest:.2f}")
        end_time = time.time()  # End timer
        execution_time = end_time - start_time
        return self.gbest, self.Fbest, self.fitness_history,self.fitness_evolution, self.temperature_evolution,execution_time
    
    def run(self, iterations=500,episodes=50):
        #print('sa softmax')
        start_time = time.time()  # Start timer
        for i in range(iterations):
            self.temperature =  (self.temperature_max-((self.temperature_max-self.tempmin)*((i+1))) /iterations  )
            self.step()
            if(i+1) % episodes==0:
                self.reset_q_table()
            ##print(f"Iteration {i}, Temp: {self.temperature:.4f}, Best: {self.Fbest:.2f}")
        end_time = time.time()  # End timer
        execution_time = end_time - start_time
        return self.gbest, self.Fbest, self.fitness_history,self.fitness_evolution, self.temperature_evolution,execution_time
    
    def run_greedy2(self, iterations=500, episodes=50):
        # print('sa greedy')
        start_time = time.time()  # Start timer
        for i in range(iterations):
            self.temperature =  (self.temperature_max-((self.temperature_max-self.tempmin)*((i+1))) /iterations  )
            self.step_greedy2()
            if(i+1) % episodes==0:
                self.reset_q_table()
            #print(f"Iteration {i}, Temp: {self.temperature:.4f}, Best: {self.Fbest:.2f}")
        end_time = time.time()  # End timer
        execution_time = end_time - start_time
        return self.gbest, self.Fbest, self.fitness_history,self.fitness_evolution, self.temperature_evolution,execution_time
    
    def run2(self, iterations=500,episodes=50):
        # print('sa softmax')
        start_time = time.time()  # Start timer
        for i in range(iterations):
            self.temperature =  (self.temperature_max-((self.temperature_max-self.tempmin)*((i+1))) /iterations  )
            self.step2()
            if(i+1) % episodes==0:
                self.reset_q_table()
            ##print(f"Iteration {i}, Temp: {self.temperature:.4f}, Best: {self.Fbest:.2f}")
        end_time = time.time()  # End timer
        execution_time = end_time - start_time
        return self.gbest, self.Fbest, self.fitness_history,self.fitness_evolution, self.temperature_evolution,execution_time
    
    def run_greedy_NISA(self, iterations=500, episodes=50):
        #print('sa greedy nisa')
        start_time = time.time()  # Start timer
        for i in range(iterations):
            self.r = self.rp *(1-np.exp(-self.gamma1*(i+1)))
            self.temperature =  (self.temperature_max-((self.temperature_max-self.tempmin)*((i+1))) /iterations  )
            self.step_NISA_greedy()
            if(i+1) % episodes==0:
                self.reset_q_table()
            ##print(f"Iteration {i}, Temp: {self.temperature:.4f}, Best: {self.Fbest:.2f}")
        end_time = time.time()  # End timer
        execution_time = end_time - start_time
        return self.gbest, self.Fbest, self.fitness_history,self.fitness_evolution, self.temperature_evolution,execution_time

    def run_uniform(self, iterations=500,episodes=50):
        #print('sa softmax')
        start_time = time.time()  # Start timer
        for i in range(iterations):
            self.temperature =  (self.temperature_max-((self.temperature_max-self.tempmin)*((i+1))) /iterations  )
            self.step_uniform()
        end_time = time.time()  # End timer
        execution_time = end_time - start_time
        return self.gbest, self.Fbest, self.fitness_history,self.fitness_evolution, self.temperature_evolution,execution_time
  



runs = 30
TestsFilePath = 'inputs/'

def runAlgo(params):
    TestsFilePath, problem_name, initial_solution, param = params
    temperature = 1000.0
    cooling_rate = 0.99
    gamma1 = 0.9
    rp = 0.4
    epsilon, alpha, gamma, des, tempmin = 0.6, 0.1, 0.95, 0.001, 0.001
    #problem = tsplib95.load(TestsFilePath + problem_name + '.tsp')

    sa_obj = SimulatedAnnealing_TSP_Logging(
        TestsFilePath, problem_name, initial_solution, 
        temperature, cooling_rate, tempmin,
        epsilon, alpha, gamma, des, gamma1, rp
    )
    Iter=sa_obj.nbrville*500
    print('problem_name,Iter',problem_name,Iter)
    episodes=Iter*0.1

    if param == 1:
        return sa_obj.run2(iterations=Iter, episodes=episodes)
    elif param == 2:
        return sa_obj.run_SA(iterations=Iter)
    elif param == 3:
        return sa_obj.run_greedy2(iterations=Iter, episodes=episodes)
    # elif param == 4:
    #     return sa_obj.run_uniform(iterations=Iter, episodes=episodes)

    else:
        raise ValueError(f"Invalid param value: {param}")

# ===============================
# Task Preparation
# ===============================
def prepare_tasks(ListProb, TestsFilePath, runs):
    tasks = []
    for PROB in ListProb:
        problem = tsplib95.load(TestsFilePath + PROB + '.tsp')
        nbrville = problem.dimension
        has_node_coords = (problem.node_coords != {} or problem.display_data != {})
        for k in range(runs):
            # SEED = 0 + k*100 #42
            # np.random.seed(SEED)
            initial_solution = generate_tsp(1, nbrville, has_node_coords)[0]
            for p in range(1,4):
                print('instance ', PROB, ' run ', k, ' algo ',p)  # 1..5 algorithms
                tasks.append((TestsFilePath, PROB, initial_solution, p))
    return tasks

# ===============================
# Parallel Execution
# ===============================
def parallel_run(tasks, max_workers=None):
    if max_workers is None:
        max_workers = max(1, int(multiprocessing.cpu_count() * 0.6))  # use 80% of cores
    results = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_task = {executor.submit(runAlgo, t): t for t in tasks}
        for future in as_completed(future_to_task):
            task = future_to_task[future]
            try:
                res = future.result()
                results.append((task, res))
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

def DF_results_parallel(ListProb, TestsFilePath, runs):
    # Hyperparameters
    #Iter, episodes = 1000, 100
    Iter, episodes = 1000, 100
    epsilon, alpha, gamma, des, tempmin = 0.6, 0.1, 0.95, 0.001, 0.001
    date = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Prepare all tasks
    tasks = prepare_tasks(ListProb, TestsFilePath, runs)

    # Parallel execution
    results = parallel_run(tasks)

    # Algorithm names
    MM = ['QLSA_s', 'SA', 'QLSA_e']#, 'SA_UNIFORM']  # Internal names for CSVs
    pretty_names = {'QLSA_s': 'QLSA_s', 'SA': 'SA', 'QLSA_e': 'QLSA_ε'}#, 'SA_UNIFORM':'SA_U'}  # For plots

    # Storage
    all_conv_data = {prob: {algo: [] for algo in MM} for prob in ListProb}
    all_accepted_data = {prob: {algo: [] for algo in MM} for prob in ListProb}
    all_results = {prob: pd.DataFrame({algo: pd.Series(dtype=float) for algo in MM})
               for prob in ListProb}  # Fbest per run
    
    runtime_per_algo = {prob: {algo: [] for algo in MM} for prob in ListProb}  # seconds per run

    # Track best across runs (per prob x algo)
    best_across = {prob: {algo: {"Fbest": float("inf"), "run": None, "gbest": None}
                          for algo in MM} for prob in ListProb}

    # Output dirs
    base_dir = "./New_Results_09_09_2025"
    os.makedirs(base_dir, exist_ok=True)
    plot_dir = f"{base_dir}/Plots"
    os.makedirs(plot_dir, exist_ok=True)
    gbest_dir = f"{base_dir}/gbest_runs"
    os.makedirs(gbest_dir, exist_ok=True)

    # Master runtime CSV (append)
    runtime_master_csv = f"{base_dir}/runtime_master_{date}.csv"
    runtime_header_written = False

    # Save parameters for reproducibility
    params_data = {
        "Date": date,
        "Iterations": Iter,
        "Episodes": episodes,
        "Epsilon": epsilon,
        "Alpha": alpha,
        "Gamma": gamma,
        "Decay": des,
        "TempMin": tempmin,
        "Runs": runs,
        "Instances": ", ".join(ListProb)
    }
    pd.DataFrame([params_data]).to_csv(f"{base_dir}/parameters_{date}.csv", index=False)

    # Populate results
    for task, zres in results:
        TestsFilePath, PROB, initial_solution, param, run_id, *_ = task  # ensure prepare_tasks sets run_id (1..runs)
        algo_name = MM[param - 1]

        # --- Unpack from zres (adjust indices if your worker returns a different shape)
        gbest_route = zres[0]
        run_Fbest = float(zres[1])
        conv_curve = zres[2]
        accepted_curve = zres[3]   # THIRD RESULT: accepted fitness values
        # zres[4] could be temperature curve (unused here)
        exec_time_s = float(zres[5])  # <-- requires worker to return exec time

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
            "Fbest": run_Fbest
        }
        mode = "a" if os.path.exists(runtime_master_csv) else "w"
        df_row = pd.DataFrame([row])
        df_row.to_csv(runtime_master_csv, mode=mode, header=not os.path.exists(runtime_master_csv), index=False)

        # 3) Update per-instance DataFrame of Fbest
        current_df = all_results[PROB]
        # ensure column exists
        if algo_name not in current_df.columns:
            current_df[algo_name] = np.nan
        current_df.loc[len(current_df), algo_name] = run_Fbest
        all_results[PROB] = current_df

        # 4) Store convergence & accepted solution curves (for mean plots)
        all_conv_data[PROB][algo_name].append(conv_curve)
        all_accepted_data[PROB][algo_name].append(accepted_curve)

        # 5) Store runtimes for stats
        runtime_per_algo[PROB][algo_name].append(exec_time_s)

        # 6) Track best across runs (per prob x algo)
        if run_Fbest < best_across[PROB][algo_name]["Fbest"]:
            best_across[PROB][algo_name] = {
                "Fbest": run_Fbest,
                "run": run_id,
                "gbest": np.asarray(gbest_route, dtype=int)
            }

    # Save per-instance results, runtime stats & plots
    for prob in ListProb:
        # Save detailed Fbest runs
        file_path = f"{base_dir}/{prob}_runs_{date}.csv"
        all_results[prob].to_csv(file_path, index=False)

        # Save descriptive stats for Fbest
        desc_path = f"{base_dir}/{prob}_stats_{date}.csv"
        all_results[prob].describe().to_csv(desc_path)

        # Save runtime per algo (tidy, one column per algo + describe)
        print('runtime_per_algo',runtime_per_algo)
        runtime_df = pd.DataFrame({algo: np.asarray(runtime_per_algo[prob][algo], dtype=float) for algo in MM})
        runtime_runs_csv = f"{base_dir}/{prob}_runtime_runs_{date}.csv"
        runtime_df.to_csv(runtime_runs_csv, index=False)

        runtime_stats_csv = f"{base_dir}/{prob}_runtime_stats_{date}.csv"
        runtime_df.describe().to_csv(runtime_stats_csv)

        # Save "best across runs" metadata + gbest text
        best_meta_rows = []
        for algo in MM:
            b = best_across[prob][algo]
            # Save the best gbest as a separate file
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
        pd.DataFrame(best_meta_rows).to_csv(f"{base_dir}/{prob}_best_across_runs_{date}.csv", index=False)

        # =======================
        # Convergence plot (mean)
        # =======================
        # guard against empty lists
        if all(len(all_conv_data[prob][a]) > 0 for a in MM):
            min_len = min(min(len(c) for c in all_conv_data[prob][algo]) for algo in MM if all_conv_data[prob][algo])
            mean_conv = {algo: np.mean([np.asarray(c)[:min_len] for c in all_conv_data[prob][algo]], axis=0) for algo in MM}
            iterations = list(range(min_len))

            fig1 = go.Figure()
            for algo in MM:
                fig1.add_trace(go.Scatter(x=iterations, y=mean_conv[algo], name=pretty_names[algo], mode='lines'))

            fig1.update_layout(
                title=f"Convergence Plot (Mean Best-So-Far) - {prob}",
                xaxis_title="Iteration",
                yaxis_title="Fitness",
                template="plotly_white",
                legend=dict(x=0.01, y=0.99)
            )
            pio.write_html(fig1, file=f"{plot_dir}/{prob}_convergence_{date}.html", auto_open=False)
            pio.write_image(fig1, f"{plot_dir}/{prob}_convergence_{date}.png")

        # =======================
        # Accepted Fitness Plot (mean)
        # =======================
        if all(len(all_accepted_data[prob][a]) > 0 for a in MM):
            min_len_acc = min(min(len(c) for c in all_accepted_data[prob][algo]) for algo in MM if all_accepted_data[prob][algo])
            mean_accepted = {algo: np.mean([np.asarray(c)[:min_len_acc] for c in all_accepted_data[prob][algo]], axis=0) for algo in MM}
            iterations_acc = list(range(min_len_acc))

            fig2 = go.Figure()
            for algo in MM:
                fig2.add_trace(go.Scatter(x=iterations_acc, y=mean_accepted[algo], name=pretty_names[algo], mode='lines'))

            fig2.update_layout(
                title=f"Accepted Fitness Plot (Mean Accepted Solutions) - {prob}",
                xaxis_title="Iteration",
                yaxis_title="Fitness of Accepted Solutions",
                template="plotly_white",
                legend=dict(x=0.01, y=0.99)
            )
            pio.write_html(fig2, file=f"{plot_dir}/{prob}_accepted_fitness_{date}.html", auto_open=False)
            pio.write_image(fig2, f"{plot_dir}/{prob}_accepted_fitness_{date}.png")

                    # =======================
        # FIGURE 3: Best Routes (gbest) — 3 subplots
        # =======================
        try:
            coords = _load_coords(prob, TestsFilePath)  # shape (n,2)
            

            fig3 = make_subplots(rows=1, cols=3, subplot_titles=[pretty_names[a] for a in MM])

            # lock aspect ratio per subplot
            for i_col in range(1, 4):
                fig3.update_xaxes(scaleanchor=f"y{i_col}", scaleratio=1, row=1, col=i_col)

            for col, algo in enumerate(MM, start=1):
                b = best_across[prob][algo]  # {"Fbest": .., "run": .., "gbest": np.array or None}
                route = b["gbest"]
                if route is None or len(route) == 0:
                    fig3.add_annotation(row=1, col=col, text="No best route found", showarrow=False)
                    continue

                # If your saved indices are 1-based, uncomment:
                # route = route - 1

                route = np.asarray(route, dtype=int).flatten()
                loop = np.r_[route, route[0]]
                xs = coords[loop, 0]
                ys = coords[loop, 1]

                # edges
                fig3.add_trace(
                    go.Scatter(x=xs, y=ys, mode="lines", name=f"{pretty_names[algo]} edges", showlegend=False),
                    row=1, col=col
                )
                # nodes
                fig3.add_trace(
                    go.Scatter(x=coords[:, 0], y=coords[:, 1], mode="markers",
                               marker=dict(size=6), name=f"{pretty_names[algo]} nodes", showlegend=False),
                    row=1, col=col
                )
                # start node
                fig3.add_trace(
                    go.Scatter(x=[coords[route[0], 0]], y=[coords[route[0], 1]],
                               mode="markers+text", text=["start"], textposition="top center",
                               marker=dict(size=9, symbol="star"), showlegend=False),
                    row=1, col=col
                )

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


    return all_results, plot_dir


# ===============================
# MAIN
# ===============================
if __name__ == "__main__":
    TestsFilePath = "inputs/"  # adjust path
    runs = 30
    ListProb = ['eil101']#,'bayg29']#,'hk48']#,'gr24','gr17','bayg29','ulysses16','ulysses22','bays29']#,'dantzig42','swiss42','gr48','hk48']  # add more instances
    #ListProb = ['st70','pr76','eil76','rat99']#,'kroA100','kroB100','kroC100','kroD100','kroE100','eil101','lin105','pr124','ch150','tsp225']  # add more instances
    #ListProb = ['eil101']#,'kroA100']#,'kroB100','kroC100','kroD100','kroE100','eil101','lin105','pr124','ch150']#,'lin105','pr124','ch150','tsp225']

    #gr17,bayg29,bays29,oliver30,swiss42,eil51,berlin52,st70,pr76,eil76,rat99,kroA100,kroB100,kroC100,kroD100,kroE100,eil101,lin105,pr124,ch150,tsp225
    results, plots = DF_results_parallel(ListProb, TestsFilePath, runs)
    print("Saved detailed run results for each instance.")
    print("Plots saved in:", plots)