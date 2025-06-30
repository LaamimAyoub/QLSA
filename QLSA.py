import pandas as pd
from datetime import datetime
import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
from compute import compute_distance, softmax,generate_tsp,epsilon_greedy
from copy import deepcopy
import tsplib95
import multiprocessing

NB_ITERATIONS = 5000

class SimulatedAnnealing_TSP_Logging:
    def __init__(self, problem, initial_solution, temperature=1.0, cooling_rate=0.99, tempmin=0.01, epsilon=0.1, alpha=0.1, gamma=0.9, stagnation_window=100, stagnation_threshold=1e-6):
        self.problem = problem
        self.solution = deepcopy(initial_solution)
        self.gbest = deepcopy(initial_solution)

        self.has_node_coords = (self.problem.node_coords != {} or self.problem.display_data != {})

        self.Fbest = compute_distance(initial_solution, self.problem)
        self.pbest = deepcopy(initial_solution)
        self.Fpbest = compute_distance(initial_solution, self.problem)
        #print ('Fbest',self.Fbest)
        self.temperature = self.Fbest
        self.temperature_max = temperature
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
        self.stagnation_window = stagnation_window
        self.stagnation_threshold = stagnation_threshold

    def update_setcandidat(self):
        random_sol=generate_tsp(1,len(self.solution), self.has_node_coords)[0]
        self.setcandidat=[self.solution,self.gbest,self.pbest,random_sol]

    def two_opt(self,leader): #x une solution x=[(ci1,ci2),(),...]
        x=deepcopy(leader)        
        for k in range(1):
          p=np.random.choice(range(len(x)))
          for i in range(p,len(x) - 1):
            for j in range(i + 2, len(x) - 1):
                delta=(self.problem.get_weight(x[i], x[j]) + self.problem.get_weight(x[i+1] ,x[j+1]))-(self.problem.get_weight(x[i],x[i+1]) + self.problem.get_weight(x[j],x[j+1]))
                rnd = np.random.random_sample()
                if delta<0:
                    #print('2_opt_delta<0')
                    x[i+1:j+1]= reversed(x[i+1:j+1])
    
    def two_opt_metropolis(self,leader): #x une solution x=[(ci1,ci2),(),...]
        x=deepcopy(leader)        
        for k in range(10):
          p=np.random.choice(range(len(x)))
          #p=1
          for i in range(p,len(x) - 1):
            for j in range(i + 2, len(x) - 1):
                delta=(self.problem.get_weight(x[i], x[j]) + self.problem.get_weight(x[i+1] ,x[j+1]))-(self.problem.get_weight(x[i],x[i+1]) + self.problem.get_weight(x[j],x[j+1]))
                rnd = np.random.random_sample()
                if delta<0:
                    #print('2_opt_delta<0')
                    x[i+1:j+1]= reversed(x[i+1:j+1])
                elif rnd<np.exp(-delta/(self.temperature)):
                    #print('2_opt_metropolis')
                    x[i+1:j+1]= reversed(x[i+1:j+1])
          if compute_distance(x,self.problem) != compute_distance(leader,self.problem):
              break
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
        leader= epsilon_greedy(q_values,self.epsilon)
        self.leader_count[0][leader] += 1
        return leader
    
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

    def step(self):
            old_score = compute_distance(self.solution, self.problem)
            leader_idx = self.select_leader()
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
            reward = old_score - candidate_score
            self.update_q_table(0, leader_idx, reward)
            self.update_setcandidat()

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
            reward = old_score - candidate_score
            self.update_q_table(0, leader_idx, reward)
            self.update_setcandidat()

    def run(self, iterations=5000):
        use_qlsa = False
        for i in range(iterations):
            if use_qlsa:
                self.step()
            else:
                self.step_SA()
                if i > self.stagnation_window and (self.fitness_history[-self.stagnation_window] - self.Fbest) < self.stagnation_threshold:
                    use_qlsa = True

            self.temperature =  (self.temperature_max-((self.temperature_max-self.tempmin)*((i+1))) /iterations  )
            #print(f"Iteration {i}, Temp: {self.temperature:.4f}, Best: {self.Fbest:.2f}")
        return self.gbest, self.Fbest, self.fitness_history
    
    def run_SA(self, iterations=5000):
        for i in range(iterations):
            self.step_SA()
            self.temperature =  (self.temperature_max-((self.temperature_max-self.tempmin)*((i+1))) /iterations  )
            #print(f"Iteration {i}, Temp: {self.temperature:.4f}, Best: {self.Fbest:.2f}")
        return self.gbest, self.Fbest, self.fitness_history
    
    def run_greedy(self, iterations=5000):
        use_qlsa = False
        for i in range(iterations):
            if use_qlsa:
                self.step_greedy()
            else:
                self.step_SA()
                if i > self.stagnation_window and (self.fitness_history[-self.stagnation_window] - self.Fbest) < self.stagnation_threshold:
                    use_qlsa = True

            self.temperature =  (self.temperature_max-((self.temperature_max-self.tempmin)*((i+1))) /iterations  )
            #print(f"Iteration {i}, Temp: {self.temperature:.4f}, Best: {self.Fbest:.2f}")
        return self.gbest, self.Fbest, self.fitness_history


runs = 
TestsFilePath = 'inputs/'

def runAlgo(params):
    problem, initial_solution=params[1:]
    param=params[0]

    if param==1:
        qlsa = SimulatedAnnealing_TSP_Logging(problem, initial_solution, temperature=1000.0, cooling_rate=0.99, tempmin=0.001,epsilon=0.3, alpha=0.2, gamma=0.8)
        zqlsa=qlsa.run(NB_ITERATIONS)
        return zqlsa
    elif param==2:
        sa = SimulatedAnnealing_TSP_Logging(problem, initial_solution, temperature=1000.0, cooling_rate=0.99, tempmin=0.001,epsilon=0.3, alpha=0.2, gamma=0.8)
        zsa=sa.run_SA(NB_ITERATIONS)
        return zsa
    
    elif param==3:
        sag = SimulatedAnnealing_TSP_Logging(problem, initial_solution, temperature=1000.0, cooling_rate=0.99, tempmin=0.001,epsilon=0.3, alpha=0.2, gamma=0.8)
        zsag=sag.run_greedy(NB_ITERATIONS)
        return zsag
    
def DF_results(nbrville, problem,runs,TestsFilePath):
    
    ktest=runs
    MM=['SA','QLSA softmax','QLSA epsilon_greedy']
    df=pd.DataFrame(columns=MM)
    dfbstpop=pd.DataFrame(columns=MM)
    dfTime=pd.DataFrame(columns=MM)
    convSA=[]
    convQLSA=[]
    convQLSAg=[]
    print(problem.name)
    for k in range(ktest):
      print('k: ',k)
      l=[]
      lPOP=[]
      Time=[]
      initial_solution = generate_tsp(1,nbrville)[0]
      params=[(p,problem, initial_solution)for p in range(1, 4)]
      pool = multiprocessing.Pool()
      zres=pool.map(runAlgo, params)
      #zres = [runAlgo(p) for p in params]
      pool.close()
      pool.join()
      #print(zres)
      zQLSA=zres[0]
      Zsa=zres[1]
      zQLSAg=zres[2]
     

      l=[float(Zsa[1])]+[float(zQLSA[1])]+[float(zQLSAg[1])]
      lPOP=[Zsa[0]]+[zQLSA[0]]+[zQLSAg[0]]

      df2=pd.DataFrame([l],columns=MM)
      df = pd.concat([df, df2], ignore_index=True)

      df3=pd.DataFrame([lPOP],columns=MM)
      dfbstpop = pd.concat([dfbstpop, df3], ignore_index=True)

      

      convQLSA.append(zQLSA[2])
      convSA.append(Zsa[2])
      convQLSAg.append(zQLSAg[2])


    date = datetime.now().strftime('%Y%m%d_%H%M%S')

    namefile="./Result/Fbests_SA_TSP_Results_" + date + "_"+problem.name+"_SAs_compgraph.txt"
    namefile1="./Result/conv_SA_TSP_Results_" + date + "_"+problem.name+"_SAs_compgraph.txt"

    f= open(namefile,'w+')
    f1= open(namefile1,'w+')

    f.write(str(ktest)+' tests:')
    f.write("\n")
    f.write(df.to_string())
    f.write("\n")
    f.write(df.describe().to_string())
    f.write("\n")
    f.write(dfTime.to_string())
    f.write("\n")
    f.write(dfTime.describe().to_string())
    f1.write('Pour la convergence de SA: ')
    f1.write("\n")
    f1.write(str(convSA))
    f1.write("\n")

    f1.write('Pour la convergence de qlSAg: ')
    f1.write("\n")
    f1.write(str(convQLSAg))
    f1.write("\n")

    f1.write('Pour la convergence de qlsa: ')
    f1.write("\n")
    f1.write(str(convQLSA))
    f1.write("\n")
    f.close()
    f1.close()
    #print(problem)
    return df.describe()

if __name__ == "__main__":
    
    #ListProb = ['bayg29','bays29','burma14','berlin52','ulysses16','ulysses22']
    ListProb = ['berlin52','eil51','eil76']

    for PROB in ListProb:
        problem = tsplib95.load(TestsFilePath + PROB + '.tsp')
        nbrville = problem.dimension
        a=DF_results(nbrville, problem,runs,TestsFilePath)
        print(a)

    




