import pandas as pd
from datetime import datetime
import numpy as np
from compute import compute_distance, softmax,generate_tsp,epsilon_greedy
from copy import deepcopy
import tsplib95
import multiprocessing
import plotly.graph_objects as go
import plotly.io as pio

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
        # self.temperature = self.Fbest/2
        # self.temperature_max = self.Fbest/2
        self.temperature = temperature
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

        self.fitness_history.append(self.Fbest)
        self.fitness_evolution.append(current_score)
        self.temperature_evolution.append(self.temperature)

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

        self.fitness_history.append(self.Fbest)
        self.fitness_evolution.append(current_score)
        self.temperature_evolution.append(self.temperature)

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

  
    
    def run_Soft_NISA(self, iterations=500,episodes=50):
        #print('nisa softmax')
        for i in range(iterations):
            #print('i',i)
            self.r = self.rp *(1-np.exp(-self.gamma1*(i+1)))
            self.temperature =  (self.temperature_max-((self.temperature_max-self.tempmin)*((i+1))) /iterations  )
            #print('bef step_Soft_NISA')
            self.step_Soft_NISA()
            #print('after step_Soft_NISA')
            if iterations %episodes==0:
                self.reset_q_table()
            ###print(f"Iteration {i}, Temp: {self.temperature:.4f}, Best: {self.Fbest:.2f}")
        #print('nisa soft',self.Fbest)
        return self.gbest, self.Fbest, self.fitness_history,self.fitness_evolution, self.temperature_evolution
    
    def run_SA(self, iterations=500):
        #print('sa ')
        for i in range(iterations):
            self.temperature =  (self.temperature_max-((self.temperature_max-self.tempmin)*((i+1))) /iterations  )
            self.step_SA()
            ###print(f"Iteration {i}, Temp: {self.temperature:.4f}, Best: {self.Fbest:.2f}")
        return self.gbest, self.Fbest, self.fitness_history,self.fitness_evolution, self.temperature_evolution
    
    def run_NISA(self, iterations=500):
        #print('nisa ')
        for i in range(iterations):
            self.r = self.rp *(1-np.exp(-self.gamma1*(i+1)))
            self.temperature =  (self.temperature_max-((self.temperature_max-self.tempmin)*((i+1))) /iterations  )
            self.step_NISA()
            ###print(f"Iteration {i}, Temp: {self.temperature:.4f}, Best: {self.Fbest:.2f}")
        return self.gbest, self.Fbest, self.fitness_history,self.fitness_evolution, self.temperature_evolution
    
    def run_greedy(self, iterations=500, episodes=50):
        print('sa greedy')
        for i in range(iterations):
            self.temperature =  (self.temperature_max-((self.temperature_max-self.tempmin)*((i+1))) /iterations  )
            self.step_greedy()
            if iterations %episodes==0:
                self.reset_q_table()
            #print(f"Iteration {i}, Temp: {self.temperature:.4f}, Best: {self.Fbest:.2f}")
        return self.gbest, self.Fbest, self.fitness_history,self.fitness_evolution, self.temperature_evolution
    
    def run(self, iterations=500,episodes=50):
        print('sa softmax')
        for i in range(iterations):
            self.temperature =  (self.temperature_max-((self.temperature_max-self.tempmin)*((i+1))) /iterations  )
            self.step()
            if iterations %episodes==0:
                self.reset_q_table()
            ##print(f"Iteration {i}, Temp: {self.temperature:.4f}, Best: {self.Fbest:.2f}")
        return self.gbest, self.Fbest, self.fitness_history,self.fitness_evolution, self.temperature_evolution
    
    def run_greedy_NISA(self, iterations=500, episodes=50):
        #print('sa greedy nisa')
        for i in range(iterations):
            self.r = self.rp *(1-np.exp(-self.gamma1*(i+1)))
            self.temperature =  (self.temperature_max-((self.temperature_max-self.tempmin)*((i+1))) /iterations  )
            self.step_NISA_greedy()
            if iterations %episodes==0:
                self.reset_q_table()
            ##print(f"Iteration {i}, Temp: {self.temperature:.4f}, Best: {self.Fbest:.2f}")
        return self.gbest, self.Fbest, self.fitness_history,self.fitness_evolution, self.temperature_evolution





runs = 5
TestsFilePath = 'inputs/'

def runAlgo(params):
    #print("Running algorithm", params)
    param,TestsFilePath,problem, initial_solution=params
    Iter=1
    episodes=50

    if param==1:
        qlsa = SimulatedAnnealing_TSP_Logging(TestsFilePath,problem, initial_solution, temperature=1000.0, cooling_rate=0.99, tempmin=0.001,epsilon=0.6, alpha=0.1, gamma=0.95 ,des=0.001,gamma1=0.9,rp=0.4)
        zqlsa=qlsa.run(iterations=Iter, episodes=episodes)
        return zqlsa
    elif param==2:
        sa = SimulatedAnnealing_TSP_Logging(TestsFilePath,problem, initial_solution, temperature=1000.0, cooling_rate=0.99, tempmin=0.001,epsilon=0.6, alpha=0.1, gamma=0.95,des=0.001,gamma1=0.9,rp=0.4)
        zsa=sa.run_SA(iterations=Iter)
        return zsa
    
    elif param==3:
        sag = SimulatedAnnealing_TSP_Logging(TestsFilePath,problem, initial_solution, temperature=1000.0, cooling_rate=0.99, tempmin=0.001,epsilon=0.6, alpha=0.1, gamma=0.95,des=0.001,gamma1=0.9,rp=0.4)
        zsag=sag.run_greedy(iterations=Iter, episodes=episodes)
        return zsag
    
    if param==4:
        qlnisa = SimulatedAnnealing_TSP_Logging(TestsFilePath,problem, initial_solution, temperature=1000.0, cooling_rate=0.99, tempmin=0.001,epsilon=0.6, alpha=0.1, gamma=0.95 ,des=0.001,gamma1=0.9,rp=0.4)
        zqlnisa=qlnisa.run_NISA(iterations=Iter)
        return zqlnisa
    
    elif param==5:
        nisa = SimulatedAnnealing_TSP_Logging(TestsFilePath,problem, initial_solution, temperature=1000.0, cooling_rate=0.99, tempmin=0.001,epsilon=0.6, alpha=0.1, gamma=0.95,des=0.001,gamma1=0.9,rp=0.4)
        znisa=nisa.run_greedy_NISA(iterations=Iter, episodes=episodes)
        return znisa
    
    elif param==6:
        nisag = SimulatedAnnealing_TSP_Logging(TestsFilePath,problem, initial_solution, temperature=1000.0, cooling_rate=0.99, tempmin=0.001,epsilon=0.6, alpha=0.1, gamma=0.95,des=0.001,gamma1=0.9,rp=0.4)
        znisag=nisag.run_Soft_NISA(iterations=Iter, episodes=episodes)
        return znisag
    
def DF_results(nbrville, problem,runs,TestsFilePath):
    
    ktest=runs
    MM=['SA','QLSA softmax','QLSA epsilon_greedy','NISA', 'QLNISA greedy','QLNISA softmax']
    df=pd.DataFrame(columns=MM)
    dfbstpop=pd.DataFrame(columns=MM)
    convSA=[]
    convQLSA=[]
    convQLSAg=[]
    convQLSA_vals = []      # best-so-far
    fitQLSA_vals = []       # accepted
    tempQLSA_vals = []      # temperature

    convSA_vals = []
    fitSA_vals = []
    tempSA_vals = []

    convQLSAg_vals = []
    fitQLSAg_vals = []
    tempQLSAg_vals = []
    print(problem.name)
    for k in range(ktest):
        #print('k: ',k)
        l=[]
        lPOP=[]
        Time=[]
        has_node_coords = (problem.node_coords != {} or problem.display_data != {})
        initial_solution = generate_tsp(1,nbrville, has_node_coords)[0]
        params=[(p,TestsFilePath,problem.name, initial_solution)for p in range(1, 7)]
        pool = multiprocessing.Pool()
        zres=pool.map(runAlgo, params)
        #zres = [runAlgo(p) for p in params]
        pool.close()
        pool.join()
        ##print(zres)
        zQLSA=zres[0]
        Zsa=zres[1]
        zQLSAg=zres[2]

        zQLNISA=zres[3]
        ZNIsa=zres[4]
        zQLNISAg=zres[5]
        

        l=[float(Zsa[1])]+[float(zQLSA[1])]+[float(zQLSAg[1])]+[float(ZNIsa[1])]+[float(zQLNISA[1])]+[float(zQLNISAg[1])]
        lPOP=[Zsa[0]]+[zQLSA[0]]+[zQLSAg[0]]+[ZNIsa[0]]+[zQLNISA[0]]+[zQLNISAg[0]]

        df2=pd.DataFrame([l],columns=MM)
        df = pd.concat([df, df2], ignore_index=True)

        df3=pd.DataFrame([lPOP],columns=MM)
        dfbstpop = pd.concat([dfbstpop, df3], ignore_index=True)

        

        convQLSA.append(zQLSA[2])
        convSA.append(Zsa[2])
        convQLSAg.append(zQLSAg[2])

        convQLSA_vals.append(zQLSA[2])
        fitQLSA_vals.append(zQLSA[3])
        tempQLSA_vals.append(zQLSA[4])

        convSA_vals.append(Zsa[2])
        fitSA_vals.append(Zsa[3])
        tempSA_vals.append(Zsa[4])

        convQLSAg_vals.append(zQLSAg[2])
        fitQLSAg_vals.append(zQLSAg[3])
        tempQLSAg_vals.append(zQLSAg[4])

        



    #date = datetime.now().strftime('%Y%m%d_%H%M%S')

    # namefile="./Result/Fbests_SA_TSP_Results_" + date + "_"+problem.name+"_SAs_compgraph.txt"
    # namefile1="./Result/conv_SA_TSP_Results_" + date + "_"+problem.name+"_SAs_compgraph.txt"

    # f= open(namefile,'w+')
    # f1= open(namefile1,'w+')

    # f.write(str(ktest)+' tests:')
    # f.write("\n")
    # f.write(df.to_string())
    # f.write("\n")
    # f.write(df.describe().to_string())
    # f.write("\n")
    # f.write(dfTime.to_string())
    # f.write("\n")
    # f.write(dfTime.describe().to_string())
    # f1.write('Pour la convergence de SA: ')
    # f1.write("\n")
    # f1.write(str(convSA))
    # f1.write("\n")

    # f1.write('Pour la convergence de qlSAg: ')
    # f1.write("\n")
    # f1.write(str(convQLSAg))
    # f1.write("\n")

    # f1.write('Pour la convergence de qlsa: ')
    # f1.write("\n")
    # f1.write(str(convQLSA))
    # f1.write("\n")
    # f.close()
    # f1.close()
    date = datetime.now().strftime('%Y%m%d_%H%M%S')
    df.describe().to_csv("./New_Results/SA_TSP_Results_" + date + "_" + problem.name +".csv", index=False)
    # Create folder for plots
    plot_dir = "./New_Results/Plots"
    #os.makedirs(plot_dir, exist_ok=True)

    # Trim all convergence lists to the same length
    min_len = min(min(len(c) for c in convSA),
                min(len(c) for c in convQLSA),
                min(len(c) for c in convQLSAg))

    

    # Convert to numpy arrays
    fit_SA = np.array([x[1:min_len] for x in fitSA_vals])
    conv_SA = np.array([x[1:min_len] for x in convSA_vals])
    temp_SA = np.array([x[1:min_len] for x in tempSA_vals])

    fit_QLSA = np.array([x[1:min_len] for x in fitQLSA_vals])
    conv_QLSA = np.array([x[1:min_len] for x in convQLSA_vals])

    fit_QLSAg = np.array([x[1:min_len] for x in fitQLSAg_vals])
    conv_QLSAg = np.array([x[1:min_len] for x in convQLSAg_vals])
    # x-axis: iterations
    iterations = list(range(conv_SA.shape[1]))

    # Mean curves across runs
    mean_conv_SA = np.mean(conv_SA, axis=0)
    mean_fit_SA = np.mean(fit_SA, axis=0)

    mean_conv_QLSA = np.mean(conv_QLSA, axis=0)
    mean_fit_QLSA = np.mean(fit_QLSA, axis=0)

    mean_conv_QLSAg = np.mean(conv_QLSAg, axis=0)
    mean_fit_QLSAg = np.mean(fit_QLSAg, axis=0)

    # Mean temperature (SA only)
    mean_temp_SA = np.mean(temp_SA, axis=0)


    fig1 = go.Figure()

    fig1.add_trace(go.Scatter(x=iterations, y=mean_conv_SA, name="SA", mode='lines'))
    fig1.add_trace(go.Scatter(x=iterations, y=mean_conv_QLSA, name="QLSA Softmax", mode='lines'))
    fig1.add_trace(go.Scatter(x=iterations, y=mean_conv_QLSAg, name="QLSA Epsilon-Greedy", mode='lines'))
    # Temperature on secondary y-axis
    fig1.add_trace(go.Scatter(
        x=iterations, y=mean_temp_SA,
        name="SA Temperature",
        line=dict(color='red', dash='dot'),
        yaxis="y2"
    ))

    fig1.update_layout(
    title="Convergence Plot: Mean Best-So-Far Fitness - " +problem.name,
    xaxis_title="Iteration",
    yaxis_title="Fitness",
    yaxis2=dict(
        title="Temperature",
        overlaying="y",
        side="right",
        showgrid=False
    ),
    hovermode="x unified",
    template="plotly_white",
    legend=dict(x=0.01, y=0.99)
    )


    # Show and save
    #fig1.show()
    pio.write_html(fig1, file="./"+plot_dir+"/interactive_convergence_"+date+"_" +problem.name+".html", auto_open=False)
    # Optional PNG export (requires kaleido)
    pio.write_image(fig1, "./"+plot_dir+"/interactive_convergence_"+date+"_" +problem.name+".png")

    fig2 = go.Figure()

    fig2.add_trace(go.Scatter(x=iterations, y=mean_fit_SA, name="SA", mode='lines'))
    fig2.add_trace(go.Scatter(x=iterations, y=mean_fit_QLSA, name="QLSA Softmax", mode='lines'))
    fig2.add_trace(go.Scatter(x=iterations, y=mean_fit_QLSAg, name="QLSA Epsilon-Greedy", mode='lines'))

    # Temperature on secondary y-axis
    fig2.add_trace(go.Scatter(
        x=iterations, y=mean_temp_SA,
        name="SA Temperature",
        line=dict(color='red', dash='dot'),
        yaxis="y2"
    ))

    fig2.update_layout(
    title="Accepted Fitness Plot: Mean Fitness of Accepted Solutions - " +problem.name,
    xaxis_title="Iteration",
    yaxis_title="Fitness",
    yaxis2=dict(
        title="Temperature",
        overlaying="y",
        side="right",
        showgrid=False
    ),
    hovermode="x unified",
    template="plotly_white",
    legend=dict(x=0.01, y=0.99)
    )


    # Show and save
    #fig2.show()
    pio.write_html(fig2, file="./"+plot_dir+"/interactive_accepted_fitness_"+date+"_" +problem.name+".html", auto_open=False)
    # Optional PNG export
    pio.write_image(fig2, "./"+plot_dir+"/interactive_accepted_fitness_"+date+"_" +problem.name+".png")



    ##print(problem)
    return df.describe(), plot_dir

if __name__ == "__main__":
    #ListProb = ['kroC100']
    ListProb = ['bayg29','bays29','burma14','berlin52','eil51','eil76','kroA100','kroB100','kroC100','pr107']
    # ListProb = ['kroA100','kroB100','kroC100','pr107']
    #ListProb = ['a280','ali535','att532','d1291','d1655','d15112','d18512','pla85900']

    for PROB in ListProb:
        problem = tsplib95.load(TestsFilePath + PROB + '.tsp')
        nbrville = problem.dimension
        a=DF_results(nbrville, problem,runs,TestsFilePath)
        print(a)

    




