import tsplib95
#from QLSA import runAlgo, TestsFilePath
from QLSA5alg2_reset import runAlgo, TestsFilePath
from compute import generate_tsp
import multiprocessing
import os
import glob

file_lock = multiprocessing.Lock()

OUTPUT_FOLDER = "results"
OUTPUT_FILE = f"{OUTPUT_FOLDER}/optimals.csv"
NB_RUNS =5
NB_PROCESS = int(multiprocessing.cpu_count() * 0.8)
# Hyperparameters
Iter, episodes = 1000, 100
# Iter, episodes = 300000, 100
epsilon, alpha, gamma, des, tempmin = 0.6, 0.1, 0.95, 0.001, 0.001

ALGO_MAPPING = {
    1: "QL-SA_softmax",
    2: "SA",
    3: "Greedy",
    4: "Uniform",
    5: "QL-SA_softmax_without_reset",
    6: "Greedy_without_reset",
    7: "QL-SA_softmax_state",
    8: "Greedy_state",
    9: "Greedy_state_without_reset",
    10: "QL-SA_softmax_state_without_reset"
}

class Task:

    def __init__(self, problem, run_number, algo):
        self.problem = problem
        self.run_number = run_number
        self.algo = algo
        self.task_id = f"{problem}_{ALGO_MAPPING[self.algo]}_{run_number}"

    def run(self):
        problem = tsplib95.load_problem(f"{TestsFilePath}/{self.problem}.tsp")
        has_node_coords = (problem.node_coords != {} or problem.display_data != {})
        initial_solution = generate_tsp(1, problem.dimension, has_node_coords)[0]
        res = runAlgo((TestsFilePath, self.problem, initial_solution, self.algo, self.run_number, Iter, episodes,
     epsilon, alpha, gamma, des, tempmin))
        self.write_optimal(res[1])
        self.write_all_results(res[2])

    def write_optimal(self, optimal):
        add_header = False
        if not os.path.exists(OUTPUT_FILE):
            add_header = True

        with file_lock:
            with open(OUTPUT_FILE, "a") as f:
                if add_header:
                    f.write("Problem,run,Algo,Optimal\n")
                f.write(f"{self.problem},{self.run_number},{ALGO_MAPPING[self.algo]},{optimal}\n")

    def write_all_results(self, res):
        with open(os.path.join(OUTPUT_FOLDER, f"{self.task_id}.txt"), "w") as f:
            f.write(",".join([str(r) for r in res]))



def run_task(t :Task):
    # try:
    print("Running", t.task_id)
    t.run()
    # except Exception as e:
    #     print(t.task_id, "Failed With exception", e)

def clean_up_output():
    if os.path.exists(OUTPUT_FILE):
        os.remove(OUTPUT_FILE)

def get_list_problems(input_folder):
    list_inputs = glob.glob(f"{input_folder}/*.tsp")

    list_inputs = [os.path.basename(r).replace(".tsp", "") for r in list_inputs]
    return list_inputs

def build_and_run_tasks():

    clean_up_output()

    list_problems = get_list_problems("inputs")
    nb_runs = NB_RUNS
    algos = range(1, 11)

    list_tasks = []

    for i in range(1, nb_runs+1):
        for j in algos:
            for p in list_problems:
                t = Task(p, i, j)
                list_tasks.append(t)
    pool = multiprocessing.Pool(NB_PROCESS)
    pool.map(run_task, list_tasks)

    pool.close()
    pool.join()


if __name__ == "__main__":
    build_and_run_tasks()
