import tsplib95
from QLSA import runAlgo, TestsFilePath
from compute import generate_tsp
import multiprocessing
import os
import glob

file_lock = multiprocessing.Lock()

OUTPUT_FOLDER = "results"
OUTPUT_FILE = f"{OUTPUT_FOLDER}/optimals.csv"
NB_RUNS = 30
NB_PROCESS = 3

ALGO_MAPPING = {
    1 : "QLSA softmax",
    2: "SA",
    3: "QLSA epsilon_greedy"
}

class Task:

    def __init__(self, problem, run_number, algo):
        self.problem = problem
        self.run_number = run_number
        self.algo = algo
        self.task_id = f"{problem}_{ALGO_MAPPING[self.algo]}_{run_number}"

    def run(self):
        problem = tsplib95.load_problem(f"{TestsFilePath}/{self.problem}.tsp")
        initial_solution = generate_tsp(1, problem.dimension)[0]
        res = runAlgo([self.algo, problem, initial_solution])
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
    try:
        t.run()
    except Exception as e:
        print(t.task_id, "Failed With exception", e)

def clean_up_output():
    if os.path.exists(OUTPUT_FILE):
        os.remove(OUTPUT_FILE)

def get_list_problems(input_folder):
    list_inputs = glob.glob(f"{input_folder}/*.tsp")
    list_inputs = [r.replace(input_folder, "").replace(".tsp", "") for r in list_inputs]
    return list_inputs

def build_and_run_tasks():

    clean_up_output()

    list_problems = get_list_problems(TestsFilePath)
    nb_runs = NB_RUNS
    algos = range(1, 4)

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
