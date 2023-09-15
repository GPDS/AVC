from class_based_network import Rede

def runner():
    rede = Rede(run_alt = True)
    rede.set_executions_per_trials(number=10)
    rede.set_max_trials(number=10)
    rede.set_epochs(number = 80)
    rede.set_epochs_search(number = 10)
    rede.run_model_n_times(number = 10,save_results=True, file_name="results_haralick.txt")
    rede.print_parameters()

if __name__ == "__main__":
    runner()