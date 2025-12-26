import numpy as np
from src.entities.task import Task
from src import hp

class WorkloadGenerator:
    def __init__(self, workload_config, service_config, terminals):
        self.arrival_rate = workload_config['arrival_rate']
        self.zipf_param = workload_config['zipf_param']
        self.terminals = terminals
        self.services = service_config
        
        # Zipf Probabilities
        ranks = np.arange(1, len(self.services) + 1)
        weights = ranks ** (-self.zipf_param)
        self.service_probs = weights / weights.sum()

    def generate(self, current_time_slot)->list[Task]:
        num_tasks = np.random.poisson(self.arrival_rate)
        generated_tasks: list[Task] = []
        
        for _ in range(num_tasks):
            if not self.terminals: break
            
            # 1. Terminal & Service
            term_id = np.random.choice(self.terminals)
            svc_idx = np.random.choice(len(self.services), p=self.service_probs)# required service
            svc_profile = self.services[svc_idx]
            
            # 2. generate random Min Accuracy 
            # acc in [Min_Model_Acc, Max_Model_Acc] 
            available_accs = [m['accuracy'] for m in svc_profile['models']]
            min_possible = min(available_accs)
            max_possible = max(available_accs)
            
            req_acc = np.random.uniform(min_possible, max_possible)

            # batch_size
            random_batch_size = np.random.randint(hp.MIN_BATCH_SIZE, hp.MAX_BATCH_SIZE) 
            # 3. Tạo Task Request
            task = Task(
                task_id=f"T{current_time_slot}_{np.random.randint(100000)}",
                service_id=svc_profile['id'],
                terminal_id=term_id,
                unit_size=svc_profile['input_data_size'],
                batch_size= random_batch_size, 
                deadline=svc_profile['deadline'],
                min_accuracy=req_acc, 
                omega=svc_profile['omega'],
                created_at=current_time_slot
            )
            generated_tasks.append(task)
            
        return generated_tasks