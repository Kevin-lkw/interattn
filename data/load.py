from data.regular.parity import ParityCheck
from data.retrieval.task_copy import Copy
from data.retrieval.task_mqar import mqar

def load_data(dataset_name, config, device):
    if dataset_name == 'parity':
        dataset = ParityCheck(device)
    elif dataset_name == 'copy':
        dataset = Copy(device)
    elif dataset_name == 'MQAR':
        # randomize is set to False for MQAR task
        dataset = mqar(device=device, 
                num_kv_pairs= config.num_kv_pair, power_a=config.power_a, random_non_queries = config.random_non_queries)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    return dataset