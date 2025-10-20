from data.regular.parity import ParityCheck
from data.retrieval.copy import Copy
from data.retrieval.MQAR import MQAR

def load_data(dataset_name, batch_size, config, device):
    length = config.training_length
    randomize = config.randomize
    if dataset_name == 'parity':
        dataset = ParityCheck(batch_size, length, randomize, device)
    elif dataset_name == 'copy':
        dataset = Copy(batch_size, length, randomize, device)
    elif dataset_name == 'MQAR':
        # randomize is set to False for MQAR task
        assert randomize == False
        dataset = MQAR(batch_size, length, randomize, device=device, 
                       num_kv_pairs= config.num_kv_pair, power_a=config.power_a, random_non_queries = config.random_non_queries)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    return dataset