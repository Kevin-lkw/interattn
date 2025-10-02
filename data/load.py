from data.regular.parity import ParityCheck

def load_data(dataset_name, batch_size, length, randomize, device):
    if dataset_name == 'parity':
        dataset = ParityCheck(batch_size, length, randomize, device)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    return dataset