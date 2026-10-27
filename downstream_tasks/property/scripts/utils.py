
def set_seed(seed):
    import random
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def setup_logging(output_dir, mode):
    import logging
    import os
    os.makedirs(output_dir, exist_ok=True)
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        handlers=[
                            logging.FileHandler(os.path.join(output_dir, f'{mode}.log')),
                            logging.StreamHandler()
                        ])

def enforce_arg_types(args, parser):
    return args


