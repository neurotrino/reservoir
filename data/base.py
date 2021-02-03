import tensorflow as tf

class BaseDataGenerator:
    def __init__(self, cfg):
        self.cfg = cfg
        self.dataset = None

    def next_batch(self, batch_size):
        raise NotImplementedError(
            "Data generator has no custom iterator:"
            + " iterate over the dataset attribute directly"
        )
        yield batch_x, batch_y
