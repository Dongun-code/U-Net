import tensorflow as tf

class Train:
    def __init__(self, model, loss_function, optimizer, epcohs):
        self.model = model
        self.loss_fucntion = loss_function
        self.optimizer = optimizer
        self.epoch = epcohs
