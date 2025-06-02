from abc import ABC


class BaseAgent(ABC):

    def __init__(self):
        pass

    def init_hidden_states(self, **kwargs):
        raise NotImplementedError

    def sample(self, **kwargs):
        raise NotImplementedError

    def predict(self, **kwargs):
        raise NotImplementedError

    def learn(self, **kwargs):
        raise NotImplementedError

    def save_model(self, **kwargs):
        raise NotImplementedError

    def load_model(self, **kwargs):
        raise NotImplementedError