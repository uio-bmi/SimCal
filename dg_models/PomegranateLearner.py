from dg_models.DGModel import DGModel
from pomegranate.BayesianNetwork import BayesianNetwork
from utils.Data import Data
from abc import ABC, abstractmethod


class PomegranateLearner(DGModel):
    def __init__(self, name, algorithm: str = "greedy", **kwargs):
        super().__init__(name=name, SLClass=BayesianNetwork, **kwargs)
        self.algorithm = algorithm

    def instantiate(self):
        self.model = self.SLClass()

    def fit(self, data, **kwargs):
        data = data.all
        self.model = self.model.from_samples(X=data, algorithm=self.algorithm, state_names=data.columns.values)
        self.model = self.model.fit(X=data)

    def _generate(self, num_samples, outcome_name: str, world: str = "real", dataset: str = "train"):
        data = self.model.sample(n=num_samples)
        data = Data(name=world + self.name + dataset, X=data.drop(outcome_name, axis=1),
                    y=data.loc[:, outcome_name])
        return data


class SomeAbstractClass(ABC):
    @abstractmethod
    def get(self, *args, **kwargs):
        """
        Returns smth
        """

    @abstractmethod
    def set(self, key, value):
        """
        Sets smth
        """


class Implementation(SomeAbstractClass):
    def set(self, key, value):
        pass

    def get(self, some_var, another_one):
        pass
# if __name__ == "__main__":
#
