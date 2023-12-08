import random
import pandas as pd
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import *
from sklearn.naive_bayes import BernoulliNB, GaussianNB, MultinomialNB, ComplementNB, CategoricalNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
import warnings
import time
from sklearn.tree import DecisionTreeClassifier
from ml_models.SklearnModel import SklearnModel
from Evaluator import Evaluator
from utils.Win95_Dag import *
from utils.Andes_Dag import *
from utils.Postprocressing import Postprocessing
from sklearn.metrics import accuracy_score, balanced_accuracy_score, auc, average_precision_score
from matplotlib import pyplot as plt
import seaborn as sns
import os
import numpy as np
warnings.simplefilter(action='ignore', category=FutureWarning)
starttime = time.time()


def test_no_bias_in_limited_real():
    ds_model = get_printer()
    list_sklearn = [SklearnModel(f'{learner.__name__}', learner) for learner in[AdaBoostClassifier, RandomForestClassifier]]#, KNeighborsClassifier]], GradientBoostingClassifier]] #Additional NB classifiers GaussianNB, BernoulliNB, MultinomialNB, ComplementNB, CategoricalNB,
    list_sklearn.append(SklearnModel("MLPClassifier", MLPClassifier, solver='lbfgs', max_iter=1000, hidden_layer_sizes=(10, 2)))
    list_sklearn.append(SklearnModel("DecisionTreeClassifier_gini", DecisionTreeClassifier, criterion="gini"))
    list_sklearn.append(SklearnModel("DecisionTreeClassifier_entropy", DecisionTreeClassifier, criterion="entropy"))

    # Structural Learner configuration
    structural_learner_list = []
    evaluator = Evaluator(ml_models=list_sklearn, dg_models=structural_learner_list, real_models=[ds_model],
                          scores=[balanced_accuracy_score], outcome_name="Y")
    pp = Postprocessing()

    # Provide the Bayesian network, train/test splits, and repetitions at different parts of the meta-simulation
    interworld_benchmarks = evaluator.meta_simulate(ds_model, n_learning=0,
                                                    n_train=200,
                                                    n_test=200,
                                                    n_true_repetitions=50,
                                                    n_practitioner_repititions=15,
                                                    n_sl_repititions=100)
    pp.meta_simulation_visualise(interworld_benchmarks)
    endtime = time.time()

    #todo: Get the raw scores, check that they make statistically sense
    print("Time taken: ", endtime - starttime)


if __name__ == "__main__":
    test_no_bias_in_limited_real()
