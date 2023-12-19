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
from dg_models.Bnlearner import Bnlearner
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

########################################################################################################################
# To run a meta-simulation in SimCal, the user needs to specify three elements
#
# 1) Configure the ground truth DAG of the real-world, specifying the structure and parameters of the Bayesian Network
# 2) Configure the Machine Learning Estimators used for benchmarking and their hyper-parameters
# 3) Configure the Structural Learners used to learn the underlying distribution (i.e., estimate DAGs) from limited data
########################################################################################################################

########################################################################################################################
# Bayesian Network configuration, specify a custom network or import an existing network (www.bnlearn.com/bnrepository/)
# For example:
# ds_model = get_asia()
# ds_model = get_printer()
########################################################################################################################
ds_model = get_printer()

########################################################################################################################
# Machine Learning configuration, select and specialise the algorithm used a custom network or import an existing network (www.bnlearn.com/bnrepository/)
# For example:
# list_sklearn = []
# list_sklearn.append(SklearnModel("SVCSigmoid", svm.SVC, kernel="sigmoid"))
# list_sklearn.append(SklearnModel("GradientBoostingClassifier_logloss", GradientBoostingClassifier, loss="log_loss"))
# list_sklearn.append(SklearnModel("RandomForestClassifier_entropy", RandomForestClassifier, criterion="entropy"))
# list_sklearn.append(SklearnModel("LogisticLASSO", LogisticRegression, penalty="l1", solver="liblinear"))
########################################################################################################################
list_sklearn = [SklearnModel(f'{learner.__name__}', learner) for learner in[AdaBoostClassifier, RandomForestClassifier]]
list_sklearn.append(SklearnModel("MLPClassifier", MLPClassifier, solver='lbfgs', max_iter=1000, hidden_layer_sizes=(10, 2)))
list_sklearn.append(SklearnModel("DecisionTreeClassifier_gini", DecisionTreeClassifier, criterion="gini"))
list_sklearn.append(SklearnModel("DecisionTreeClassifier_entropy", DecisionTreeClassifier, criterion="entropy"))

########################################################################################################################
# Structural Learner configuration, define and configure the learning algorithms used to estimate DAGs from limited data
# For example:
# structural_learner_list = []
# structural_learner_list.append(Bnlearner(name="hc", SLClass="hc"))
# structural_learner_list.append(Bnlearner(name="iamb", SLClass="iamb"))
# structural_learner_list.append(NotearsLearner(name="notears_linear", SLClass="notears_linear", loss_type='logistic', lambda1=0.01))
########################################################################################################################
structural_learner_list = [Bnlearner(name="hc", SLClass="hc"), Bnlearner(name="tabu", SLClass="tabu"), Bnlearner(name="rsmax2", SLClass="rsmax2"), Bnlearner(name="mmhc", SLClass="mmhc"), Bnlearner(name="h2pc", SLClass="h2pc"), Bnlearner(name="gs", SLClass="gs"), Bnlearner(name="pc.stable", SLClass="pc.stable")]

evaluator = Evaluator(ml_models=list_sklearn, dg_models=structural_learner_list, real_models=[ds_model],scores=[balanced_accuracy_score], outcome_name="Y")
interworld_benchmarks = evaluator.meta_simulate(ds_model, n_learning=0, n_train=200,n_test=200, n_true_repetitions=1000, n_practitioner_repititions=30, n_sl_repititions=500)
pp = Postprocessing()
pp.meta_simulation_visualise(interworld_benchmarks)
endtime = time.time()
print("Time taken: ", endtime - starttime)

