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
from dg_models.BnlearnLearner import Bnlearner
from ml_models.SklearnModel import SklearnModel
from Evaluator import Evaluator
from utils.Win95_Dag import *
from utils.Andes_Dag import *
from utils.Postprocressing import Postprocessing
from sklearn.metrics import accuracy_score, balanced_accuracy_score, auc, average_precision_score
from matplotlib import pyplot as plt
import seaborn as sns
import os
import plotly.express as px
import numpy as np
warnings.simplefilter(action='ignore', category=FutureWarning)
starttime = time.time()


ds_model = get_andes()

########################################################################################################################
# Machine Learning configuration, select and specialise the algorithm used  a custom network or import an existing network (www.bnlearn.com/bnrepository/)
# For example:
# ds_model = get_asia()
# ds_model = get_printer()
########################################################################################################################

# Machine-Learning Estimator configuration
list_sklearn = [SklearnModel(f'{learner.__name__}', learner) for learner in[AdaBoostClassifier, RandomForestClassifier]]#, KNeighborsClassifier]], GradientBoostingClassifier]] #Additional NB classifiers GaussianNB, BernoulliNB, MultinomialNB, ComplementNB, CategoricalNB,
#list_sklearn.append(SklearnModel("SVCRbf", svm.SVC, kernel="rbf"))
#list_sklearn.append(SklearnModel("SVCSigmoid", svm.SVC, kernel="sigmoid"))

#list_sklearn.append(SklearnModel("SVCLinear", svm.SVC, kernel="linear"))
#list_sklearn.append(SklearnModel("KNeighborsClassifier_uniform", KNeighborsClassifier, weights="uniform"))
#list_sklearn.append(SklearnModel("KNeighborsClassifier_distance", KNeighborsClassifier, weights="distance"))
#list_sklearn.append(SklearnModel("GradientBoostingClassifier_logloss", GradientBoostingClassifier, loss="log_loss"))
#list_sklearn.append(SklearnModel("GradientBoostingClassifier_exponential", GradientBoostingClassifier, loss="exponential"))
#list_sklearn.append(SklearnModel("RandomForestClassifier_gini", RandomForestClassifier, criterion="gini"))
#list_sklearn.append(SklearnModel("RandomForestClassifier_entropy", RandomForestClassifier, criterion="entropy"))
#list_sklearn.append(SklearnModel("LogisticLASSO", LogisticRegression, penalty="l1", solver="liblinear"))
#list_sklearn.append(SklearnModel("SVCSigmoid", svm.SVC, kernel="sigmoid"))
list_sklearn.append(SklearnModel("MLPClassifier", MLPClassifier, solver='lbfgs', max_iter=1000, hidden_layer_sizes=(10, 2)))
list_sklearn.append(SklearnModel("DecisionTreeClassifier_gini", DecisionTreeClassifier, criterion="gini"))
list_sklearn.append(SklearnModel("DecisionTreeClassifier_entropy", DecisionTreeClassifier, criterion="entropy"))

# Structural Learner configuration
structural_learner_list = [Bnlearner(name="hc", SLClass="hc"), Bnlearner(name="tabu", SLClass="tabu"), Bnlearner(name="rsmax2", SLClass="rsmax2"), Bnlearner(name="mmhc", SLClass="mmhc"), Bnlearner(name="h2pc", SLClass="h2pc"), Bnlearner(name="gs", SLClass="gs"), Bnlearner(name="pc.stable", SLClass="pc.stable")] #, NotearsLearner(name="notears_linear", SLClass="notears_linear", loss_type='logistic', lambda1=0.01), Bnlearner(name="iamb", SLClass="iamb"), Bnlearner(name="fast.iamb", SLClass="fast.iamb"),Bnlearner(name="iamb.fdr", SLClass="iamb.fdr")]

def scorer1(predicted, true):
    return np.random.normal(0.5, 0.1)


evaluator = Evaluator(ml_models=list_sklearn, dg_models=structural_learner_list, real_models=[ds_model],
                      scores=[balanced_accuracy_score], outcome_name="Y")
pp = Postprocessing()

# Provide the Bayesian network, train/test splits, and repetitions at different parts of the meta-simulation
interworld_benchmarks = evaluator.meta_simulate(ds_model, n_learning=0, n_train=200,n_test=200, n_true_repetitions=1000, n_practitioner_repititions=30, n_sl_repititions=500)
pp.meta_simulation_visualise(interworld_benchmarks)

endtime = time.time()
print("Time taken: ", endtime - starttime)
