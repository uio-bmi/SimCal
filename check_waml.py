import random

from sklearn import svm
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import *
from sklearn.naive_bayes import BernoulliNB, GaussianNB, MultinomialNB, ComplementNB, CategoricalNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import linear_model
from pgmpy.estimators import *
import warnings

from sklearn.tree import DecisionTreeClassifier

from dg_models.PgmpyLearner import PgmpyModel
from dg_models.NotearsLearner import NotearsLearner
from dg_models.DagsimModel import DagsimModel
from ml_models.SklearnModel import SklearnModel
import numpy as np
from dagsim.base import Graph, Node
from Evaluator import Evaluator
from utils.Postprocressing import Postprocessing
from sklearn.metrics import accuracy_score, balanced_accuracy_score, auc, average_precision_score

warnings.simplefilter(action='ignore', category=FutureWarning)

# ML and SL configuration for pipelines
list_pgmpy = [PgmpyModel(f'{learner.__name__}', learner, "Y") for learner in
              [HillClimbSearch, TreeSearch, MmhcEstimator]]  # , ExhaustiveSearch]]

no_tears_linear_default = NotearsLearner(name="notears_linear", SLClass="linear", loss_type='logistic', lambda1=0.01)

list_sklearn = [SklearnModel(f'{learner.__name__}', learner) for learner in
                [DecisionTreeClassifier, RandomForestClassifier, GaussianNB, BernoulliNB, MultinomialNB, ComplementNB, CategoricalNB, KNeighborsClassifier, GradientBoostingClassifier]]
list_sklearn.append(SklearnModel("SVCRbf", svm.SVC, kernel="rbf"))
list_sklearn.append(SklearnModel("SVCLinear", svm.SVC, kernel="linear"))
list_sklearn.append(SklearnModel("SVCSigmoid", svm.SVC, kernel="sigmoid"))
list_sklearn.append(SklearnModel("LogisticLASSO", LogisticRegression, penalty="l1", solver="liblinear"))
list_sklearn.append(SklearnModel("MLPClassifier", MLPClassifier, solver='lbfgs', max_iter=1000, hidden_layer_sizes=(5, 2)))

# Ground truth DAG for pretended real-world
def log_transformation(params0, params1, params2, params3):
    sum = params0 * 2 + params1 - params2 + params3 + random.randint(0, 1)
    y = 1 / (1 + np.exp(-sum))
    y = 1 if y > 0.75 else 0
    return y

def get_asia():
    if np.random.binomial(n=1, p=0.01):
        return 0
    else:
        return 1

def get_tub(asia):
    if asia == 0:
        if np.random.binomial(n=1, p=0.05):
            return 0
        else:
            return 1
    elif asia == 1:
        if np.random.binomial(n=1, p=0.01):
            return 0
        else:
            return 1

def get_smoker_truth():
    if np.random.binomial(n=1, p=0.5):
        return 0
    else:
        return 1

def get_lung_truth(smoke):
    if smoke == 0:
        if np.random.binomial(n=1, p=0.1):
            return 0
        else:
            return 1
    elif smoke == 1:
        if np.random.binomial(n=1, p=0.01):
            return 0
        else:
            return 1

def get_bronc_truth(smoke):
    if smoke == 0:
        if np.random.binomial(n=1, p=0.6):
            return 0
        else:
            return 1
    elif smoke == 1:
        if np.random.binomial(n=1, p=0.3):
            return 0
        else:
            return 1

def get_either_truth(lung, tub):
    if lung == 0 or tub == 0:
            return 0
    else:
        return 1

def get_xray_truth(either):
    if either == 0:
        if np.random.binomial(n=1, p=0.98):
            return 0
        else:
            return 1
    elif either == 1:
        if np.random.binomial(n=1, p=0.05):
            return 0
        else:
            return 1

def get_dyspnoea_truth(bronc, either):
    if bronc == 0 and either == 0:
        if np.random.binomial(n=1, p=0.9):
            return 0
        else:
            return 1
    elif bronc == 1 and either == 0:
        if np.random.binomial(n=1, p=0.7):
            return 0
        else:
            return 1
    elif bronc == 0 and either == 1:
        if np.random.binomial(n=1, p=0.8):
            return 0
        else:
            return 1
    elif bronc == 1 and either == 1:
        if np.random.binomial(n=1, p=0.1):
            return 0
        else:
            return 1

#Prior1 = Node(name="A", function=np.random.binomial, kwargs={"n": 1, "p": 0.5}, size_field="size")
#Prior2 = Node(name="B", function=np.random.binomial, kwargs={"n": 1, "p": 0.2}, size_field="size")
#Prior3 = Node(name="C", function=np.random.binomial, kwargs={"n": 1, "p": 0.7}, size_field="size")
#Prior4 = Node(name="D", function=np.random.binomial, kwargs={"n": 1, "p": 0.5}, size_field="size")
#Node1 = Node(name="Y", function=log_transformation_nonlinear,
#             kwargs={"params0": Prior1, "params1": Prior2, "params2": Prior3, "params3": Prior4})
#Node1 = Node(name="Y", function=log_transformation,
#             kwargs={"params0": Prior1, "params1": Prior2, "params2": Prior3, "params3": Prior4})

Prior1 = Node(name="Asia", function=get_asia)
Prior2 = Node(name="Tub", function=get_tub, args=[Prior1])
Prior3 = Node(name="Smoke", function=get_smoker_truth)
Prior4 = Node(name="Lung", function=get_lung_truth, args=[Prior3])
Prior5 = Node(name="Bronc", function=get_bronc_truth, args=[Prior3])
Prior6 = Node(name="Either", function=get_either_truth, args=[Prior2, Prior4])
Prior7 = Node(name="Xray", function=get_xray_truth, args=[Prior6])
Prior8 = Node(name="Y", function=get_dyspnoea_truth, args=[Prior5, Prior6])

listNodes = [Prior1, Prior2, Prior3, Prior4, Prior5, Prior6, Prior7, Prior8]
my_graph = Graph(name="Dysponea Example - Real-world", list_nodes=listNodes)

ds_model = DagsimModel("pipeline1", my_graph)
evaluator = Evaluator(ml_models=list_sklearn, dg_models=[*list_pgmpy, no_tears_linear_default], real_models=[ds_model],
                      scores=[balanced_accuracy_score], outcome_name="Y")

pp = Postprocessing()

#analysis1_results = evaluator.analysis_1_per_dg_model(dg_model_real=ds_model, n_samples=1000000, tr_frac=0.002, n_btstrps=20)
#pp.plot_analysis1(analysis1_results)

#analysis3 = evaluator.analysis_3_per_dg_model(ds_model, n_learning=10000, n_train=10000,n_test=1000)
#pp.plot_analysis3(analysis3_results=analysis3)

analysis4 = evaluator.analysis_4_per_dg_model(ds_model, n_samples=10000, tr_frac=0.002, n_reps=20)
pp.plot_analysis4(analysis4)

# Extra analysis, DAG-benchmarking and violin plot
#analysis_coef = evaluator.analysis_coef_per_dg_model(ds_model)
#pp.plot_analysis_coef_gks(analysis_coef)
#analysis_violin_repeat = evaluator.analysis_violin_per_dg_model(ds_model, 200, 0.5, 10)
#pp.plot_analysis_violin(analysis_violin_repeat)
