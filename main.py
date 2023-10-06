import random
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import *
from sklearn.naive_bayes import BernoulliNB, GaussianNB, MultinomialNB, ComplementNB, CategoricalNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
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
warnings.simplefilter(action='ignore', category=FutureWarning)
starttime = time.time()

# To design a meta-simulation:
# 1) define the Bayesian network for the problem-at-hand in DagSim, specify the structure and parameters of the network
# 2) specify the ML estimators to perform benchmarking
# 3) specify the structural learners (SL) to handle real-world samples in the meta-simulation

# Structural Learner configuration
structural_learner_list = [Bnlearner(name="hc", SLClass="hc"), Bnlearner(name="tabu", SLClass="tabu"), Bnlearner(name="rsmax2", SLClass="rsmax2"), Bnlearner(name="mmhc", SLClass="mmhc"), Bnlearner(name="h2pc", SLClass="h2pc"), Bnlearner(name="gs", SLClass="gs"), Bnlearner(name="pc.stable", SLClass="pc.stable")] #, NotearsLearner(name="notears_linear", SLClass="notears_linear", loss_type='logistic', lambda1=0.01), Bnlearner(name="iamb", SLClass="iamb"), Bnlearner(name="fast.iamb", SLClass="fast.iamb"),Bnlearner(name="iamb.fdr", SLClass="iamb.fdr")]

# Machine-Learning Estimator configuration
list_sklearn = [SklearnModel(f'{learner.__name__}', learner) for learner in[DecisionTreeClassifier, RandomForestClassifier, KNeighborsClassifier, GradientBoostingClassifier]] #Additional NB classifiers GaussianNB, BernoulliNB, MultinomialNB, ComplementNB, CategoricalNB,
#list_sklearn.append(SklearnModel("SVCRbf", svm.SVC, kernel="rbf"))
#list_sklearn.append(SklearnModel("SVCSigmoid", svm.SVC, kernel="sigmoid"))

list_sklearn.append(SklearnModel("SVCLinear", svm.SVC, kernel="linear"))
list_sklearn.append(SklearnModel("LogisticLASSO", LogisticRegression, penalty="l1", solver="liblinear"))
list_sklearn.append(SklearnModel("MLPClassifier", MLPClassifier, solver='lbfgs', max_iter=1000, hidden_layer_sizes=(5, 2)))

#list_sklearn = [SklearnModel(f'{learner.__name__}', learner) for learner in[GaussianNB]]

# Bayesian Network configuration, included are examples from bnlearn's public repository
ds_model = get_printer()
#ds_model = get_asia()
#ds_model = get_andes()

evaluator = Evaluator(ml_models=list_sklearn, dg_models=structural_learner_list, real_models=[ds_model],scores=[balanced_accuracy_score], outcome_name="Y")
pp = Postprocessing()

# Provide the Bayesian network, train/test splits, and repetitions at different parts of the meta-simulation
interworld_benchmarks = evaluator.meta_simulate(ds_model, n_learning=0, n_train=200,n_test=200, n_true_repetitions=1000, n_practitioner_repititions=30, n_sl_repititions=500)
pp.meta_simulation_visualise(interworld_benchmarks)

endtime = time.time()
print("Time taken: ", endtime - starttime)

# Real-world benchmarking functions
#realworld_benchmarks = evaluator.realworld_benchmark(dg_model_real=ds_model, n_repetitions=100, n_samples=2000, tr_frac=0.5)
#pp.realworld_benchmarks_visualise(realworld_benchmarks)

#realworld_benchmarks_bootstrapped = evaluator.realworld_benchmark_bootstrapping(dg_model_real=ds_model, n_samples=100, tr_frac=0.5, n_btstrps=10)
#pp.realworld_benchmarks_bootstrapping_visualise(realworld_benchmarks_bootstrapped)