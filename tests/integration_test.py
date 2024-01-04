import pytest
from numpy import mean
from scipy.stats import t
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
import warnings
import time
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from ml_models.SklearnModel import SklearnModel
from utils.Evaluator import Evaluator
from utils.Win95_Dag import *
from utils.Postprocressing import Postprocessing
from sklearn.metrics import balanced_accuracy_score

warnings.simplefilter(action='ignore', category=FutureWarning)
starttime = time.time()

@pytest.fixture
def interworld_benchmarks():
    ds_model = get_printer()
    list_sklearn = [SklearnModel(f'{learner.__name__}', learner) for learner in[AdaBoostClassifier, RandomForestClassifier]]#, KNeighborsClassifier]], GradientBoostingClassifier]] #Additional NB classifiers GaussianNB, BernoulliNB, MultinomialNB, ComplementNB, CategoricalNB,
    list_sklearn.append(SklearnModel("MLPClassifier", MLPClassifier, solver='lbfgs', max_iter=1000, hidden_layer_sizes=(10, 2)))
    list_sklearn.append(SklearnModel("DecisionTreeClassifier_gini", DecisionTreeClassifier, criterion="gini"))
    list_sklearn.append(SklearnModel("DecisionTreeClassifier_entropy", DecisionTreeClassifier, criterion="entropy"))
    evaluator = Evaluator(ml_models=list_sklearn, dg_models=[], real_models=[ds_model],scores=[balanced_accuracy_score], outcome_name="Y")
    interworld_benchmarks = evaluator.meta_simulate(ds_model, n_learning=0,n_train=200,n_test=200,n_true_repetitions=50,n_practitioner_repititions=3,n_sl_repititions=50)
    return interworld_benchmarks

def test_perf_within_confidence_interval(interworld_benchmarks):
    list_of_ntrue_accuracies = interworld_benchmarks[0]
    list_of_npractitioner_accuracies = interworld_benchmarks[1]
    ml_models = interworld_benchmarks[7]
    confidence_level = 0.95
    for ml in ml_models:
        true_ml_mean = mean(list_of_ntrue_accuracies[ml.name])
        limited_ml_mean = mean(list_of_npractitioner_accuracies[ml.name])

        std_dev_true = np.std(list_of_ntrue_accuracies[ml.name], ddof=1)
        std_dev_limited = np.std(list_of_npractitioner_accuracies[ml.name], ddof=1)
        sem_true = std_dev_true / np.sqrt(len(list_of_ntrue_accuracies[ml.name]))
        sem_limited = std_dev_limited / np.sqrt(len(list_of_npractitioner_accuracies[ml.name]))
        margin_of_error_true = t.ppf((1 + confidence_level) / 2, len(list_of_ntrue_accuracies[ml.name]) - 1) * sem_true
        margin_of_error_limited = t.ppf((1 + confidence_level) / 2,len(list_of_npractitioner_accuracies[ml.name]) - 1) * sem_limited
        true_interval_lower, true_interval_upper = (true_ml_mean - margin_of_error_true, true_ml_mean + margin_of_error_true)
        limited_interval_lower, limited_interval_upper = (limited_ml_mean - margin_of_error_limited, limited_ml_mean + margin_of_error_limited)

        if true_interval_upper < limited_interval_lower or limited_interval_upper < true_interval_lower:
            assert False

if __name__ == "__main__":
    interworld_benchmarks()
