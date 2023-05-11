import random
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import *
from sklearn.naive_bayes import BernoulliNB, GaussianNB, MultinomialNB, ComplementNB, CategoricalNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
import warnings
from sklearn.tree import DecisionTreeClassifier
from dg_models.BnlearnLearner import Bnlearner
from ml_models.SklearnModel import SklearnModel
from Evaluator import Evaluator
from utils.Win95_Dag import *
from utils.Andes_Dag import *
from utils.Postprocressing import Postprocessing
from sklearn.metrics import accuracy_score, balanced_accuracy_score, auc, average_precision_score
warnings.simplefilter(action='ignore', category=FutureWarning)

# ML pipeline configuration
structural_learner_list = [Bnlearner(name="hc", SLClass="hc"), Bnlearner(name="tabu", SLClass="tabu"), Bnlearner(name="rsmax2", SLClass="rsmax2"), Bnlearner(name="mmhc", SLClass="mmhc"), Bnlearner(name="h2pc", SLClass="h2pc"), Bnlearner(name="gs", SLClass="gs"), Bnlearner(name="pc.stable", SLClass="pc.stable")] #, NotearsLearner(name="notears_linear", SLClass="notears_linear", loss_type='logistic', lambda1=0.01), Bnlearner(name="iamb", SLClass="iamb"), Bnlearner(name="fast.iamb", SLClass="fast.iamb"),Bnlearner(name="iamb.fdr", SLClass="iamb.fdr")]

list_sklearn = [SklearnModel(f'{learner.__name__}', learner) for learner in[DecisionTreeClassifier, RandomForestClassifier, KNeighborsClassifier, GradientBoostingClassifier]] #Additional NB classifiers GaussianNB, BernoulliNB, MultinomialNB, ComplementNB, CategoricalNB,
#list_sklearn.append(SklearnModel("SVCRbf", svm.SVC, kernel="rbf"))
list_sklearn.append(SklearnModel("SVCLinear", svm.SVC, kernel="linear"))
#list_sklearn.append(SklearnModel("SVCSigmoid", svm.SVC, kernel="sigmoid"))
list_sklearn.append(SklearnModel("LogisticLASSO", LogisticRegression, penalty="l1", solver="liblinear"))
list_sklearn.append(SklearnModel("MLPClassifier", MLPClassifier, solver='lbfgs', max_iter=1000, hidden_layer_sizes=(5, 2)))

# SL world configuration
ds_model = get_printer()
#ds_model = get_andes()

evaluator = Evaluator(ml_models=list_sklearn, dg_models=structural_learner_list, real_models=[ds_model],scores=[balanced_accuracy_score], outcome_name="Y")
pp = Postprocessing()

#analysis0_results = evaluator.analysis_0_per_dg_model(dg_model_real=ds_model, n_repetitions=100, n_samples=2000, tr_frac=0.5)
#pp.plot_analysis0(analysis0_results)

#analysis1_results = evaluator.analysis_1_per_dg_model(dg_model_real=ds_model, n_samples=100, tr_frac=0.5, n_btstrps=10)
#pp.plot_analysis1(analysis1_results)

analysis3 = evaluator.analysis_3_per_dg_model(ds_model, n_learning=0, n_train=1000,n_test=1000, n_true_repetitions=10, n_practitioner_repititions=10, n_sl_repititions=10)
pp.plot_analysis3(analysis3)

# Extra analysis, DAG-benchmarking, scatterplots and violin plot
#analysis2 = evaluator.analysis_2_per_dg_model(ds_model, n_learning=100, n_train=1000,n_test=1000, n_repetitions=1000)
#pp.plot_analysis2(analysis2)
#analysis2b = evaluator.analysis_2b_per_dg_model(ds_model, n_samples=1000, tr_frac=0.5,n_reps=20)
#pp.plot_analysis2b(analysis2b)
#analysis_coef = evaluator.analysis_coef_per_dg_model(ds_model)
#pp.plot_analysis_coef_gks(analysis_coef)
#analysis_violin_repeat = evaluator.analysis_violin_per_dg_model(ds_model, 200, 0.5, 10)
#pp.plot_analysis_violin(analysis_violin_repeat)
