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
# pip install git+https://github.com/xunzheng/notears.git#egg=notears if notears not installed
from sklearn.tree import DecisionTreeClassifier

from dg_models.PgmpyLearner import PgmpyModel
from dg_models.NotearsLearner import NotearsLearner
from dg_models.DagsimModel import DagsimModel
from ml_models.SklearnModel import SklearnModel
import numpy as np
from dagsim.base import Graph, Node
from Evaluator import Evaluator
from utils.Dags import *
from utils.Postprocressing import Postprocessing
from sklearn.metrics import accuracy_score, balanced_accuracy_score, auc, average_precision_score

warnings.simplefilter(action='ignore', category=FutureWarning)

# ML and SL configuration for pipelines
list_pgmpy = [PgmpyModel(f'{learner.__name__}', learner, "Y") for learner in
              [HillClimbSearch, TreeSearch, MmhcEstimator]]  # , ExhaustiveSearch]]

no_tears_linear_default = NotearsLearner(name="notears_linear", SLClass="linear", loss_type='logistic', lambda1=0.01)

list_sklearn = [SklearnModel(f'{learner.__name__}', learner) for learner in
                [DecisionTreeClassifier, RandomForestClassifier, KNeighborsClassifier, GradientBoostingClassifier]] #Additional NB classifiers GaussianNB, BernoulliNB, MultinomialNB, ComplementNB, CategoricalNB,

list_sklearn.append(SklearnModel("SVCRbf", svm.SVC, kernel="rbf"))
list_sklearn.append(SklearnModel("SVCLinear", svm.SVC, kernel="linear"))
list_sklearn.append(SklearnModel("SVCSigmoid", svm.SVC, kernel="sigmoid"))
list_sklearn.append(SklearnModel("LogisticLASSO", LogisticRegression, penalty="l1", solver="liblinear"))
list_sklearn.append(SklearnModel("MLPClassifier", MLPClassifier, solver='lbfgs', max_iter=1000, hidden_layer_sizes=(5, 2)))

# DAG Example 0 - Dummy Example
#Prior1 = Node(name="A", function=np.random.binomial, kwargs={"n": 1, "p": 0.5}, size_field="size")
#Prior2 = Node(name="B", function=np.random.binomial, kwargs={"n": 1, "p": 0.2}, size_field="size")
#Prior3 = Node(name="C", function=np.random.binomial, kwargs={"n": 1, "p": 0.7}, size_field="size")
#Prior4 = Node(name="D", function=np.random.binomial, kwargs={"n": 1, "p": 0.5}, size_field="size")
#Node1 = Node(name="Y", function=log_transformation,
#             kwargs={"params0": Prior1, "params1": Prior2, "params2": Prior3, "params3": Prior4})

# DAG Example 1 - Asia Cancer/Dysponea Example

Prior1 = Node(name="Asia", function=get_asia)
Prior2 = Node(name="Tub", function=get_tub, args=[Prior1])
Prior3 = Node(name="Smoke", function=get_smoker_truth)
Prior4 = Node(name="Lung", function=get_lung_truth, args=[Prior3])
Prior5 = Node(name="Bronc", function=get_bronc_truth, args=[Prior3])
Prior6 = Node(name="Either", function=get_either_truth, args=[Prior2, Prior4])
Prior7 = Node(name="Xray", function=get_xray_truth, args=[Prior6])
Prior8 = Node(name="Y", function=get_dyspnoea_truth, args=[Prior5, Prior6])
# Y is the predicted node Dysponea

listNodes = [Prior1, Prior2, Prior3, Prior4, Prior5, Prior6, Prior7, Prior8]
asia = Graph(name="Asia Cancer Dysponea Example - Real-world", list_nodes=listNodes)
ds_model = DagsimModel("pipeline1", asia)

# DAG Example 2 - Windows 95 Printer Example

#Prior1 = Node(name="AppOK", function=get_AppOK)
#Prior2 = Node(name="DataFile", function=get_DataFile)
#Prior3 = Node(name="AppData", function=get_AppData, args=[Prior1, Prior2])
#Prior4 = Node(name="DskLocal", function=get_DskLocal)
#Prior5 = Node(name="PrtSpool", function=get_PrtSpool)
#Prior6 = Node(name="PrtOn", function=get_PrtOn)
#Prior7 = Node(name="PrtPaper", function=get_PrtPaper)
#Prior8 = Node(name="NetPrint", function=get_NetPrint)
#Prior9 = Node(name="PrtDriver", function=get_PrtDriver)
#Prior10 = Node(name="PrtThread", function=get_PrtThread)
#Prior11 = Node(name="EMFOK", function=get_EMFOK,args=[Prior3, Prior4, Prior10])
#Prior12 = Node(name="GDIIN", function=get_GDIIN,args=[Prior3, Prior5, Prior11])
#Prior13 = Node(name="DrvSet", function=get_DrvSet)
#Prior14 = Node(name="DrvOK", function=get_DrvOK)
#Prior15 = Node(name="GDIOUT", function=get_GDIOUT,args=[Prior9, Prior12, Prior13, Prior14])
#Prior16 = Node(name="PrtSel", function=get_PrtSel)
#Prior17 = Node(name="PrtDataOut", function=get_PrtDataOut,args=[Prior15, Prior16])
#Prior18 = Node(name="PrtPath", function=get_PrtPath)
#Prior19 = Node(name="NtwrkCnfg", function=get_NtwrkCnfg)
#Prior20 = Node(name="PTROFFLINE", function=get_PTROFFLINE)
#Prior21 = Node(name="NetOK", function=get_NetOK,args=[Prior18, Prior19, Prior20])
#Prior22 = Node(name="PrtCbl", function=get_PrtCbl)
#Prior23 = Node(name="PrtPort", function=get_PrtPort)
#Prior24 = Node(name="CblPrtHrdwrOK", function=get_CblPrtHrdwrOK)
#Prior25 = Node(name="LclOK", function=get_LclOK,args=[Prior22, Prior23, Prior24])
#Prior26 = Node(name="DSApplctn", function=get_DSApplctn)
#Prior27 = Node(name="PrtMpTPth", function=get_PrtMpTPth)
#Prior28 = Node(name="DS_NTOK", function=get_DS_NTOK,args=[Prior3, Prior18, Prior27, Prior19, Prior20])
#Prior29 = Node(name="DS_LCLOK", function=get_DS_LCLOK,args=[Prior3, Prior22, Prior23, Prior24])
#Prior30 = Node(name="PC2PRT", function=get_PC2PRT,args=[Prior8, Prior17, Prior21, Prior25, Prior26, Prior28, Prior29])
#Prior31 = Node(name="PrtMem", function=get_PrtMem)
#Prior32 = Node(name="PrtTimeOut", function=get_PrtTimeOut)
#Prior33 = Node(name="FllCrrptdBffr", function=get_FllCrrptdBffr)
#Prior34 = Node(name="TnrSpply", function=get_TnrSpply)
#Prior35 = Node(name="Y", function=get_PrtData,args=[Prior6, Prior7, Prior30, Prior31, Prior32, Prior33, Prior34])
#Prior36 = Node(name="Problem1", function=get_Problem1,args=[Prior35])
#Prior37 = Node(name="AppDtGnTm", function=get_AppDtGnTm,args=[Prior5])
#Prior38 = Node(name="PrntPrcssTm", function=get_PrntPrcssTm,args=[Prior5])
#Prior39 = Node(name="DeskPrntSpd", function=get_DeskPrntSpd,args=[Prior31, Prior37, Prior38])
#Prior40 = Node(name="PgOrnttnOK", function=get_PgOrnttnOK)
#Prior41 = Node(name="PrntngArOK", function=get_PrntngArOK)
#Prior42 = Node(name="ScrnFntNtPrntrFnt", function=get_ScrnFntNtPrntrFnt)
#Prior43 = Node(name="CmpltPgPrntd", function=get_CmpltPgPrntd,args=[Prior31, Prior40, Prior41])
#Prior44 = Node(name="GrphcsRltdDrvrSttngs", function=get_GrphcsRltdDrvrSttngs)
#Prior45 = Node(name="EPSGrphc", function=get_EPSGrphc)
#Prior46 = Node(name="NnPSGrphc", function=get_NnPSGrphc,args=[Prior31, Prior44, Prior45])
#Prior47 = Node(name="PrtPScript", function=get_PrtPScript)
#Prior48 = Node(name="PSGRAPHIC", function=get_PSGRAPHIC,args=[Prior31, Prior44, Prior45])
#Prior49 = Node(name="Problem4", function=get_Problem4,args=[Prior46, Prior47, Prior48])
#Prior50 = Node(name="TrTypFnts", function=get_TrTypFnts)
#Prior51 = Node(name="FntInstlltn", function=get_FntInstlltn)
#Prior52 = Node(name="PrntrAccptsTrtyp", function=get_PrntrAccptsTrtyp)
#Prior53 = Node(name="TTOK", function=get_TTOK,args=[Prior31, Prior51, Prior52])
#Prior54 = Node(name="NnTTOK", function=get_NnTTOK,args=[Prior31, Prior42, Prior51])
#Prior55 = Node(name="Problem5", function=get_Problem5,args=[Prior50, Prior53, Prior54])
#Prior56 = Node(name="LclGrbld", function=get_LclGrbld,args=[Prior3, Prior9, Prior31,Prior24])
#Prior57 = Node(name="NtGrbld", function=get_LclGrbld,args=[Prior3, Prior9, Prior31,Prior19])
#Prior58 = Node(name="GrbldOtpt", function=get_GrbldOtpt,args=[Prior8, Prior56, Prior57])
#Prior59 = Node(name="HrglssDrtnAftrPrnt", function=get_HrglssDrtnAftrPrnt,args=[Prior38])
#Prior60 = Node(name="REPEAT", function=get_REPEAT,args=[Prior24, Prior19])
#Prior61 = Node(name="AvlblVrtlMmry", function=get_AvlblVrtlMmry,args=[Prior47])
#Prior62 = Node(name="PSERRMEM", function=get_PSERRMEM,args=[Prior47, Prior61])
#Prior63 = Node(name="TstpsTxt", function=get_TstpsTxt,args=[Prior47, Prior61])
#Prior64 = Node(name="GrbldPS", function=get_GrbldPS,args=[Prior58, Prior61])
#Prior65 = Node(name="IncmpltPS", function=get_IncmpltPS,args=[Prior43, Prior61])
#Prior66 = Node(name="PrtFile", function=get_PrtFile,args=[Prior17])
#Prior67 = Node(name="PrtIcon", function=get_PrtIcon,args=[Prior19, Prior20])
#Prior68 = Node(name="Problem6", function=get_Problem6,args=[Prior58, Prior47, Prior64])
#Prior69 = Node(name="Problem3", function=get_Problem3,args=[Prior43, Prior47, Prior65])
#Prior70 = Node(name="PrtQueue", function=get_PrtQueue)
#Prior71 = Node(name="NtSpd", function=get_NtSpd,args=[Prior39, Prior19, Prior70])
#Prior72 = Node(name="Problem2", function=get_Problem2,args=[Prior8, Prior39, Prior71])
#Prior73 = Node(name="PrtStatPaper", function=get_PrtStatPaper,args=[Prior7])
#Prior74 = Node(name="PrtStatToner", function=get_PrtStatToner,args=[Prior34])
#Prior75 = Node(name="PrtStatMem", function=get_PrtStatMem,args=[Prior31])
#Prior76 = Node(name="PrtStatOff", function=get_PrtStatOff,args=[Prior6])
# Y is the predicted node PrtData

#listNodes = [Prior1, Prior2, Prior3, Prior4, Prior5, Prior6, Prior7, Prior8,Prior9, Prior10, Prior11,
#             Prior12, Prior13, Prior14, Prior15, Prior16, Prior17, Prior18, Prior19, Prior20,
#             Prior21, Prior22, Prior23, Prior24, Prior25, Prior26, Prior27, Prior28, Prior29, Prior30,
#             Prior31, Prior32, Prior33, Prior34, Prior35, Prior36, Prior37, Prior38, Prior39, Prior40,
#             Prior41, Prior42, Prior43, Prior44, Prior45, Prior46, Prior47, Prior48, Prior49, Prior50,
#             Prior51, Prior52, Prior53, Prior54, Prior55, Prior56, Prior57, Prior58, Prior59, Prior60,
#             Prior61, Prior62, Prior63, Prior64, Prior65, Prior66, Prior67, Prior68, Prior69, Prior70,
#             Prior71, Prior72, Prior73, Prior74, Prior75, Prior76]
#printer = Graph(name="Windows 95 Printer Example - Real-world", list_nodes=listNodes)
#ds_model = DagsimModel("pipeline1", printer)

evaluator = Evaluator(ml_models=list_sklearn, dg_models=[*list_pgmpy, no_tears_linear_default], real_models=[ds_model],
                      scores=[balanced_accuracy_score], outcome_name="Y")

pp = Postprocessing()

analysis0_results = evaluator.analysis_0_per_dg_model(dg_model_real=ds_model, n_repetitions=10000, n_samples=1000, tr_frac=0.5)
pp.plot_analysis0(analysis0_results)

analysis1_results = evaluator.analysis_1_per_dg_model(dg_model_real=ds_model, n_samples=10000, tr_frac=0.5, n_btstrps=20)
pp.plot_analysis1(analysis1_results)

#analysis3 = evaluator.analysis_3_per_dg_model(ds_model, n_learning=1000, n_train=1000,n_test=500)
#pp.plot_analysis3(analysis3_results=analysis3)

#analysis3b = evaluator.analysis_3b_per_dg_model(ds_model, n_samples=1000, tr_frac=0.5, n_reps=20)
#pp.plot_analysis3b(analysis3b)

# Extra analysis, DAG-benchmarking and violin plot
#analysis_coef = evaluator.analysis_coef_per_dg_model(ds_model)
#pp.plot_analysis_coef_gks(analysis_coef)
#analysis_violin_repeat = evaluator.analysis_violin_per_dg_model(ds_model, 200, 0.5, 10)
#pp.plot_analysis_violin(analysis_violin_repeat)
