import pandas as pd
from typing import List, Callable

from numpy import mean

from dg_models.DGModel import DGModel
from ml_models.MachineLearner import MachineLearner
from utils.Data import Data
import numpy as np
import rpy2
import rpy2.robjects as robjects
import rpy2.robjects.packages as rpackages
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter
from rpy2.robjects.vectors import DataFrame, StrVector
from rpy2.robjects.packages import importr
pandas2ri.activate()

class Evaluator:
    def __init__(self, ml_models: List[MachineLearner], dg_models: List[DGModel], real_models: List[DGModel],
                 scores: List[Callable], outcome_name: str = "Y"):
        self.scores = scores
        self.real_models = real_models
        self.ml_models = ml_models
        self.dg_models = dg_models
        self.outcome_name = outcome_name

    def analysis_0_per_dg_model(self, dg_model_real: DGModel, n_repetitions, n_samples: int, tr_frac: float):
        """

        :param dg_model_real: a (pretended) real-world DagSim model
        :n_repetitions: number of times to repeat training-testing
        :param n_samples: number of samples to generate for training + testing
        :param tr_frac: fraction of data for training
        :param n_btstrps: numer of times to  perform bootstrap
        :return: btstrp_results: dict of shape {model_name: {score_name: list_of_score_values], ...}, ...}
        """
        results = pd.DataFrame(data=[[[] for _ in range(len(self.ml_models))] for __ in range(len(self.scores))],
                                      index=[sc.__name__ for sc in self.scores],
                                      columns=[md.name for md in self.ml_models])
        for _ in range(n_repetitions):
            orig_train_data, test_data = self._get_train_and_test_from_dg(dg_model_real, n_samples, tr_frac)
            assert len(orig_train_data) + len(test_data) == n_samples

            for ml_model in self.ml_models:
                result_per_repetition_per_model = self._develop_ml_model(ml_model, orig_train_data, test_data)
                for score_name in results.index.values.tolist():
                    results[ml_model.name][score_name].append(result_per_repetition_per_model[score_name])

        difference_to_average_by_ml = {}
        for ml_model in self.ml_models:
            average_performance = np.mean(results[ml_model.name][score_name])
            list_of_difference_to_average = []
            for num_repititon in range(n_repetitions):
                difference_to_average = abs(results[ml_model.name][score_name][num_repititon] - average_performance)
                list_of_difference_to_average.append(difference_to_average)
            difference_to_average_by_ml[ml_model.name] = list_of_difference_to_average
        pd.DataFrame(difference_to_average_by_ml).to_csv("analysis_0_benchmarking_infinite_relevancy.csv") #store every repitition difference to true performance by ML
        return results

    def analysis_1_per_dg_model(self, dg_model_real: DGModel, n_samples: int, tr_frac: float, n_btstrps: int):
        """

        :param dg_model_real: a (pretended) real-world DagSim model
        :param n_samples: number of samples to generate for training + testing
        :param tr_frac: fraction of data for training
        :param n_btstrps: numer of times to  perform bootstrap
        :return: btstrp_results: dict of shape {model_name: {score_name: list_of_score_values], ...}, ...}
        """
        orig_train_data, test_data = self._get_train_and_test_from_dg(dg_model_real, n_samples, tr_frac)

        assert len(orig_train_data) + len(test_data) == n_samples

        btstrp_results = pd.DataFrame(data=[[[] for _ in range(len(self.ml_models))] for __ in range(len(self.scores))],
                                      index=[sc.__name__ for sc in self.scores],
                                      columns=[md.name for md in self.ml_models])
        for _ in range(n_btstrps):
            btstr_tr = orig_train_data.bootstrap()
            for ml_model in self.ml_models:
                result_per_btsrtp_per_model = self._develop_ml_model(ml_model, btstr_tr, test_data)
                for score_name in btstrp_results.index.values.tolist():
                    btstrp_results[ml_model.name][score_name].append(result_per_btsrtp_per_model[score_name])

        #difference_to_average_by_ml = {}
        #for ml_model in self.ml_models:
        #    average_performance = np.mean(btstrp_results[ml_model.name][score_name])
        #    list_of_difference_to_average = []
        #    for num_btr in range(n_btstrps):
        #        difference_to_average = abs(btstrp_results[ml_model.name][score_name][num_btr] - average_performance)
        #        list_of_difference_to_average.append(difference_to_average)
        #    difference_to_average_by_ml[ml_model.name] = list_of_difference_to_average
        #pd.DataFrame(difference_to_average_by_ml).to_csv("analysis_1_benchmarking_limited_relevancy.csv")
        return btstrp_results

    def analysis_2_per_dg_model(self, dg_model_real: DGModel, n_learning: int, n_train: int,n_test: int, n_repetitions):
        list_of_results = {}
        metrics = {}
        dg_metrics, learning_data_real, test_data = self._get_performance_by_repetition(dg_model_real, n_train+n_test,0.5, n_repetitions)
        #list_of_results["Real-world"] = dg_metrics
        for dg_model in self.dg_models:
            repetition_results = pd.DataFrame(data=[[[] for _ in range(len(self.ml_models))] for __ in range(len(self.scores))],index=[sc.__name__ for sc in self.scores],columns=[md.name for md in self.ml_models])
            for _ in range(n_repetitions):
                dg_metrics, _, _ = self._evaluate_bnlearn_dg_model(dg_model=dg_model, learning_data_real=learning_data_real,n_learning=n_learning, n_train=n_train, n_test=n_test,SLClass=dg_model.SLClass)
                for score_name in repetition_results.index.values.tolist():
                    for ml_model in self.ml_models:
                        repetition_results[ml_model.name][score_name].append(dg_metrics[dg_model.SLClass][ml_model.name][score_name])
            list_of_results[dg_model.SLClass] = repetition_results
        return list_of_results

    def analysis_3_per_dg_model(self, dg_model_real: DGModel, n_learning: int, n_train: int,n_test: int, n_true_repetitions: int, n_learning_repititions: int, n_sl_repititions: int):
        dg_metrics, train_data, test_data = self._get_performance_by_repetition(dg_model_real, n_train+n_test,0.5, n_true_repetitions)
        list_of_top_true_ranks = []
        list_of_top_true_accuracies = []
        list_of_avg_all_ml_true_accuracies = []
        list_of_all_ml_true_accuracies = []
        list_of_top_ranks_from_practitioner_limited_world = []
        list_of_top_accuracies_from_practitioner_limited_world = []
        list_of_avg_all_ml_accuracies_from_practitioner_limited_world = {"DecisionTreeClassifier":[],"RandomForestClassifier":[],"KNeighborsClassifier":[],"GradientBoostingClassifier":[],"SVCRbf":[],"SVCLinear":[],"SVCSigmoid":[],"LogisticLASSO":[],"MLPClassifier":[]}
        list_of_all_ml_accuracies_from_practitioner_limited_world = []
        list_of_top_ranks_from_practitioner_sl_world = {"hc":[], "tabu":[],"rsmax2":[],"mmhc":[],"h2pc":[],"gs":[], "notears_linear":[]}#,"iamb":[],"fast.iamb":[],"iamb.fdr":[]}
        list_of_top_accuracies_from_practitioner_sl_world = {"hc": [],"tabu": [], "rsmax2": [], "mmhc": [], "h2pc": [],"gs":[], "notears_linear":[]}#,"iamb":[],"fast.iamb":[],"iamb.fdr":[]}
        list_of_avg_all_ml_accuracies_from_practitioner_sl_world = {"hc": {"DecisionTreeClassifier":[],"RandomForestClassifier":[],"KNeighborsClassifier":[],"GradientBoostingClassifier":[],"SVCRbf":[],"SVCLinear":[],"SVCSigmoid":[],"LogisticLASSO":[],"MLPClassifier":[]},"tabu": {"DecisionTreeClassifier":[],"RandomForestClassifier":[],"KNeighborsClassifier":[],"GradientBoostingClassifier":[],"SVCRbf":[],"SVCLinear":[],"SVCSigmoid":[],"LogisticLASSO":[],"MLPClassifier":[]}, "rsmax2": {"DecisionTreeClassifier":[],"RandomForestClassifier":[],"KNeighborsClassifier":[],"GradientBoostingClassifier":[],"SVCRbf":[],"SVCLinear":[],"SVCSigmoid":[],"LogisticLASSO":[],"MLPClassifier":[]}, "mmhc": {"DecisionTreeClassifier":[],"RandomForestClassifier":[],"KNeighborsClassifier":[],"GradientBoostingClassifier":[],"SVCRbf":[],"SVCLinear":[],"SVCSigmoid":[],"LogisticLASSO":[],"MLPClassifier":[]}, "h2pc": {"DecisionTreeClassifier":[],"RandomForestClassifier":[],"KNeighborsClassifier":[],"GradientBoostingClassifier":[],"SVCRbf":[],"SVCLinear":[],"SVCSigmoid":[],"LogisticLASSO":[],"MLPClassifier":[]}, "gs": {"DecisionTreeClassifier":[],"RandomForestClassifier":[],"KNeighborsClassifier":[],"GradientBoostingClassifier":[],"SVCRbf":[],"SVCLinear":[],"SVCSigmoid":[],"LogisticLASSO":[],"MLPClassifier":[]}, "notears_linear": {"DecisionTreeClassifier":[],"RandomForestClassifier":[],"KNeighborsClassifier":[],"GradientBoostingClassifier":[],"SVCRbf":[],"SVCLinear":[],"SVCSigmoid":[],"LogisticLASSO":[],"MLPClassifier":[]}}#"iamb": {"DecisionTreeClassifier":[],"RandomForestClassifier":[],"KNeighborsClassifier":[],"GradientBoostingClassifier":[],"SVCRbf":[],"SVCLinear":[],"SVCSigmoid":[],"LogisticLASSO":[],"MLPClassifier":[]}, "fast.iamb": {"DecisionTreeClassifier":[],"RandomForestClassifier":[],"KNeighborsClassifier":[],"GradientBoostingClassifier":[],"SVCRbf":[],"SVCLinear":[],"SVCSigmoid":[],"LogisticLASSO":[],"MLPClassifier":[]}, "iamb.fdr": {"DecisionTreeClassifier":[],"RandomForestClassifier":[],"KNeighborsClassifier":[],"GradientBoostingClassifier":[],"SVCRbf":[],"SVCLinear":[],"SVCSigmoid":[],"LogisticLASSO":[],"MLPClassifier":[]}}
        list_of_all_ml_accuracies_from_practitioner_sl_world = {"hc": [],"tabu": [], "rsmax2": [], "mmhc": [], "h2pc": [],"gs":[], "notears_linear":[]}#,"iamb":[],"fast.iamb":[],"iamb.fdr":[]}
        list_of_comparable_ranks_in_one_repitition = {} #used to store all ml methods to pick the max-scored option
        for rep in range(0, n_true_repetitions):
            for ml in dg_metrics:
                list_of_comparable_ranks_in_one_repitition[ml] = dg_metrics[ml]['balanced_accuracy_score'][rep]
            list_of_all_ml_true_accuracies.append(list_of_comparable_ranks_in_one_repitition.copy())
            list_of_top_true_ranks.append(max(list_of_comparable_ranks_in_one_repitition, key=list_of_comparable_ranks_in_one_repitition.get))
            list_of_top_true_accuracies.append(max(list_of_comparable_ranks_in_one_repitition.values()))
            list_of_comparable_ranks_in_one_repitition.clear()
        for ml in dg_metrics:
            list_of_avg_all_ml_true_accuracies.append(mean(dg_metrics[ml]['balanced_accuracy_score']))
        for learning_rep in range(0, n_learning_repititions):
            print("learning rep: ",learning_rep)
            train_data, test_data = self._get_train_and_test_from_dg(dg_model_real, n_train+n_test, 0.5)
            limited_dg_metrics = self._develop_all_ml_models(train_data, test_data)
            for ml in self.ml_models:
                list_of_comparable_ranks_in_one_repitition[ml.name] = limited_dg_metrics[ml.name]['balanced_accuracy_score']
                list_of_avg_all_ml_accuracies_from_practitioner_limited_world[ml.name].append(limited_dg_metrics[ml.name]['balanced_accuracy_score'])
            list_of_all_ml_accuracies_from_practitioner_limited_world.append(list_of_comparable_ranks_in_one_repitition.copy())
            list_of_top_ranks_from_practitioner_limited_world.append(max(list_of_comparable_ranks_in_one_repitition, key=list_of_comparable_ranks_in_one_repitition.get))
            list_of_top_accuracies_from_practitioner_limited_world.append(max(list_of_comparable_ranks_in_one_repitition.values()))
            list_of_comparable_ranks_in_one_repitition.clear()
            for dg_model in self.dg_models:
                print(dg_model.SLClass)
                repetition_results = pd.DataFrame(
                    data=[[[] for _ in range(len(self.ml_models))] for __ in range(len(self.scores))],
                    index=[sc.__name__ for sc in self.scores], columns=[md.name for md in self.ml_models])
                for sl_rep in range(0, n_sl_repititions):
                    if dg_model.SLClass == "hc" or dg_model.SLClass =="tabu" or dg_model.SLClass =="rsmax2" or dg_model.SLClass == "mmhc" or dg_model.SLClass =="h2pc" or dg_model.SLClass =="gs" or dg_model.SLClass =="iamb" or dg_model.SLClass =="fast.iamb" or dg_model.SLClass =="iamb.fdr":
                        sl_dg_metrics, sl_train_data, sl_test_data = self._evaluate_bnlearn_dg_model(dg_model=dg_model,learning_data_real=train_data,n_learning=n_learning, n_train=n_train,n_test=n_test, SLClass=dg_model.SLClass)
                    elif dg_model.SLClass == "notears_linear":
                        dg_model.fit(train_data)
                        sl_dg_metrics, sl_train_data, sl_test_data = self._evaluate_dg_model(dg_model=dg_model,n_learning=n_learning,n_train=n_train,n_test=n_test)
                    for score_name in repetition_results.index.values.tolist():
                        for ml in self.ml_models:
                            repetition_results[ml.name][score_name].append(sl_dg_metrics[dg_model.SLClass][ml.name][score_name])
                for ml in self.ml_models:
                    repetition_results[ml.name][score_name] = mean(repetition_results[ml.name][score_name])
                    list_of_comparable_ranks_in_one_repitition[ml.name] = repetition_results[ml.name]['balanced_accuracy_score']
                    list_of_avg_all_ml_accuracies_from_practitioner_sl_world[dg_model.SLClass][ml.name].append(repetition_results[ml.name]['balanced_accuracy_score'])
                list_of_all_ml_accuracies_from_practitioner_sl_world[dg_model.SLClass].append(list_of_comparable_ranks_in_one_repitition.copy())
                top_rank_per_sl = max(list_of_comparable_ranks_in_one_repitition, key=list_of_comparable_ranks_in_one_repitition.get)
                top_accuracies_per_sl = max(list_of_comparable_ranks_in_one_repitition.values())
                list_of_comparable_ranks_in_one_repitition.clear()
                list_of_top_ranks_from_practitioner_sl_world[dg_model.SLClass].append(top_rank_per_sl)
                list_of_top_accuracies_from_practitioner_sl_world[dg_model.SLClass].append(top_accuracies_per_sl)
        for ml in self.ml_models:
            list_of_avg_all_ml_accuracies_from_practitioner_limited_world[ml.name] = mean(list_of_avg_all_ml_accuracies_from_practitioner_limited_world[ml.name])
        for dg_model in self.dg_models:
            for ml in self.ml_models:
                list_of_avg_all_ml_accuracies_from_practitioner_sl_world[dg_model.SLClass][ml.name] = mean(list_of_avg_all_ml_accuracies_from_practitioner_sl_world[dg_model.SLClass][ml.name])
        print("----- Output of analysis -----")
        print("Infinite scenario ranks: ")
        print(list_of_top_true_ranks)
        print("Infinite scenario accuracies: ")
        print(list_of_top_true_accuracies)
        print("Infinite scenario avg accuracies for all ml methods: ")
        print(list_of_avg_all_ml_true_accuracies)
        print("Infinite scenario all accuracies for all ml methods: ")
        print(list_of_all_ml_true_accuracies)
        print("Limited real-world scenario ranks: ")
        print(list_of_top_ranks_from_practitioner_limited_world)
        print("Limited real-world scenario accuracies: ")
        print(list_of_top_accuracies_from_practitioner_limited_world)
        print("Limited real-world avg accuracies for all ml methods: ")
        print(list_of_avg_all_ml_accuracies_from_practitioner_limited_world)
        print("Limited real-world all accuracies for all ml methods: ")
        print(list_of_all_ml_accuracies_from_practitioner_limited_world)
        print("SL-supported scenario ranks: ")
        print(list_of_top_ranks_from_practitioner_sl_world)
        print("SL-supported scenario accuracies: ")
        print(list_of_top_accuracies_from_practitioner_sl_world)
        print("SL-supported avg accuracies for all ml methods: ")
        print(list_of_avg_all_ml_accuracies_from_practitioner_sl_world)
        print("SL-supported all accuracies for all ml methods: ")
        print(list_of_all_ml_accuracies_from_practitioner_sl_world)
        print("----- End of Analysis Output -----")
        return [list_of_top_true_ranks, list_of_top_true_accuracies, list_of_avg_all_ml_true_accuracies, list_of_all_ml_true_accuracies, list_of_top_ranks_from_practitioner_limited_world, list_of_top_accuracies_from_practitioner_limited_world, list_of_avg_all_ml_accuracies_from_practitioner_limited_world, list_of_all_ml_accuracies_from_practitioner_limited_world, list_of_top_ranks_from_practitioner_sl_world, list_of_top_accuracies_from_practitioner_sl_world, list_of_avg_all_ml_accuracies_from_practitioner_sl_world, list_of_all_ml_accuracies_from_practitioner_sl_world]

    def analysis_coef_per_dg_model(self, dg_model_real: DGModel, n_learning: int = 100):
        corr_dict = {}
        real_data, corr_dict[dg_model_real.name] = self._get_corr(dg_model_real)

        for dg_model in self.dg_models:
            dg_model.fit(real_data[0:n_learning])
            _, corr_dict[dg_model.name] = self._get_corr(dg_model)

        return corr_dict

    def analysis_violin_per_dg_model(self, dg_model_real: DGModel, n_samples: int, tr_frac: float, n_reps: int):
        scores_per_dg_model = {}
        dg_model_real_scores, train_data, test_data = self._get_performance_by_repetition(dg_model_real, n_samples,
                                                                                          tr_frac, n_reps)
        scores_per_dg_model[dg_model_real.name] = dg_model_real_scores
        for dg_model in self.dg_models:
            dg_model.fit(train_data)
            if dg_model.num_vars != len(test_data.all.columns):
                continue
            dg_model_scores, *_ = self._get_performance_by_repetition(dg_model, n_samples, tr_frac, n_reps,
                                                                      test_data=test_data)
            scores_per_dg_model[dg_model.name] = dg_model_scores
        return scores_per_dg_model

    def analysis_2b_per_dg_model(self, dg_model_real: DGModel,n_samples: int, tr_frac: float, n_reps: int):
        scores_per_dg_model = {}
        dg_model_real_scores, train_data, test_data = self._get_performance_by_repetition(dg_model_real, n_samples,
                                                                                          tr_frac, n_reps)
        scores_per_dg_model[dg_model_real.name] = dg_model_real_scores
        for dg_model in self.dg_models:
            dg_model.fit(train_data)
            if dg_model.num_vars != len(test_data.all.columns):
                continue
            dg_model_scores, *_ = self._get_performance_by_repetition(dg_model, n_samples, tr_frac, n_reps,
                                                                      test_data=test_data)
            scores_per_dg_model[dg_model.name] = dg_model_scores
        #print(scores_per_dg_model)
        return scores_per_dg_model

    def _evaluate_ml_model(self, ml_model: MachineLearner, test_data: Data):
        '''
        Given a **trained** ml model, evaluate its performance using all the defined metrics on the test set.

        '''
        y_pred = ml_model.predict(test_data)
        metrics = {}
        for score in self.scores:
            metrics.update({score.__name__: score(y_pred=y_pred, y_true=test_data.y)})
        return metrics

    def _develop_all_ml_models(self, train_data: Data, test_data: Data):  # level 3 repetition
        '''
        Given a training set and a test set, train each machine learning model on the training data and test it on the
        test set.
        '''
        metrics = {}
        for ml_model in self.ml_models:
            metrics[f'{ml_model.name}'] = self._develop_ml_model(ml_model=ml_model, train_data=train_data,
                                                                 test_data=test_data)
        return metrics

    def _evaluate_dg_model(self, dg_model: DGModel, n_learning: int, n_train: int, n_test: int, test_data: Data = None):

        train_data = dg_model.generate(n_train, self.outcome_name)

        if test_data is None:
            test_data = dg_model.generate(n_test, self.outcome_name)

        metrics = self._develop_all_ml_models(train_data, test_data)
        metrics = {f'{dg_model.name}': metrics}
        learning_data = train_data[0:n_learning:1] if n_learning > 0 else None
        return metrics, learning_data, test_data

    def _evaluate_bnlearn_dg_model(self, dg_model: DGModel, learning_data_real, n_learning: int, n_train: int, n_test: int,SLClass: str):
        learning_data_real = learning_data_real.all
        learning_data_real.loc[0] = 1

        #learning_data_real.loc[0,"AppDtGnTm"] = 1
        #learning_data_real.loc[0, "PrtThread"] = 1
        #learning_data_real.loc[0, "TnrSpply"] = 1
        #learning_data_real.loc[0, "AvlblVrtlMmry"] = 1
        #learning_data_real.loc[0, "AppData"] = 1
        #learning_data_real.loc[0, "AppOK"] = 1
        #learning_data_real.loc[0, "CblPrtHrdwrOK"] = 1
        #print(count(learning_data_real$PrtThread))
        #print(table(is.na(training_output)))
        #print(sapply(learning_data_real, levels))

        if SLClass == "hc":
            robjects.r('''
                        library(bnlearn)
                        library(plyr)
                        bn_learn <- function(learning_data_real, n_train, n_test, verbose=FALSE) {     
                        learning_data_real <- data.frame(lapply(learning_data_real,factor), stringsAsFactors=TRUE)   
                        my_bn <- hc(learning_data_real, replace.unidentifiable=TRUE, method='bayes')
                        fit = bn.fit(my_bn, learning_data_real)
                        training_output = rbn(my_bn, n_train, learning_data_real)
                        testing_output = rbn(my_bn, n_test, learning_data_real)
                        training_output[is.na(training_output)] <- 0
                        testing_output[is.na(testing_output)] <- 0
                        list_output <- list(training_output, testing_output)
                        }
                        ''')
        elif SLClass == "tabu":
            robjects.r('''
                        library(bnlearn)
                        library(plyr)
                        bn_learn <- function(learning_data_real, n_train, n_test, verbose=FALSE) {     
                        learning_data_real <- data.frame(lapply(learning_data_real,factor), stringsAsFactors=TRUE)   
                        my_bn <- tabu(learning_data_real)
                        fit = bn.fit(my_bn, learning_data_real)
                        training_output = rbn(my_bn, n_train, learning_data_real)
                        testing_output = rbn(my_bn, n_test, learning_data_real)
                        training_output[is.na(training_output)] <- 0
                        testing_output[is.na(testing_output)] <- 0
                        list_output <- list(training_output, testing_output)
                        }
                        ''')
        elif SLClass == "rsmax2":
            robjects.r('''
                        library(bnlearn)
                        library(plyr)
                        bn_learn <- function(learning_data_real, n_train, n_test, verbose=FALSE) {     
                        learning_data_real <- data.frame(lapply(learning_data_real,factor), stringsAsFactors=TRUE)   
                        my_bn <- rsmax2(learning_data_real)
                        fit = bn.fit(my_bn, learning_data_real)
                        training_output = rbn(my_bn, n_train, learning_data_real)
                        testing_output = rbn(my_bn, n_test, learning_data_real)
                        training_output[is.na(training_output)] <- 0
                        testing_output[is.na(testing_output)] <- 0
                        list_output <- list(training_output, testing_output)
                        }
                        ''')
        elif SLClass == "mmhc":
            robjects.r('''
                        library(bnlearn)
                        library(plyr)
                        bn_learn <- function(learning_data_real, n_train, n_test, verbose=FALSE) {     
                        learning_data_real <- data.frame(lapply(learning_data_real,factor), stringsAsFactors=TRUE)   
                        my_bn <- mmhc(learning_data_real)
                        fit = bn.fit(my_bn, learning_data_real)
                        training_output = rbn(my_bn, n_train, learning_data_real)
                        testing_output = rbn(my_bn, n_test, learning_data_real)
                        training_output[is.na(training_output)] <- 0
                        testing_output[is.na(testing_output)] <- 0
                        list_output <- list(training_output, testing_output)
                        }
                        ''')
        elif SLClass == "h2pc":
            robjects.r('''
                        library(bnlearn)
                        library(plyr)
                        bn_learn <- function(learning_data_real, n_train, n_test, verbose=FALSE) {     
                        learning_data_real <- data.frame(lapply(learning_data_real,factor), stringsAsFactors=TRUE)   
                        my_bn <- h2pc(learning_data_real)
                        fit = bn.fit(my_bn, learning_data_real)
                        training_output = rbn(my_bn, n_train, learning_data_real)
                        testing_output = rbn(my_bn, n_test, learning_data_real)
                        training_output[is.na(training_output)] <- 0
                        testing_output[is.na(testing_output)] <- 0
                        list_output <- list(training_output, testing_output)
                        }
                        ''')
        elif SLClass == "gs":
            robjects.r('''
                        library(bnlearn)
                        library(plyr)
                        bn_learn <- function(learning_data_real, n_train, n_test, verbose=FALSE) {     
                        learning_data_real <- data.frame(lapply(learning_data_real,factor), stringsAsFactors=TRUE)   
                        my_bn <- cextend(gs(learning_data_real), strict=FALSE)
                        fit = bn.fit(my_bn, learning_data_real)
                        training_output = rbn(my_bn, n_train, learning_data_real)
                        testing_output = rbn(my_bn, n_test, learning_data_real)
                        training_output[is.na(training_output)] <- 0
                        testing_output[is.na(testing_output)] <- 0
                        list_output <- list(training_output, testing_output)
                        }
                        ''')
        elif SLClass == "iamb":
            robjects.r('''
                        library(bnlearn)
                        library(plyr)
                        bn_learn <- function(learning_data_real, n_train, n_test, verbose=FALSE) {     
                        learning_data_real <- data.frame(lapply(learning_data_real,factor), stringsAsFactors=TRUE)   
                        my_bn <- cextend(iamb(learning_data_real), strict=FALSE)
                        fit = bn.fit(my_bn, learning_data_real)
                        training_output = rbn(my_bn, n_train, learning_data_real)
                        testing_output = rbn(my_bn, n_test, learning_data_real)
                        training_output[is.na(training_output)] <- 0
                        testing_output[is.na(testing_output)] <- 0
                        list_output <- list(training_output, testing_output)
                        }
                        ''')
        elif SLClass == "fast.iamb":
            robjects.r('''
                        library(bnlearn)
                        library(plyr)
                        bn_learn <- function(learning_data_real, n_train, n_test, verbose=FALSE) {     
                        learning_data_real <- data.frame(lapply(learning_data_real,factor), stringsAsFactors=TRUE)   
                        my_bn <- cextend(fast.iamb(learning_data_real), strict=FALSE)
                        fit = bn.fit(my_bn, learning_data_real)
                        training_output = rbn(my_bn, n_train, learning_data_real)
                        testing_output = rbn(my_bn, n_test, learning_data_real)
                        training_output[is.na(training_output)] <- 0
                        testing_output[is.na(testing_output)] <- 0
                        list_output <- list(training_output, testing_output)
                        }
                        ''')
        elif SLClass == "iamb.fdr":
            robjects.r('''
                        library(bnlearn)
                        library(plyr)
                        bn_learn <- function(learning_data_real, n_train, n_test, verbose=FALSE) {     
                        learning_data_real <- data.frame(lapply(learning_data_real,factor), stringsAsFactors=TRUE)   
                        my_bn <- cextend(iamb.fdr(learning_data_real), strict=FALSE)
                        fit = bn.fit(my_bn, learning_data_real)
                        training_output = rbn(my_bn, n_train, learning_data_real)
                        testing_output = rbn(my_bn, n_test, learning_data_real)
                        training_output[is.na(training_output)] <- 0
                        testing_output[is.na(testing_output)] <- 0
                        list_output <- list(training_output, testing_output)
                        }
                        ''')

        bn_hc = robjects.r['bn_learn']
        bn_train_output = bn_hc(learning_data_real, n_train, n_test)

        train_data = np.array(bn_train_output[0])
        test_data = np.array(bn_train_output[1])
        X = pd.DataFrame(train_data[:, :-1])
        y = pd.Series(train_data[:, -1], name="Y")
        train_data = Data(name="train", X=X, y=y)
        X = pd.DataFrame(test_data[:, :-1])
        y = pd.Series(test_data[:, -1], name="Y")
        test_data = Data(name="test", X=X, y=y)

        metrics = self._develop_all_ml_models(train_data, test_data)
        metrics = {f'{dg_model.name}': metrics}
        learning_data = train_data[0:n_learning:1] if n_learning > 0 else None
        return metrics, learning_data, test_data

    def _evaluate_bnstruct_dg_model(self, dg_model: DGModel, learning_data_real, n_learning: int, n_train: int, n_test: int,SLClass: str):
        learning_data_real = learning_data_real.all

        #learning_data_real.loc[0,"AppDtGnTm"] = 1
        #learning_data_real.loc[0, "PrtThread"] = 1
        #learning_data_real.loc[0, "TnrSpply"] = 1
        #learning_data_real.loc[0, "AvlblVrtlMmry"] = 1
        #learning_data_real.loc[0, "AppData"] = 1
        #learning_data_real.loc[0, "AppOK"] = 1
        #learning_data_real.loc[0, "CblPrtHrdwrOK"] = 1
        #print(count(learning_data_real$PrtThread))
        #print(table(is.na(training_output)))
        #print(sapply(learning_data_real, levels))

        if SLClass == "sem":
            robjects.r('''
                        library(bnlearn)
                        library(plyr)
                        bn_struct <- function(learning_data_real, n_train, n_test, verbose=FALSE) {     
                        learning_data_real <- data.frame(lapply(learning_data_real,factor), stringsAsFactors=TRUE)   
                        my_bn <- cextend(iamb.fdr(learning_data_real, undirected=FALSE), strict=FALSE)
                        fit = bn.fit(my_bn, learning_data_real)
                        training_output = rbn(my_bn, n_train, learning_data_real)
                        testing_output = rbn(my_bn, n_test, learning_data_real)
                        training_output[is.na(training_output)] <- 0
                        testing_output[is.na(testing_output)] <- 0
                        list_output <- list(training_output, testing_output)
                        }
                        ''')
        elif SLClass == "second":
            robjects.r('''
                        install.packages("tetrad")
                        library(tetrad)
                        bn_struct <- function(learning_data_real, n_train, n_test, verbose=FALSE) {     
                        my_bn <- pc(learning_data_real, indepTest=binCItest, labels=colnames(learning_data_real))
                        print(my_bn)
                        training_output = simulateSEM(my_bn, n=n_train)
                        testing_output = simulateSEM(my_bn, n=n_test)
                        print(training_output)
                        print(testing_output)
                        training_output[is.na(training_output)] <- 0
                        testing_output[is.na(testing_output)] <- 0
                        list_output <- list(training_output, testing_output)
                        }
                        ''')
        elif SLClass == "third":
            robjects.r('''
                        install.packages("bnstruct")
                        library(bnstruct)
                        bn_struct <- function(learning_data_real, n_train, n_test, verbose=FALSE) {     
                        my_bn <- learn.network(x=BNDataset(learning_data_real, discreteness=rep(TRUE, ncol(learning_data_real)), variables=colnames(learning_data_real), algo="sem", scoring.func="AIC"))
                        print(my_bn)
                        training_output = sample_from_dag(my_bn, n_train)
                        testing_output = sample_from_dag(my_bn, n_test)
                        print(training_output)
                        print(testing_output)
                        training_output[is.na(training_output)] <- 0
                        testing_output[is.na(testing_output)] <- 0
                        list_output <- list(training_output, testing_output)
                        }
                        ''')
        elif SLClass == "fourth":
            robjects.r('''
                        install.packages("BDgraph")
                        library(BDgraph)
                        bn_struct <- function(learning_data_real, n_train, n_test, verbose=FALSE) {     
                        fit <- bdgraph(learning_data_real, algorithm = "rjmcmc")
                        training_output = bdgraph.sim(fit, n = n_train)
                        testing_output = bdgraph.sim(fit, n = n_test)
                        print(training_output)
                        print(testing_output)
                        training_output[is.na(training_output)] <- 0
                        testing_output[is.na(testing_output)] <- 0
                        list_output <- list(training_output, testing_output)
                        }
                        ''')
        elif SLClass == "Fifth":
            robjects.r('''
                        install.packages("bnlearn")
                        install.packages("pcalg")
                        library(pcalg)
                        library(bnlearn)
                        bn_struct <- function(learning_data_real, n_train, n_test, verbose=FALSE) {     
                        suffstat <- list(C = cor(learning_data_real), n = nrow(learning_data_real)) 
                        pcalg <- pc(suffstat, indepTest=gaussCItest, p=ncol(learning_data_real), labels=colnames(learning_data_real), alpha=0.05)
                        dag <- pcalg2dag(pcalg$graph)
                        training_output = rbn(dag, n = n_train)
                        testing_output = rbn(dag, n = n_test)
                        print(training_output)
                        print(testing_output)
                        training_output[is.na(training_output)] <- 0
                        testing_output[is.na(testing_output)] <- 0
                        list_output <- list(training_output, testing_output)
                        }
                        ''')

        bn_sem = robjects.r['bn_struct']
        bn_train_output = bn_sem(learning_data_real, n_train, n_test)

        train_data = np.array(bn_train_output[0])
        test_data = np.array(bn_train_output[1])
        X = pd.DataFrame(train_data[:, :-1])
        y = pd.Series(train_data[:, -1], name="Y")
        train_data = Data(name="train", X=X, y=y)
        X = pd.DataFrame(test_data[:, :-1])
        y = pd.Series(test_data[:, -1], name="Y")
        test_data = Data(name="test", X=X, y=y)

        metrics = self._develop_all_ml_models(train_data, test_data)
        metrics = {f'{dg_model.name}': metrics}
        learning_data = train_data[0:n_learning:1] if n_learning > 0 else None
        return metrics, learning_data, test_data

    def _get_performance_by_repetition(self, dg_model: DGModel, n_samples_real: int, tr_frac: float, n_reps: int,
                                       test_data=None):
        all_scores = {ml_model.name: {score.__name__: [] for score in self.scores} for ml_model in self.ml_models}
        train_data = None

        for rep in range(n_reps):
            train_data = dg_model.generate(int(n_samples_real * tr_frac), self.outcome_name)
            if test_data is None:
                test_data = dg_model.generate(int(n_samples_real * (1 - tr_frac)), self.outcome_name)
            for ml_model in self.ml_models:
                scores = self._develop_ml_model(ml_model, train_data, test_data)
                for score_name in scores.keys():
                    all_scores[ml_model.name][score_name].append(scores[score_name])
        return all_scores, train_data, test_data

    def _develop_ml_model(self, ml_model: MachineLearner, train_data: Data, test_data: Data):
        '''
        For a given ml model, train it on the provided training set and evaluate it on the test set.

        '''
        ml_model.learn(train_data)
        scores = self._evaluate_ml_model(ml_model, test_data)
        return scores

    def _get_corr(self, dg_model):
        data = dg_model.generate(num_samples=100000, outcome_name=self.outcome_name)
        up_tri_mat = np.triu(np.ones(data.all.corr().shape), k=1).astype(bool)
        corr_df = data.all.corr().where(up_tri_mat)
        corr_df = corr_df.stack().reset_index()
        return data, corr_df

    def _get_train_and_test_from_dg(self, dg_model_real, n_samples, tr_frac):
        train_data = dg_model_real.generate(int(n_samples * tr_frac), self.outcome_name)
        test_data = dg_model_real.generate(int(n_samples * (1 - tr_frac)), self.outcome_name)
        return train_data, test_data
