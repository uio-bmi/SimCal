import pandas as pd
from typing import List, Callable

from numpy import mean, int64

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

# This class provides high-level data-generation and ML benchmarking functions for a user-specified world with parameters
class Evaluator:
    def __init__(self, ml_models: List[MachineLearner], dg_models: List[DGModel], real_models: List[DGModel],
                 scores: List[Callable], outcome_name: str = "Y"):
        self.scores = scores
        self.real_models = real_models
        self.ml_models = ml_models
        self.dg_models = dg_models
        self.outcome_name = outcome_name

    def realworld_benchmark(self, dg_model_real: DGModel, n_repetitions, n_samples: int, tr_frac: float):
        """
        :param dg_model_real: a real-world DagSim model
        :n_repetitions: number of times to repeat training-testing
        :param n_samples: number of samples to generate for training + testing
        :param tr_frac: fraction of data for training
        :return: results: benchmarks in the shape {model_name: {score_name: list_of_score_values], ...}, ...}
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
        return results

    def realworld_benchmark_bootstrapping(self, dg_model_real: DGModel, n_samples: int, tr_frac: float, n_btstrps: int):
        """
        :param dg_model_real: a real-world DagSim model
        :param n_samples: number of samples to generate for training + testing
        :param tr_frac: fraction of data for training
        :param n_btstrps: number of times to perform bootstrapping
        :return: results: benchmarks in the shape {model_name: {score_name: list_of_score_values], ...}, ...}
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
        return btstrp_results

    def meta_simulate(self, dg_model_real: DGModel, n_learning: int, n_train: int,n_test: int, n_true_repetitions: int, n_practitioner_repititions: int, n_sl_repititions: int):
        """
            :param dg_model_real: a real-world DagSim model
            :param n_learning: number of samples to draw from n_train to use in learners
            :param n_train: number of samples to generate for training
            :param n_test: number of samples to generate for testing
            :param n_true_repetitions: number of times to repeat benchmarking in the true real-world
            :param n_practitioner_repititions: number of times to repeat benchmarking in a limited real-world
            :param n_sl_repititions: number of times to repeat benchmarking in the learner
            :return: results: benchmarks in the shape {model_name: {learner: {score_name: list_of_score_values], ...}, ...}, ...}
        """
        list_of_ntrue_accuracies = {method.name: [] for method in self.ml_models}
        list_of_npractitioner_accuracies = {method.name: [] for method in self.ml_models}
        list_of_nsl_accuracies = {learner.SLClass: {method.name: [] for method in self.ml_models} for learner in self.dg_models}
        dg_metrics, _, _ = self._get_performance_by_repetition(dg_model_real, n_train + n_test,0.5, n_true_repetitions)
        limited_dg_metrics, train_data_list, test_data_list = self._get_performance_by_repetition_return_data(dg_model_real, n_train + n_test,0.5, n_practitioner_repititions)
        for ml in dg_metrics: # extract balanced_accuracy_score organisation
            list_of_ntrue_accuracies[ml] = dg_metrics[ml]['balanced_accuracy_score']
            list_of_npractitioner_accuracies[ml] = limited_dg_metrics[ml]['balanced_accuracy_score']
        for practitioner_rep in range(0, n_practitioner_repititions):
            print("practitioner rep: ", practitioner_rep)
            #train_data, test_data = self._get_train_and_test_from_dg(dg_model_real, n_train + n_test, 0.5)
            #limited_dg_metrics = self._develop_all_ml_models(train_data, test_data)
            #list_of_npractitioner_accuracies.append(limited_dg_metrics)
            for dg_model in self.dg_models:
                print(dg_model.SLClass)
                repetition_results = pd.DataFrame(
                    data=[[[] for _ in range(len(self.ml_models))] for __ in range(len(self.scores))],
                    index=[sc.__name__ for sc in self.scores], columns=[md.name for md in self.ml_models])
                for sl_rep in range(0, n_sl_repititions):
                    sl_dg_metrics, sl_train_data, sl_test_data = self._evaluate_bnlearn_dg_model(dg_model=dg_model,learning_data_real=train_data_list[practitioner_rep],n_learning=n_learning,n_train=n_train,n_test=n_test,SLClass=dg_model.SLClass)
                    for score_name in repetition_results.index.values.tolist():
                        for ml in self.ml_models:
                            repetition_results[ml.name][score_name].append(sl_dg_metrics[dg_model.SLClass][ml.name][score_name])
                for ml in self.ml_models:
                    repetition_results[ml.name][score_name] = mean(repetition_results[ml.name][score_name])
                    list_of_nsl_accuracies[dg_model.SLClass][ml.name].append(repetition_results[ml.name][score_name])
        print("----- Output of analysis -----")
        print("Infinite scenario all accuracies for all ml methods: ")
        print(list_of_ntrue_accuracies)
        print("Limited real-world all accuracies for all ml methods: ")
        print(list_of_npractitioner_accuracies)
        print("Learned worlds all accuracies for all ml methods: ")
        print(list_of_nsl_accuracies)
        print("----- End of Analysis Output -----")
        df_true = pd.DataFrame(list_of_ntrue_accuracies)
        df_true.to_csv("ntrue_accuracies.csv")
        df_limited = pd.DataFrame(list_of_npractitioner_accuracies)
        df_limited.to_csv("npractitioner_accuracies.csv")
        df_nsl = pd.DataFrame(list_of_nsl_accuracies)
        df_nsl.to_csv("nsl_accuracies.csv")
        return [list_of_ntrue_accuracies, list_of_npractitioner_accuracies, list_of_nsl_accuracies, n_true_repetitions, n_practitioner_repititions, n_sl_repititions, self.dg_models, self.ml_models]

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
            metrics[f'{ml_model.name}'] = self._develop_ml_model(ml_model=ml_model, train_data=train_data,test_data=test_data)
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
        learning_data_real.loc[1] = 0

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

        train_data = np.array(bn_train_output[0], dtype=int64)
        test_data = np.array(bn_train_output[1], dtype=int64)
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

    def _get_performance_by_repetition(self, dg_model: DGModel, n_samples_real: int, tr_frac: float, n_reps: int,test_data=None):
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

    def _get_performance_by_repetition_return_data(self, dg_model: DGModel, n_samples_real: int, tr_frac: float, n_reps: int,test_data=None):
        all_scores = {ml_model.name: {score.__name__: [] for score in self.scores} for ml_model in self.ml_models}
        train_data = None
        train_data_list = []
        test_data_list = []
        for rep in range(n_reps):
            train_data = dg_model.generate(int(n_samples_real * tr_frac), self.outcome_name)
            train_data_list.append(train_data)
            if test_data is None:
                test_data = dg_model.generate(int(n_samples_real * (1 - tr_frac)), self.outcome_name)
                test_data_list.append(test_data)
            for ml_model in self.ml_models:
                scores = self._develop_ml_model(ml_model, train_data, test_data)
                for score_name in scores.keys():
                    all_scores[ml_model.name][score_name].append(scores[score_name])
        return all_scores, train_data_list, test_data_list


    def _develop_ml_model(self, ml_model: MachineLearner, train_data: Data, test_data: Data):
        '''
        For a given ml model, train it on the provided training set and evaluate it on the test set.

        '''
        ml_model.learn(train_data)
        scores = self._evaluate_ml_model(ml_model, test_data)
        return scores

    def _get_train_and_test_from_dg(self, dg_model_real, n_samples, tr_frac):
        train_data = dg_model_real.generate(int(n_samples * tr_frac), self.outcome_name)
        test_data = dg_model_real.generate(int(n_samples * (1 - tr_frac)), self.outcome_name)
        return train_data, test_data
