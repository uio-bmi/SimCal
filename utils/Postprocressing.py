import matplotlib.pyplot as plt
import pandas as pd
import scipy
from transpose_dict import TD
from scipy.stats import sem
import numpy as np
import seaborn as sns


class Postprocessing():
    def __init__(self):
        pass

    def plot_analysis0(self, analysis0_results: pd.DataFrame):
        score_names = analysis0_results.index
        for score_name in score_names:
            y = [np.mean(analysis0_results[alg][score_name]) for alg in analysis0_results.columns]
            y_err = [scipy.stats.sem(analysis0_results[alg][score_name]) for alg
                       in analysis0_results.columns]

            alg_names = analysis0_results.columns
            x_pos = np.arange(len(alg_names))

            fig, ax = plt.subplots()
            ax.bar(x_pos, y, yerr=y_err, align='center', alpha=0.5, ecolor='black', capsize=10)
            ax.set_ylim(0, 1)
            ax.set_ylabel(score_name)
            ax.set_xticks(x_pos)
            ax.set_xticklabels(alg_names, rotation=90)
            # ax.set_title(f'{score_name} of different ML models')
            ax.yaxis.grid(True)

            # Save the figure and show
            plt.tight_layout()
            plt.savefig('analysis_0_bar_plot_with_error_bars.png')
            plt.show()

    def plot_analysis1(self, analysis1_results: pd.DataFrame):
        score_names = analysis1_results.index
        for score_name in score_names:
            y = [np.mean(analysis1_results[alg][score_name]) for alg in analysis1_results.columns]

            y_err = [np.std(analysis1_results[alg][score_name]) for alg
                       in analysis1_results.columns]
            alg_names = analysis1_results.columns
            x_pos = np.arange(len(alg_names))

            fig, ax = plt.subplots()
            ax.bar(x_pos, y, yerr=y_err, align='center', alpha=0.5, ecolor='black', capsize=10)
            ax.set_ylim(0, 1)
            ax.set_ylabel(score_name)
            ax.set_xticks(x_pos)
            ax.set_xticklabels(alg_names, rotation=90)
            # ax.set_title(f'{score_name} of different ML models')
            ax.yaxis.grid(True)

            # Save the figure and show
            plt.tight_layout()
            plt.savefig('analysis_1_bar_plot_with_error_bars.png')
            plt.show()

    def plot_analysis2(self, analysis2_results: dict):
        corr_df = self.corr_dict_to_pd(analysis2_results)
        sns_plot = sns.scatterplot(x='pair', y='correlation', data=corr_df, hue='world', alpha=0.6, style="world")
        plt.show()

    def plot_analysis_coef_gks(self, analysis2_results: dict, pipeline: str = "pipeline1"):
        markers = ['x', 'o', '^', '*', '+', 's', 'p']
        c = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
        corr_df = self.corr_dict_to_pd(analysis2_results)
        real_corr = corr_df.loc[corr_df['world'] == pipeline].correlation
        for i, world in enumerate(set(corr_df.world)):
            if world == "PC":
                continue
            world_corr = corr_df.loc[corr_df['world'] == world].correlation

            plt.scatter(real_corr.tolist(), world_corr.tolist(), marker=markers[i], c=c[i], label=world)

        plt.title("pair-wise correlations of learned vs real world")
        plt.xlabel("real world correlations")
        plt.ylabel("learned world correlations")
        plt.legend()
        plt.tight_layout()
        plt.savefig('scatter_plot_of_ccs.png')
        plt.show()

    def plot_analysis3(self, analysis3_results: dict):

        for score_name in TD(analysis3_results, 2).keys():
            data = self.dict_to_list(score_name, analysis3_results)
            worlds = list(analysis3_results.keys())
            df = pd.DataFrame(data, columns=["ML", *worlds])
            methods = df.transpose().iloc[0].values
            grouped_df = df.transpose()
            grouped_df = grouped_df.rename(columns=grouped_df.iloc[0]).drop(grouped_df.index[0])
            grouped_df = grouped_df.reset_index()

            #ax = df.plot(x="ML", y=worlds, kind="bar", figsize=(11, 11))
            ax = grouped_df.plot(x="index", y=methods, kind="bar", figsize=(11, 11))
            ax.set_ylim(0, 1)
            ax.set_ylabel(score_name)
            plt.title("Inter-world Benchmarking of ML Pipelines Grouped by SL")
            plt.tight_layout()
            plt.savefig('interworld_benchmarking_of_mlpipelines_by_SL.png')
            plt.show()

            ax = df.plot(x="ML", y=worlds, kind="bar", figsize=(11, 11))
            ax.set_ylim(0, 1)
            ax.set_ylabel(score_name)
            plt.title("Inter-world Benchmarking of ML Pipelines Grouped by ML")
            plt.tight_layout()
            plt.savefig('interworld_benchmarking_of_mlpipelines_by_ML.png')
            plt.show()

    def plot_analysis_violin(self, analysis4_results: dict, score_name='balanced_accuracy_score'):
        fig, ax = plt.subplots(figsize=(20, 20))
        unfolded_scores = self.dict_to_dataframe_sns(analysis4_results, score_name)
        ax = sns.violinplot(x="world", y="score", hue="ml_model", data=unfolded_scores)
        plt.legend()
        plt.title("Benchmarking of ML Pipelines in Real and Learned worlds")
        plt.tight_layout()
        plt.legend()
        plt.savefig('repeated_interworld_benchmarking_of_mlpipelines.png')
        plt.show()

    def plot_analysis3b(self, analysis4_results: dict):
        methods = ['DecisionTreeClassifier','RandomForestClassifier','KNeighborsClassifier','GradientBoostingClassifier','SVCRbf','SVCLinear','SVCSigmoid','LogisticLASSO','MLPClassifier']
        # Additional NBMethods 'GaussianNB','BernoulliNB','MultinomialNB','ComplementNB','CategoricalNB',
        method_max_real = {}
        fig, ax = plt.subplots(figsize=(10, 10))
        real_scores_list = []
        learned_scores_list = []
        learned_scores_dev = []
        score_df = pd.DataFrame.from_dict(analysis4_results)
        #print(score_df)
        real_raw_scores = score_df["pipeline1"]
        i = 0
        for measure in real_raw_scores:
            i += 1
            valuesList = [measure[key] for key in measure]
            real_scores_list.append(np.mean(valuesList))
        #print("real-world measures:")
        #print(real_scores_list)
        score_df.drop('pipeline1', inplace=True, axis=1)
        #print("learned-world measures:")
        for (columnName, columnData) in score_df.iteritems():
            #print('Column Name : ', columnName)
            #print('Column Contents : ', columnData.values)
            for learned_measure in columnData.values:
                learnedValuesList = [learned_measure[v] for v in learned_measure]
                learned_scores_list.append(np.mean(learnedValuesList))
                learned_scores_dev.append(np.std(learnedValuesList)/np.sqrt(len(learnedValuesList)))
            #print("----")
            #print(real_scores_list)
            #print(learned_scores_list)
            #method_max_learned = {
            #    methods[learned_scores_list.index(max(learned_scores_list))]: max(learned_scores_list)}

            plt.scatter(real_scores_list, learned_scores_list,label=columnName)
            #plt.errorbar(real_scores_list, learned_scores_list, yerr=learned_scores_dev, fmt='o', label=columnName)
            learned_scores_list.clear()
            learned_scores_dev.clear()
        #plt.scatter(real_scores_list, learned_scores_list, label=columnName)

        ranges_of_performance = []
        #for ml in bootstrap_real_results:
        #    print(ml)
        #    print(bootstrap_real_results[ml]['balanced_accuracy_score'])
        #    range_ml = np.std(bootstrap_real_results[ml]['balanced_accuracy_score'])
        #    ranges_of_performance.append(range_ml)
        #    print("ml :", ml)
        #    print("score range :", range_ml)

        method_max_real = {methods[real_scores_list.index(max(real_scores_list))]: max(real_scores_list)}

        print("Max val", method_max_real)
        #print("Max val", method_max_learned)

        plt.scatter(method_max_real.values(), method_max_real.values(), marker ="^",s = 200,label=method_max_real.keys())
        #plt.errorbar(real_scores_list, real_scores_list, yerr=ranges_of_performance, fmt='o', label='pipeline1')
        plt.title("Rank-order performances of ML methods between Real vs Learned worlds")
        plt.xlabel("Real world performance")
        plt.ylabel("Learned world performance")
        plt.legend()
        plt.xlim(0.4, 1)
        plt.ylim(0.4, 1)

        plt.tight_layout()
        plt.savefig('scatter_plot_of_real_learned_scores.png')
        plt.show()
        #print(analysis4_results)

    def dict_to_list(self, score_name, all_results: dict):
        data = []
        ml_algs = list(TD(all_results, 1).keys())
        worlds = list(all_results.keys())
        for alg in ml_algs:
            inner_list = [alg]
            for world in worlds:
                inner_list.append(all_results[world][alg][score_name])
            data.append(inner_list)
        return data

    # not used for now
    def get_true_performance_stats(self, scores: dict):
        # scores: dict of shape {"ml_model_name": {"score_name": list_of_values}}
        score_names = TD(scores, 1).keys()
        stats = {ml_model_name: {score_name: {} for score_name in score_names} for ml_model_name in scores.keys()}
        for ml_model_name in scores.keys():
            for score in score_names:
                stats[ml_model_name][score]["min"] = np.min(scores[ml_model_name][score])
                stats[ml_model_name][score]["max"] = np.max(scores[ml_model_name][score])
                stats[ml_model_name][score]["mean"] = np.mean(scores[ml_model_name][score])
        return stats

    def dict_to_dataframe_sns(self, scores: dict, score_name: str = 'balanced_accuracy_score') -> pd.DataFrame:
        '''

        :param scores: dict of form {world_name: {ml_model_name: {score_name: list of scores]...}...}...}
        :param score_name:
        :return: dataframe with columns=[world, ml_model, score] and rows corresponding to individual entries
        '''
        score_dict = TD(scores, 2)[score_name]
        raw_scores_df = pd.DataFrame(score_dict)
        list_of_lists = []

        for col in raw_scores_df.columns:
            for row in raw_scores_df.index:
                for i in range(5):
                    list_of_lists.append([col, row, raw_scores_df[col][row][i]])

        return pd.DataFrame(list_of_lists, columns=["world", "ml_model", "score"])

    def corr_dict_to_pd(self, corr_dict):
        corr_pd = pd.concat(corr_dict)
        corr_pd.rename(columns={0: 'correlation'}, inplace=True)
        corr_pd.index = corr_pd.index.set_names(['world', 'pair'])
        corr_pd.reset_index(level=['world', 'pair'], inplace=True)
        corr_pd["pair"] = corr_pd["level_0"] + corr_pd["level_1"]
        corr_pd.drop(columns=["level_0", "level_1"], inplace=True)
        return corr_pd
