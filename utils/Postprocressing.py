import os

import matplotlib.pyplot as plt
import pandas as pd
import scipy
from matplotlib import transforms
from numpy import mean, int64
from transpose_dict import TD
from scipy.stats import sem, kendalltau, rankdata
import numpy as np
import seaborn as sns
figuredirname = os.sep+"figures"+os.sep

class Postprocessing():
    def __init__(self):
        if os.path.isdir(os.getcwd() + figuredirname) == False:
            os.mkdir(os.getcwd() + figuredirname)
        pass

    def realworld_benchmarks_visualise(self, analysis0_results: pd.DataFrame):
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
            plt.savefig(os.getcwd()+figuredirname+'realworld_benchmarks_bar_plot_with_error_bars.png')
            plt.show()

    def realworld_benchmarks_bootstrapping_visualise(self, analysis1_results: pd.DataFrame):
        score_names = analysis1_results.index
        for score_name in score_names:
            y = [np.mean(analysis1_results[alg][score_name]) for alg in analysis1_results.columns]

            y_err = [scipy.stats.sem(analysis1_results[alg][score_name]) for alg
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
            plt.savefig(os.getcwd()+figuredirname+'realworld_benchmarks_bootstrapped_bar_plot_with_error_bars.png')
            plt.show()

    def meta_simulation_visualise(self, results: list):
        list_of_ntrue_accuracies = results[0]
        list_of_npractitioner_accuracies = results[1]
        list_of_nsl_accuracies = results[2]
        n_true_repetitions = results[3]
        n_practitioner_repititions = results[4]
        n_sl_repititions = results[5]

        # Create boxplot for relative performance per pathways across all repetition for all ml methods
        sl_list = ["hc", "tabu", "rsmax2", "mmhc","h2pc", "gs","pc.stable"]
        ml_methods = ["DecisionTreeClassifier", "RandomForestClassifier", "KNeighborsClassifier","GradientBoostingClassifier", "SVCLinear", "LogisticLASSO","MLPClassifier"]
        sns.set_style('whitegrid')
        sns.set(rc={"figure.figsize": (10, 10)})

        # Placeholder for boxploting results, saving operation is done at the end of the function
        performances = pd.DataFrame(columns=['est-true', 'est-to-avgml','ML_method', 'SL'])
        for practitioner_rep in range(0, len(list_of_npractitioner_accuracies)):
            ml_method_performance_list = []
            practitioner_rep_ml_methods = list_of_npractitioner_accuracies[practitioner_rep]
            for item in practitioner_rep_ml_methods.values(): ml_method_performance_list.append(item["balanced_accuracy_score"])
            avg_ml_rep_performance = mean(ml_method_performance_list)
            for ml in ml_methods:
                true_rep_ml_performance = list_of_ntrue_accuracies[ml][practitioner_rep]
                performances = performances.append({'est-true': (practitioner_rep_ml_methods[ml]['balanced_accuracy_score'] - true_rep_ml_performance), 'est-to-avgml': practitioner_rep_ml_methods[ml]['balanced_accuracy_score'] - avg_ml_rep_performance,'ML_method': ml, 'SL': 'limited-real'},ignore_index=True)
        for sl in sl_list:
            for ml in ml_methods:
                for sl_rep in range(0, n_sl_repititions):
                    temp = []
                    for ml_avg in ml_methods:
                        temp.append(list_of_nsl_accuracies[sl][ml_avg][sl_rep])
                    avg_ml_rep_performance = mean(temp)
                    sl_rep_ml_performance = list_of_nsl_accuracies[sl][ml][sl_rep]
                    true_rep_ml_performance = list_of_ntrue_accuracies[ml][sl_rep]
                    performances = performances.append({'est-true': (sl_rep_ml_performance - true_rep_ml_performance), 'est-to-avgml': (sl_rep_ml_performance - avg_ml_rep_performance),'ML_method': ml, 'SL': sl},ignore_index=True)

        # Create absolute match proportion bar graph between pathway's top ml method and true ml method
        matches_from_practitioner_limited_world = 0
        sl_match_counts = {"hc": 0, "tabu": 0, "rsmax2": 0, "mmhc": 0, "h2pc": 0, "gs": 0,"pc.stable": 0}# ,"iamb":0,"fast.iamb":0,"iamb.fdr":0}
        for repetition in range(0, n_true_repetitions):
            list_of_comparable_ranks_in_one_repitition = {"DecisionTreeClassifier": 0, "RandomForestClassifier": 0,"KNeighborsClassifier": 0, "GradientBoostingClassifier": 0,"SVCLinear": 0, "LogisticLASSO": 0, "MLPClassifier": 0}
            for ml in ml_methods:
                list_of_comparable_ranks_in_one_repitition[ml] = list_of_ntrue_accuracies[ml][repetition]
                list_of_npractitioner_accuracies[repetition][ml] = list_of_npractitioner_accuracies[repetition][ml]['balanced_accuracy_score']
            if max(list_of_npractitioner_accuracies[repetition],key=list_of_npractitioner_accuracies[repetition].get) == max(list_of_comparable_ranks_in_one_repitition,key=list_of_comparable_ranks_in_one_repitition.get):
                matches_from_practitioner_limited_world += 1
        for sl in sl_match_counts.keys():
            sl_match_count = 0
            for repetition in range(0, n_true_repetitions):
                list_of_comparable_ranks_in_one_repitition = {"DecisionTreeClassifier": 0, "RandomForestClassifier": 0,"KNeighborsClassifier": 0,"GradientBoostingClassifier": 0, "SVCLinear": 0,"LogisticLASSO": 0, "MLPClassifier": 0}
                list_of_comparable_ranks_in_one_repitition_b = {"DecisionTreeClassifier": 0,"RandomForestClassifier": 0, "KNeighborsClassifier": 0,"GradientBoostingClassifier": 0, "SVCLinear": 0,"LogisticLASSO": 0, "MLPClassifier": 0}
                for ml in ml_methods:
                    list_of_comparable_ranks_in_one_repitition[ml] = list_of_ntrue_accuracies[ml][repetition]
                    list_of_comparable_ranks_in_one_repitition_b[ml] = list_of_nsl_accuracies[sl][ml][repetition]
                if max(list_of_comparable_ranks_in_one_repitition_b,key=list_of_comparable_ranks_in_one_repitition_b.get) == max(list_of_comparable_ranks_in_one_repitition, key=list_of_comparable_ranks_in_one_repitition.get):
                    sl_match_count += 1
            sl_match_counts[sl] = sl_match_count / n_true_repetitions
        sl_match_counts["limited-real"] = matches_from_practitioner_limited_world / n_true_repetitions
        fig = plt.figure(figsize=(10, 10))
        plt.bar(list(sl_match_counts.keys()), list(sl_match_counts.values()), color='maroon', width=0.4)
        plt.xlabel("Technique")
        plt.ylabel("Percentage of correctly recommended ML methods")
        plt.title("Proportion of matched top ML methods from technique to true top benchmarks")
        plt.style.use("seaborn")
        plt.tight_layout()
        plt.savefig(os.getcwd() + figuredirname + 'interworld_benchmarks_barplot_proportion_matches_max_method.png')
        plt.show()

        # Create scatterplot of relative performances for every repetition for all ml methods between alternative pathways
        list_of_xy_pairs_all_methods_limited_world = []
        mean_list_of_xy_pairs_all_methods_limited_world = []
        dict_of_xy_pairs_sl = {"hc": [], "tabu": [], "rsmax2": [], "mmhc": [], "h2pc": [], "gs": [],"pc.stable": []}  # ,"iamb":[],"fast.iamb":[],"iamb.fdr":[]}
        mean_dict_of_xy_pairs_sl = {"hc": [], "tabu": [], "rsmax2": [], "mmhc": [], "h2pc": [], "gs": [],"pc.stable": []}  # ,"iamb":[],"fast.iamb":[],"iamb.fdr":[]}
        ml_methods = ["DecisionTreeClassifier", "RandomForestClassifier", "KNeighborsClassifier","GradientBoostingClassifier", "SVCLinear", "LogisticLASSO", "MLPClassifier"]

        temp = []
        mean_list_of_npractitioner_accuracies = {"DecisionTreeClassifier": 0, "RandomForestClassifier": 0, "KNeighborsClassifier": 0,"GradientBoostingClassifier": 0, "SVCLinear": 0, "LogisticLASSO": 0,"MLPClassifier": 0}
        mean_list_of_ntrue_accuracies = {"DecisionTreeClassifier": 0, "RandomForestClassifier": 0, "KNeighborsClassifier": 0,"GradientBoostingClassifier": 0,  "SVCLinear": 0, "LogisticLASSO": 0, "MLPClassifier": 0}
        mean_list_of_nsl_accuracies = {"hc": {"DecisionTreeClassifier": [], "RandomForestClassifier": [], "KNeighborsClassifier": [],"GradientBoostingClassifier": [], "SVCLinear": [], "LogisticLASSO": [],"MLPClassifier": []},"tabu": {"DecisionTreeClassifier": [], "RandomForestClassifier": [], "KNeighborsClassifier": [],"GradientBoostingClassifier": [], "SVCLinear": [], "LogisticLASSO": [],"MLPClassifier": []},"rsmax2": {"DecisionTreeClassifier": [], "RandomForestClassifier": [], "KNeighborsClassifier": [],"GradientBoostingClassifier": [], "SVCLinear": [], "LogisticLASSO": [],"MLPClassifier": []},"mmhc": {"DecisionTreeClassifier": [], "RandomForestClassifier": [], "KNeighborsClassifier": [],"GradientBoostingClassifier": [], "SVCLinear": [], "LogisticLASSO": [],"MLPClassifier": []},"h2pc": {"DecisionTreeClassifier": [], "RandomForestClassifier": [], "KNeighborsClassifier": [],"GradientBoostingClassifier": [],"SVCLinear": [], "LogisticLASSO": [],"MLPClassifier": []},"gs": {"DecisionTreeClassifier": [], "RandomForestClassifier": [], "KNeighborsClassifier": [],"GradientBoostingClassifier": [],  "SVCLinear": [], "LogisticLASSO": [],"MLPClassifier": []},"pc.stable": {"DecisionTreeClassifier": [], "RandomForestClassifier": [], "KNeighborsClassifier": [],"GradientBoostingClassifier": [], "SVCLinear": [], "LogisticLASSO": [],"MLPClassifier": []}}

        for ml in ml_methods:
            for repetition in range(0, n_practitioner_repititions):
                temp.append(list_of_npractitioner_accuracies[repetition][ml])
            mean_list_of_npractitioner_accuracies[ml] = mean(temp)
            temp.clear()

        for ml in ml_methods:
            mean_list_of_ntrue_accuracies[ml] = mean(list_of_ntrue_accuracies[ml])
        for sl in sl_list:
            for ml in ml_methods:
                mean_list_of_nsl_accuracies[sl][ml] = mean(list_of_nsl_accuracies[sl][ml])

        for ml_index, ml_method_label in enumerate(ml_methods):
            mean_list_of_xy_pairs_all_methods_limited_world.append( (mean_list_of_ntrue_accuracies[ml_method_label], mean_list_of_npractitioner_accuracies[ml_method_label]))
        for sl in mean_dict_of_xy_pairs_sl.keys():
            for ml_index, ml_method_label in enumerate(ml_methods):
                mean_dict_of_xy_pairs_sl[sl].append((mean_list_of_ntrue_accuracies[ml_method_label], mean_list_of_nsl_accuracies[sl][ml_method_label]))

        for repetition in range(0, n_practitioner_repititions):
            for ml_index, ml_method_label in enumerate(ml_methods):
                list_of_xy_pairs_all_methods_limited_world.append( (mean(list_of_ntrue_accuracies[ml_method_label]), list_of_npractitioner_accuracies[repetition][ml_method_label]))
        for sl in dict_of_xy_pairs_sl.keys():
            for repetition in range(0, n_practitioner_repititions):
                for ml_index, ml_method_label in enumerate(ml_methods):
                    dict_of_xy_pairs_sl[sl].append( (mean(list_of_ntrue_accuracies[ml_method_label]),list_of_nsl_accuracies[sl][ml_method_label][repetition]))
        sl_colors = {"hc": 'red', "tabu": 'blue', "rsmax2": 'green', "mmhc": 'yellow', "h2pc": 'orange', "gs": "cyan","pc.stable": "brown"}  # , "iamb": "magenta", "fast.iamb": "peru", "iamb.fdr": "pink"}
        sl_offset = {"hc": 2, "tabu": 4, "rsmax2": 6, "mmhc": 8, "h2pc": 10, "gs": 12,"pc.stable": 14}
        plt.figure(figsize=(10, 10))
        for sl in dict_of_xy_pairs_sl.keys():
            offset = lambda p: transforms.ScaledTranslation(p / 72., 0, plt.gcf().dpi_scale_trans)
            trans = plt.gca().transData
            x_true_accuracies, y_sl_accuracies = zip(*dict_of_xy_pairs_sl[sl])
            corr, _ = kendalltau(x_true_accuracies, y_sl_accuracies)
            plt.scatter(x=x_true_accuracies, y=y_sl_accuracies, c=sl_colors[sl], s=10, label=sl + ' ('+ str(round(corr,2))+')', alpha=0.7, transform=trans+offset(sl_offset[sl]))
        plt.title("Scatterplot of x-y pairs between true and benchmarked accuracies for every repetition for all ml methods")
        plt.xlabel("Accuracy (True Reference)")
        plt.ylabel("Accuracy (Estimated Performance)")
        x_true_accuracies, y_limited_accuracies = zip(*list_of_xy_pairs_all_methods_limited_world)
        corr, _ = kendalltau(x_true_accuracies, y_limited_accuracies)
        plt.scatter(x=x_true_accuracies, y=y_limited_accuracies, c='black', label='limited-real ('+str(round(corr,2))+')', s=15,marker='x', alpha=1)
        plt.legend(loc='upper left')
        for sl in mean_dict_of_xy_pairs_sl.keys():
            sorted_result = sorted(mean_dict_of_xy_pairs_sl[sl])
            mean_x_true_accuracies, mean_y_sl_accuracies = zip(*sorted_result)
            plt.plot(mean_x_true_accuracies, mean_y_sl_accuracies, color=sl_colors[sl], linestyle='dashed',linewidth=0.3)
        plt.style.use("seaborn")
        sorted_result = sorted(mean_list_of_xy_pairs_all_methods_limited_world)
        mean_x_true_accuracies, mean_y_limited_accuracies = zip(*sorted_result)

        rank_methods = {'DecisionTreeClassifier': 0, 'RandomForestClassifier': 0, 'KNeighborsClassifier': 0,'GradientBoostingClassifier': 0, 'SVCLinear': 0, 'LogisticLASSO': 0, 'MLPClassifier': 0}
        for idx, item in enumerate(ml_methods):
            rank_methods[item] = mean_list_of_xy_pairs_all_methods_limited_world[idx][0]
        rank_methods = sorted(rank_methods.items(), key=lambda kv: (kv[1], kv[0]))
        ordered_list_of_methods = []
        for item in rank_methods:
            ordered_list_of_methods.append(item[0]) #grab the true rank performance, [1] would retreive limited performance
        for ml_index, ml_method_label in enumerate(ml_methods):
            plt.text(x=x_true_accuracies[ml_index], y=y_limited_accuracies[ml_index], s=ml_method_label, fontsize=8,horizontalalignment='center', verticalalignment='bottom', alpha=0.8)
        plt.tight_layout()
        plt.savefig(os.getcwd() + figuredirname + 'interworld_benchmarks_scatterplot_every_repetition_accuracies_all_methods.png')
        plt.show()

        # Boxplot calculation and saving
        performances = pd.DataFrame(columns=['est-true', 'est-to-avgml', 'ML_method', 'SL'])
        for practitioner_rep in range(0, len(list_of_npractitioner_accuracies)):
            ml_method_performance_list = []
            practitioner_rep_ml_methods = list_of_npractitioner_accuracies[practitioner_rep]
            for item in practitioner_rep_ml_methods.values(): ml_method_performance_list.append(item)
            avg_ml_rep_performance = mean(ml_method_performance_list)
            for ml in ml_methods:
                true_rep_ml_performance = list_of_ntrue_accuracies[ml][practitioner_rep]
                performances = performances.append(
                    {'est-true': (practitioner_rep_ml_methods[ml] - true_rep_ml_performance),
                     'est-to-avgml': practitioner_rep_ml_methods[ml] - avg_ml_rep_performance, 'ML_method': ml,
                     'SL': 'limited-real'}, ignore_index=True)
        for sl in sl_list:
            for ml in ml_methods:
                for sl_rep in range(0, n_sl_repititions):
                    temp = []
                    for ml_avg in ml_methods:
                        temp.append(list_of_nsl_accuracies[sl][ml_avg][sl_rep])
                    avg_ml_rep_performance = mean(temp)
                    sl_rep_ml_performance = list_of_nsl_accuracies[sl][ml][sl_rep]
                    true_rep_ml_performance = list_of_ntrue_accuracies[ml][sl_rep]
                    performances = performances.append({'est-true': (sl_rep_ml_performance - true_rep_ml_performance),
                                                        'est-to-avgml': (sl_rep_ml_performance - avg_ml_rep_performance),
                                                        'ML_method': ml, 'SL': sl}, ignore_index=True)
        plt.figure(figsize=(10, 10))
        ax = sns.boxplot(x=performances['ML_method'], y=performances['est-true'], hue=performances['SL'],showfliers=False, order=ordered_list_of_methods)
        plt.title('Boxplot of difference between estimated performances and true by ml method for strategies')
        plt.ylabel('Difference to True estimated performance')
        plt.xlabel('ML Method (least performing to best)')
        added_methods = []
        for idx, item in enumerate(rank_methods):
            added_methods.append(rank_methods[idx][0] + " (True perf: "+ str(round(rank_methods[idx][1],2))+")" )
            ax.get_xticklabels()
        ax.set_xticklabels(added_methods, rotation=90)
        plt.tight_layout()
        plt.savefig(os.getcwd() + figuredirname + 'interworld_benchmarks_boxplot_all_ml_methods_diff_to_true.png')
        plt.show()

        plt.figure(figsize=(10, 10))
        ax = sns.boxplot(x=performances['ML_method'], y=performances['est-to-avgml'], hue=performances['SL'],
                         showfliers=False)
        plt.title('Boxplot of difference between estimated and avg ml performances by ml method for strategies')
        plt.ylabel('Difference to Avg ML performance')
        plt.xlabel('ML Method (least performing to best)')
        added_methods = []
        for idx, item in enumerate(rank_methods):
            added_methods.append(rank_methods[idx][0] + " (True perf:"+ str(round(rank_methods[idx][1],2))+")" )
            ax.get_xticklabels()
        ax.set_xticklabels(added_methods, rotation=90)
        plt.tight_layout()
        plt.savefig(os.getcwd() + figuredirname + 'interworld_benchmarks_boxplot_all_ml_methods_diff_to_avg_ml_performance.png')
        plt.show()

        # Save csv of performances for every repetition for all ml methods for all alternative pathways
        csv_performances = pd.DataFrame(columns=["DecisionTreeClassifier", "RandomForestClassifier", "KNeighborsClassifier","GradientBoostingClassifier", "SVCLinear", "LogisticLASSO", "MLPClassifier", "SL"])
        ranked_csv_performances = pd.DataFrame(columns=["DecisionTreeClassifier", "RandomForestClassifier", "KNeighborsClassifier","GradientBoostingClassifier", "SVCLinear", "LogisticLASSO", "MLPClassifier", "SL"])
        violinplot_performances = pd.DataFrame({'ML_Method': pd.Series(dtype='str'),
                   'rank': pd.Series(dtype='int'),
                   'SL': pd.Series(dtype='str')})
        for practitioner_rep in range(0, len(list_of_npractitioner_accuracies)):
            practitioner_rep_ml_methods = list_of_npractitioner_accuracies[practitioner_rep]
            csv_performances = csv_performances.append({'DecisionTreeClassifier': practitioner_rep_ml_methods["DecisionTreeClassifier"], 'RandomForestClassifier': practitioner_rep_ml_methods["RandomForestClassifier"], 'KNeighborsClassifier': practitioner_rep_ml_methods["KNeighborsClassifier"], 'GradientBoostingClassifier': practitioner_rep_ml_methods["GradientBoostingClassifier"], 'SVCLinear': practitioner_rep_ml_methods["SVCLinear"], 'LogisticLASSO': practitioner_rep_ml_methods["LogisticLASSO"],'MLPClassifier': practitioner_rep_ml_methods["MLPClassifier"]}, ignore_index=True)
            #csv_performances.to_csv('limited-real_performance_per_repetition.csv')
            ranked_performances = pd.DataFrame.from_dict([dict(zip(practitioner_rep_ml_methods.keys(), rankdata([-i for i in practitioner_rep_ml_methods.values()], method='min')))], dtype=int64)
            ranked_performances["SL"] = "limited-real"
            ranked_csv_performances = ranked_csv_performances.append(ranked_performances)
            for ml in ml_methods:
                violinplot_performances = violinplot_performances.append({'ML_Method':ml, 'rank': ranked_performances[ml].loc[0], "SL":'limited-real'}, ignore_index=True)
            #ranked_csv_performances.to_csv('limited-real_rank-performance_per_repetition.csv')
        for sl in sl_list:
            csv_performances = pd.DataFrame(columns=["DecisionTreeClassifier", "RandomForestClassifier", "KNeighborsClassifier","GradientBoostingClassifier", "SVCLinear", "LogisticLASSO", "MLPClassifier"])
            ranked_csv_performances = pd.DataFrame(columns=["DecisionTreeClassifier", "RandomForestClassifier", "KNeighborsClassifier","GradientBoostingClassifier", "SVCLinear", "LogisticLASSO", "MLPClassifier"])
            for sl_rep in range(0, n_sl_repititions):
                sl_accuracies = {"DecisionTreeClassifier": 0, "RandomForestClassifier": 0,"KNeighborsClassifier": 0, "GradientBoostingClassifier": 0,"SVCLinear": 0, "LogisticLASSO": 0, "MLPClassifier": 0}
                for ml in ml_methods:
                    sl_accuracies[ml] = list_of_nsl_accuracies[sl][ml][sl_rep]
                csv_performances = csv_performances.append({'DecisionTreeClassifier': list_of_nsl_accuracies[sl]["DecisionTreeClassifier"][sl_rep],'RandomForestClassifier': list_of_nsl_accuracies[sl]["RandomForestClassifier"][sl_rep],'KNeighborsClassifier': list_of_nsl_accuracies[sl]["KNeighborsClassifier"][sl_rep],'GradientBoostingClassifier': list_of_nsl_accuracies[sl]["GradientBoostingClassifier"][sl_rep],'SVCLinear': list_of_nsl_accuracies[sl]["SVCLinear"][sl_rep],'LogisticLASSO': list_of_nsl_accuracies[sl]["LogisticLASSO"][sl_rep],'MLPClassifier': list_of_nsl_accuracies[sl]["MLPClassifier"][sl_rep]}, ignore_index=True)
                #csv_performances.to_csv(sl+'_performance_per_repetition.csv')
                ranked_performances = pd.DataFrame.from_dict([dict(zip(sl_accuracies.keys(),rankdata([-i for i in sl_accuracies.values()],method='min')))],dtype=int64)
                ranked_performances["SL"] = sl
                ranked_csv_performances = ranked_csv_performances.append(ranked_performances)
                for ml in ml_methods:
                    violinplot_performances = violinplot_performances.append({'ML_Method': ml, 'rank': ranked_performances[ml].loc[0], "SL": sl}, ignore_index=True)
                #ranked_csv_performances.to_csv(sl+'_rank-performance_per_repetition.csv')
        for ml in ml_methods:
            list_of_ntrue_accuracies[ml] = mean(list_of_ntrue_accuracies[ml])
        for sl in sl_list:
            for ml in ml_methods:
                list_of_nsl_accuracies[sl][ml] = mean(list_of_nsl_accuracies[sl][ml])

        # Create violinplot of rank spread by ml method grouped by sl
        plt.figure(figsize=(10, 10))
        #for sl in sl_list:
        #    dataset = violinplot_performances.loc[violinplot_performances['SL'] == sl]
        #    ax = sns.violinplot(x=dataset["ML_Method"], y=dataset["rank"], data=dataset, inner=None,scale='width', cut=0)
        #    plt.legend()
        #    sns.move_legend(ax, "center left", bbox_to_anchor=(1, 0.5))
        #    plt.title("Density of ML method ranks for "+sl)
        #    ax.set_xticklabels(ml_methods, rotation=90)
        #    plt.tight_layout()
        #    plt.savefig(os.getcwd() + figuredirname + 'interworld_benchmarks_violinplot_every_repetition_rank_all_methods_'+sl+'.png')
        #    plt.show()
        ax = sns.violinplot(x=violinplot_performances["ML_Method"], y=violinplot_performances["rank"], hue=violinplot_performances["SL"], inner=None,scale='width', cut=0, order=ordered_list_of_methods)
        plt.legend()
        sns.move_legend(ax, "center left", bbox_to_anchor=(1, 0.5))
        plt.title("Density of ML method ranks by alternative strategies")
        ax.set_xticklabels(ordered_list_of_methods, rotation=90)
        plt.xlabel("ML Methods (least performing to best)")
        plt.tight_layout()
        plt.savefig(os.getcwd() + figuredirname + 'interworld_benchmarks_violinplot_every_repetition_rank_all_methods_composite.png')
        plt.show()

        # Create scatterplot of relative avg performances for all ml methods between alternative pathways
        list_of_xy_pairs_all_methods_limited_world = []
        dict_of_xy_pairs_sl = {"hc": [], "tabu": [], "rsmax2": [], "mmhc": [], "h2pc": [],"gs":[], "pc.stable":[]}#,"iamb":[],"fast.iamb":[],"iamb.fdr":[]}
        ml_methods = ["DecisionTreeClassifier","RandomForestClassifier","KNeighborsClassifier","GradientBoostingClassifier", "SVCLinear", "LogisticLASSO", "MLPClassifier"]
        avg_nlearning = {"DecisionTreeClassifier":0,"RandomForestClassifier":0,"KNeighborsClassifier":0,"GradientBoostingClassifier":0, "SVCLinear":0, "LogisticLASSO":0, "MLPClassifier":0}
        plt.figure(figsize=(10, 10))
        temp = []
        for ml in ml_methods:
            for repetition in range(0, n_practitioner_repititions):
                temp.append(list_of_npractitioner_accuracies[repetition][ml])
            avg_nlearning[ml] = mean(temp)
            temp.clear()
        for ml_index, ml_method_label in enumerate(ml_methods):
            list_of_xy_pairs_all_methods_limited_world.append( (list_of_ntrue_accuracies[ml_method_label],avg_nlearning[ml_method_label]))
        for sl in dict_of_xy_pairs_sl.keys():
           for ml_index, ml_method_label in enumerate(ml_methods):
               dict_of_xy_pairs_sl[sl].append( (list_of_ntrue_accuracies[ml_method_label],list_of_nsl_accuracies[sl][ml_method_label]))
        sl_colors = {"hc": 'red',"tabu": 'blue', "rsmax2": 'green', "mmhc": 'yellow',"h2pc": 'orange', "gs": "cyan", "pc.stable": "brown"}#, "iamb": "magenta", "fast.iamb": "peru", "iamb.fdr": "pink"}
        for sl in dict_of_xy_pairs_sl.keys():
           x_true_accuracies, y_sl_accuracies = zip(*dict_of_xy_pairs_sl[sl])
           plt.scatter(x=x_true_accuracies, y=y_sl_accuracies, c=sl_colors[sl], label=sl, alpha=0.7)
        plt.title("Scatterplot of x-y pairs between true and benchmarked avg accuracies for all ml methods")
        plt.xlabel("Accuracy (True Reference)")
        plt.ylabel("Accuracy (Estimated Performance)")
        x_true_accuracies, y_sl_accuracies = zip(*list_of_xy_pairs_all_methods_limited_world)
        #plt.scatter(x=x_true_accuracies, y=y_sl_accuracies, c='black', label='limited-real', marker='x', alpha=1)
        for ml_index, ml_method_label in enumerate(ml_methods):
            plt.axvline(x=x_true_accuracies[ml_index], color='black', ls='--', lw=0.3)
            plt.text(x=x_true_accuracies[ml_index], y=y_sl_accuracies[ml_index], s=ml_method_label, fontsize=8, horizontalalignment='center',verticalalignment='bottom', alpha=0.8)
        plt.legend(loc='upper left')
        plt.style.use("seaborn")
        plt.tight_layout()
        plt.savefig(os.getcwd() + figuredirname + 'interworld_benchmarks_scatterplot_avg_accuracies_all_methods.png')
        plt.show()

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
