import argparse
import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import nltk
import os

# the default setting
# under 3 datasets
# with temperature = 1
dataset_ = ['CodeContests', 'APPS', 'HumanEval']
dataset = ['CodeContests', '       APPS       ', 'HumanEval']
temperature = 1
fig_save_dir = './fig/'

# draw the max_diff figure
def draw_max_diff(max_diff, target):
    # draw structual similarity
    if target == 'structual_similarity':
        tmp = []
        for i in range(len(max_diff)):
            for case in max_diff[i]['UnifiedDiff']:
                tmp.append([case, dataset[i], 'UnifiedDiff'])
            for case in max_diff[i]['TreeDiff']:
                tmp.append([case, dataset[i], 'TreeDiff'])

        for i in range(len(dataset)):
            for key in max_diff[i]:
                print('Dataset: %s, %s, Mean of max diff %s' % (
                dataset[i], key, np.mean([x[0] for x in tmp if x[1] == dataset[i] and x[2] == key])))
        tmp = pd.DataFrame.from_records(tmp, columns=['max_diff', 'dataset', 'type'])
        p1 = sns.boxplot(data=tmp, x='dataset', y='max_diff', hue='type', width=0.4)
        p1.set(xlabel=None)
        p1.set(ylabel=None)
        p1.set_xticklabels(p1.get_xticklabels(), fontsize=18)
        p1.tick_params(axis='y', labelsize=18)
        # p1.legend(fontsize=18)
        sns.move_legend(
            p1, "lower center",
            bbox_to_anchor=(10, 1), ncol=3, title=None, frameon=False,
            fontsize=18
        )

    elif target == 'output_feature':
        tmp = []
        for i in range(len(max_diff)):
            for key in max_diff[i]:
                for case in max_diff[i][key]:
                    if key == 'correct':
                        tmp.append([case, dataset[i], 'executable & same'])
                    elif key == 'exception':
                        tmp.append([case, dataset[i], 'other exception'])
                    elif key == 'execution_error':
                        tmp.append([case, dataset[i], 'execution error'])
                    else:
                        tmp.append([case, dataset[i], key])

        for i in range(len(dataset)):
            for key in max_diff[i]:
                if key == 'correct':
                    a = 'executable & same'
                elif key == 'exception':
                    a = 'other exception'
                elif key == 'execution_error':
                    a = 'execution error'
                else:
                    a = key
                print('Dataset: %s, %s, Mean of max diff %s' % (dataset[i], a, np.mean([x[0] for x in tmp if x[1] == dataset[i] and x[2]==a])))

        tmp = pd.DataFrame.from_records(tmp, columns=['max_diff', 'dataset', 'type'])
        p1 = sns.boxplot(data=tmp, x='dataset', y='max_diff', hue='type', width=0.4)
        p1.set(xlabel=None)
        p1.set(ylabel=None)
        p1.set_xticklabels(p1.get_xticklabels(), fontsize=18)
        p1.tick_params(axis='y', labelsize=18)
        sns.move_legend(
            p1, "lower center",
            bbox_to_anchor=(.5, 1), ncol=3, title=None, frameon=False,
            fontsize=18
        )
    elif target == 'Levenshtein_edit_distance':
        tmp = []
        for i in range(len(max_diff)):
            for case in max_diff[i]:
                tmp.append([case, dataset[i]])

        # print('Test case pass rate, %s' % 'max_diff', flush=True)

        for i in range(len(dataset)):
            # print(sum([x[0] for x in tmp if x[1] == dataset[i] and int(x[0])==1]))
            # print("Dataset: %s, %s could reach 100% max diff" % (dataset[i], sum([x[0] for x in tmp if x[1] == dataset[i] and int(x[0])==1])))
            print('Dataset: %s, Mean of max diff %s' % (dataset[i], np.mean([x[0] for x in tmp if x[1] == dataset[i]])))
        tmp = pd.DataFrame.from_records(tmp, columns=['max_diff', 'dataset'])
        p1 = sns.boxplot(x=tmp['dataset'], y=tmp['max_diff'], width=0.4)
        p1.set(xlabel=None)
        p1.set(ylabel=None)
        # point
        p1.set_xticklabels(p1.get_xticklabels(), fontsize=18)
        p1.tick_params(axis='y', labelsize=18)
        plt.xticks(rotation=25)
        plt.gca().set(ylim=(-50, 650))
    else:
        tmp = []
        for i in range(len(max_diff)):
            for case in max_diff[i]:
                tmp.append([case, dataset[i]])

        # print('Test case pass rate, %s' % 'max_diff', flush=True)

        for i in range(len(dataset)):
            # print(sum([x[0] for x in tmp if x[1] == dataset[i] and int(x[0])==1]))
            # print("Dataset: %s, %s could reach 100% max diff" % (dataset[i], sum([x[0] for x in tmp if x[1] == dataset[i] and int(x[0])==1])))
            print('Dataset: %s, Mean of max diff %s' % (dataset[i], np.mean([x[0] for x in tmp if x[1] == dataset[i]])))
        tmp = pd.DataFrame.from_records(tmp, columns=['max_diff', 'dataset'])
        p1 = sns.boxplot(x=tmp['dataset'], y=tmp['max_diff'], width=0.4)
        p1.set(xlabel=None)
        p1.set(ylabel=None)
        # point
        p1.set_xticklabels(p1.get_xticklabels(), fontsize=18)
        p1.tick_params(axis='y', labelsize=18)
        plt.xticks(rotation=25)

# this function is for draw Mean and Variance
def draw(x, target, option='var'):
    if target == 'structual_similarity':
        tmp = []
        for i in range(len(x)):
            for case in x[i]['structual_similarity_UnifiedDiff']:
                if case:
                    if option == 'var':
                        var = np.var([i[0] for i in case])
                        tmp.append([var, dataset[i], 'UnifiedDiff'])
                    elif option == 'mean':
                        mean = np.mean([i[0] for i in case])
                        tmp.append([mean, dataset[i], 'UnifiedDiff'])

            for case in x[i]['structual_similarity_TreeDiff']:
                if case:
                    if option == 'var':
                        var = np.var([i[0] for i in case])
                        tmp.append([var, dataset[i], 'TreeDiff'])
                    elif option == 'mean':
                        mean = np.mean([i[0] for i in case])
                        tmp.append([mean, dataset[i], 'TreeDiff'])
            if option == 'var':
                print('UnitedDiff, Dataset: %s, %s' % (dataset[i], option), flush=True)
                print(np.mean([j[0] for j in tmp if j[1] == dataset[i] and j[2] == 'UnifiedDiff']), flush=True)
                print('TreeDiff, Dataset: %s, %s' % (dataset[i], option), flush=True)
                print(np.mean([j[0] for j in tmp if j[1] == dataset[i] and j[2] == 'TreeDiff']), flush=True)

            elif option == 'mean':
                print('UnitedDiff, Dataset: %s, %s' % (dataset[i], option), flush=True)
                print(np.mean([j[0] for j in tmp if j[1] == dataset[i] and j[2] == 'UnifiedDiff']), flush=True)
                print('TreeDiff, Dataset: %s, %s' % (dataset[i], option), flush=True)
                print(np.mean([j[0] for j in tmp if j[1] == dataset[i] and j[2] == 'TreeDiff']), flush=True)

        tmp = pd.DataFrame.from_records(tmp, columns=[option, 'dataset', 'type'])
        p1 = sns.boxplot(data=tmp, x='dataset', y=option, hue='type', width=0.4)
        p1.set(xlabel=None)
        p1.set(ylabel=None)
        p1.set_xticklabels(p1.get_xticklabels(), fontsize=18)
        p1.tick_params(axis='y', labelsize=18)
        if option == 'mean':
            sns.move_legend(
                p1, "lower center",
                bbox_to_anchor=(.5, 1), ncol=3, title=None, frameon=False,
                fontsize=18
            )
        else:
            sns.move_legend(
                p1, "lower center",
                bbox_to_anchor=(10, 1), ncol=3, title=None, frameon=False,
                fontsize=18
            )
    elif target == 'structual_similarity worst':
        tmp = []
        for i in range(len(x)):
            for case in x[i]['structual_similarity_UnifiedDiff']:
                if case:
                    if option == 'var':
                        var = np.var([i[0] for i in case])
                        tmp.append([var, dataset[i], 'UnifiedDiff'])
                    elif option == 'mean':
                        min_v = min([i[0] for i in case])
                        tmp.append([min_v, dataset[i], 'UnifiedDiff'])

            for case in x[i]['structual_similarity_TreeDiff']:
                if case:
                    if option == 'var':
                        var = np.var([i[0] for i in case])
                        tmp.append([var, dataset[i], 'TreeDiff'])
                    elif option == 'mean':
                        min_v = min([i[0] for i in case])
                        tmp.append([min_v, dataset[i], 'TreeDiff'])

        tmp = pd.DataFrame.from_records(tmp, columns=[option, 'dataset', 'type'])
        p1 = sns.boxplot(data=tmp, x='dataset', y=option, hue='type', width=0.4)
        p1.set(xlabel=None)
        p1.set(ylabel=None)
        p1.set_xticklabels(p1.get_xticklabels(), fontsize=18)
        p1.tick_params(axis='y', labelsize=18)
        sns.move_legend(
            p1, "lower center",
            bbox_to_anchor=(10, 1), ncol=3, title=None, frameon=False,
            fontsize=18
        )
    elif target == 'output_feature' or target == 'LCS worst' or target == 'LED worst':
        tmp = []
        for i in range(len(x)):
            for case in x[i]:
                # if case:
                if option == 'var':
                    tmp.append([np.var(case), dataset[i]])
                elif option == 'mean':
                    tmp.append([case, dataset[i]])

        tmp = pd.DataFrame.from_records(tmp, columns=[option, 'dataset'])
        p1 = sns.boxplot(x=tmp['dataset'], y=tmp[option], width=0.4)
        # p1 = sns.boxplot(data=tmp, x='dataset', y=option, hue='output_feature', width=0.4)
        p1.set(xlabel=None)
        p1.set(ylabel=None)
        p1.set_xticklabels(p1.get_xticklabels(), fontsize=18)
        p1.tick_params(axis='y', labelsize=18)
        plt.xticks(rotation=25)
    elif target == 'Levenshtein_edit_distance':
        tmp = []
        for i in range(len(x)):
            for case in x[i]:
                if case:
                    if option == 'var':
                        var = np.var(case)
                        tmp.append([var, dataset[i]])
                    elif option == 'mean':
                        mean = np.mean(case)
                        tmp.append([mean, dataset[i]])
        # print('Test case pass rate, %s' % option, flush=True)

        tmp = pd.DataFrame.from_records(tmp, columns=[option, 'dataset'])
        p1 = sns.boxplot(x=tmp['dataset'], y=tmp[option], width=0.4)
        p1.set(xlabel=None)
        p1.set(ylabel=None)
        p1.set_xticklabels(p1.get_xticklabels(), fontsize=18)
        p1.tick_params(axis='y', labelsize=18)
        plt.xticks(rotation=25)
        if option =='var':
            plt.gca().set(ylim=(-300, 5000))
        else:
            plt.gca().set(ylim=(-50, 650))
    elif target == 'LCS':
        tmp = []
        for i in range(len(x)):
            for case in x[i]:
                if case:
                    if option == 'var':
                        var = np.var(case)
                        tmp.append([var, dataset[i]])
                    elif option == 'mean':
                        mean = np.mean(case)
                        tmp.append([mean, dataset[i]])
        # print('Test case pass rate, %s' % option, flush=True)

        tmp = pd.DataFrame.from_records(tmp, columns=[option, 'dataset'])
        p1 = sns.boxplot(x=tmp['dataset'], y=tmp[option], width=0.4)
        p1.set(xlabel=None)
        p1.set(ylabel=None)
        p1.set_xticklabels(p1.get_xticklabels(), fontsize=18)
        p1.tick_params(axis='y', labelsize=18)
        plt.xticks(rotation=25)
        if option =='var':
            plt.gca().set(ylim=(-0.003, 0.065))
    else:
        tmp = []
        for i in range(len(x)):
            for case in x[i]:
                if case:
                    if option == 'var':
                        var = np.var(case)
                        tmp.append([var, dataset[i]])
                    elif option == 'mean':
                        mean = np.mean(case)
                        tmp.append([mean, dataset[i]])
        print('Test case pass rate, %s' % option, flush=True)
        for j in range(len(dataset)):
            print(dataset[j], np.mean([i[0] for i in tmp if i[1] == dataset[j]]))
        tmp = pd.DataFrame.from_records(tmp, columns=[option, 'dataset'])
        p1 = sns.boxplot(x=tmp['dataset'], y=tmp[option], width=0.4)
        p1.set(xlabel=None)
        p1.set(ylabel=None)
        p1.set_xticklabels(p1.get_xticklabels(), fontsize=18)
        p1.tick_params(axis='y', labelsize=18)
        plt.xticks(rotation=25)


# BLEU score correct
def generate_fig(target='test_case_pass_rate', palette='', save=False):
    if palette:
        sns.set_palette(palette)
    tmp_3 = []
    tmp_max_diff_3 = []
    # sns.set(style="ticks")
    for data in dataset_:
        with open('fig/%s_%s/intermediate_result_top0_5.json' % (data, temperature), 'r') as f:
        # with open('fig/%s_%s/intermediate_result_among5.json' % (data, temperature), 'r') as f:
            intermediate_result = json.load(f)
        if target == 'Levenshtein_edit_distance':
            if data == 'CodeContests':
                with open('rerun/%s_%s/rerun_intermediate_result_top0_5.json' % ('code_contest', temperature), 'r') as f:
                    intermediate_result = json.load(f)
            else:
                with open('rerun/%s_%s/rerun_intermediate_result_top0_5.json' % (data, temperature), 'r') as f:
                    intermediate_result = json.load(f)
            new_tmp = []
            for case in intermediate_result:
                new_tmp.append(intermediate_result[case]['syntatic_similarity']['Levenshtein_edit_distance'])
            tmp = new_tmp
        elif target == 'LED worst':
            if data == 'CodeContests':
                with open('rerun/%s_%s/rerun_intermediate_result_top0_5.json' % ('code_contest', temperature), 'r') as f:
                    intermediate_result = json.load(f)
            else:
                with open('rerun/%s_%s/rerun_intermediate_result_top0_5.json' % (data, temperature), 'r') as f:
                    intermediate_result = json.load(f)
            new_tmp = []
            for case in intermediate_result:
                new_tmp.append(max(intermediate_result[case]['syntatic_similarity']['Levenshtein_edit_distance']))
            tmp = new_tmp
        elif target == 'LCS':
            if data == 'CodeContests':
                with open('rerun/%s_%s/rerun_intermediate_result_top0_5.json' % ('code_contest', temperature),
                          'r') as f:
                    intermediate_result = json.load(f)
            else:
                with open('rerun/%s_%s/rerun_intermediate_result_top0_5.json' % (data, temperature), 'r') as f:
                    intermediate_result = json.load(f)
            new_tmp = []
            for case in intermediate_result:
                new_tmp.append(intermediate_result[case]['LCS'])
            tmp = new_tmp
        elif target == 'LCS worst':
            if data == 'CodeContests':
                with open('rerun/%s_%s/rerun_intermediate_result_top0_5.json' % ('code_contest', temperature),
                          'r') as f:
                    intermediate_result = json.load(f)
            else:
                with open('rerun/%s_%s/rerun_intermediate_result_top0_5.json' % (data, temperature), 'r') as f:
                    intermediate_result = json.load(f)
            new_tmp = []
            for case in intermediate_result:
                new_tmp.append(min(intermediate_result[case]['LCS']))
            tmp = new_tmp
        elif target == 'BLEU_score_between_5_list':
            tmp = intermediate_result['tmp_BLEU_score_between_5_list']
        elif target == 'structual_similarity' or target == 'structual_similarity worst':
            with open('fig/structural_similarity/%s_%s_structual_similarity_top0_5.json' % (data, temperature), 'r') as f:
            # with open('fig/structural_similarity/%s_%s_structual_similarity_among5.json' % (data, temperature), 'r') as f:
                tmp = {'structual_similarity_UnifiedDiff':[], 'structual_similarity_TreeDiff': []}
                problem_dic = json.load(f)
                for key in problem_dic:
                    a = problem_dic[key]['structual_similarity']['structual_similarity_UnifiedDiff']
                    if a != [-1,-1,-1,-1] and a != [-2,-2,-2,-2] and a != [-3,-3,-3,-3]:
                        tmp['structual_similarity_UnifiedDiff'].append(a)
                    a = problem_dic[key]['structual_similarity']['structual_similarity_TreeDiff']
                    if a != [-1,-1,-1,-1] and a != [-2,-2,-2,-2] and a != [-3,-3,-3,-3]:
                        tmp['structual_similarity_TreeDiff'].append(a)

        elif target == 'output_feature':
            if data == 'CodeContests':
                fig_dir = './rerun/%s_%s/' % ('code_contest', temperature)
            else:
                fig_dir = './rerun/%s_%s/' % (data, temperature)
            with open(fig_dir + 'rerun_intermediate_result_top0_5.json', 'r') as f:
                rerun_intermediate_result = json.load(f)
            OER = []
            OER_ow = []
            for case in rerun_intermediate_result:
                OER.append(rerun_intermediate_result[case]['syntatic_similarity']['same_output_between_5'])
                OER_ow.append(rerun_intermediate_result[case]['syntatic_similarity']['same_output_between_5_correct'])

            # tmp = OER
            tmp = OER_ow
        else:
            tmp = intermediate_result[target]
        tmp_max_diff = []
        if target == 'structual_similarity' or target == 'structual_similarity worst':
            tmp_max_diff = {'UnifiedDiff': [], 'TreeDiff': []}
            for case in tmp['structual_similarity_UnifiedDiff']:
                if case:
                    max_val = max([i[0] for i in case])
                    min_val = min([i[0] for i in case])
                    tmp_max_diff['UnifiedDiff'].append(max_val - min_val)
            for case in tmp['structual_similarity_TreeDiff']:
                if case:
                    max_val = max([i[0] for i in case])
                    min_val = min([i[0] for i in case])
                    tmp_max_diff['TreeDiff'].append(max_val - min_val)
            for key in tmp_max_diff:
                print('MAX')
                print(max(tmp_max_diff[key]))
        elif target == 'output_feature' or target == 'LCS worst' or target == 'LED worst':
            tmp_max_diff = {
            }
        else:
            for case in tmp:
                if case:
                    max_val = max(case)
                    min_val = min(case)
                    tmp_max_diff.append(max_val - min_val)
        tmp_max_diff_3.append(tmp_max_diff)
        tmp_3.append(tmp)

    if target == 'Levenshtein_edit_distance':
        figsize = (4, 4)
    elif target == 'structual_similarity' or target == 'structual_similarity worst':
        figsize = (5, 4)
    elif target == 'output_feature':
        figsize = (4, 4)
    else:
        figsize = (4, 4)
    # point
    fig = plt.figure(figsize=figsize)

    draw(tmp_3, target, 'mean')
    if target == 'output_feature':
        fig.subplots_adjust(top=0.95, bottom=0.25, left=0.2)
    elif target == 'structual_similarity' or target == 'structual_similarity worst':
        fig.subplots_adjust(top=0.75, bottom=0.1, left=0.15)
    else:
        fig.subplots_adjust(top=0.95, bottom=0.25, left=0.2)
    if save:
        fig.savefig(fig_save_dir + '%s_%s.pdf' % (target, 'mean'))
    else:
        fig.show()

    # fig = plt.figure(figsize=figsize)
    # draw(tmp_3, target, 'var')
    #
    # if target == 'output_feature':
    #     fig.subplots_adjust(top=0.75, bottom=0.1, left=0.1)
    # elif target == 'structual_similarity':
    #     fig.subplots_adjust(top=0.75, bottom=0.1, left=0.15)
    # else:
    #     fig.subplots_adjust(top=0.95, bottom=0.25, left=0.2)
    # if save:
    #     fig.savefig(fig_save_dir + '%s_%s.pdf' % (target, 'var'))
    # else:
    #     fig.show()
    # fig = plt.figure(figsize=figsize)
    #
    # draw_max_diff(tmp_max_diff_3, target)
    #
    # if target == 'output_feature':
    #     fig.subplots_adjust(top=0.95, bottom=0.1, left=0.1)
    # elif target == 'structual_similarity':
    #     fig.subplots_adjust(top=0.75, bottom=0.1, left=0.15)
    # else:
    #     fig.subplots_adjust(top=0.95, bottom=0.25, left=0.2)
    #
    # if save:
    #     fig.savefig(fig_save_dir + '%s_%s.pdf' % (target, 'max_diff'))
    # else:
    #     fig.show()

# tmp_3 = []
# for data in dataset_:
#     if data == 'CodeContests':
#         fig_dir = './rerun/%s_%s/' % ('code_contest', temperature)
#     else:
#         fig_dir = './rerun/%s_%s/' % (data, temperature)
#     with open(fig_dir + 'rerun_intermediate_result_top0_5.json', 'r') as f:
#         rerun_intermediate_result = json.load(f)
#     OER = []
#     OER_ow = []
#     for case in rerun_intermediate_result:
#         OER.append(rerun_intermediate_result[case]['syntatic_similarity']['same_output_between_5'])
#         OER_ow.append(rerun_intermediate_result[case]['syntatic_similarity']['same_output_between_5_correct'])
#
#     # tmp = OER
#     tmp = OER
#     tmp_3.append(tmp)