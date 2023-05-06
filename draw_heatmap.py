import json
import re
import numpy as np
import scipy.stats as stats
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def response_2_code(response):
    code_template = re.compile('```.*\n([\s\S]+?)\n```', re.M)
    code = code_template.findall(response)
    if len(code) > 0:
        return code[-1]
    else:
        return ''

def shift(test_pass_rate):
    list_5 = []
    count = 0
    while count<len(test_pass_rate[0]):
        tmp = []
        for i in range(len(test_pass_rate)):
            tmp.append(test_pass_rate[i][count])
        list_5.append(tmp)
        count += 1
    return list_5

# avg_test_pass_rate = [np.mean(i) for i in test_case_pass_rate]
def apply_test(test_pass_rate):
    a = shift(test_pass_rate)
    h_stat, p_value = stats.kruskal(a[0], a[1], a[2], a[3], a[4])

    # Print the results
    print("H-statistic: {:.2f}".format(h_stat))
    print("p-value: {:.4f}".format(p_value))

    # Determine if the p-value is significant at the 95% confidence level
    alpha = 0.05
    if p_value < alpha:
        print("At least one group has a different distribution from the others.")
    else:
        print("There is not enough evidence to suggest that the distributions of the five groups are significantly different.")

dataset_ = ['code_contest', 'APPS', 'HumanEval']
dataset = dataset_[1]
# dataset_0 = 'CodeContests'
request_way_ = ['R1', 'R2']
request_way = request_way_[0]
temperature_ = [0, 1, 2]
temperature = temperature_[1]
problem_list = []

if dataset == 'code_contest':
    # with open('./tmp2/code_contests_test.json', 'r') as f:
    with open('./dataset/code_contests_test.json', 'r') as f:
        problem_list = json.load(f)


elif dataset == 'HumanEval':
    with open('./HumanEval/HumanEval.jsonl', 'r') as f:
        for line in f.readlines():
            problem_list.append(json.loads(line))

elif dataset == 'APPS':
    path = './APPS/test/'
    for dirpath, dirnames, filenames in os.walk(path):
        # iterating for every problem
        for dirname in dirnames[:500]:
            # description
            with open(path + dirname + '/question.txt', 'r', encoding='utf-8') as f:
                description = f.read()
            problem_list.append({'name': dirname, 'description': description})
if dataset != 'code_contest':
    if request_way == 'R1':
        with open('fig/%s_%s/intermediate_result_among5.json' % (dataset, temperature), 'r') as f:
            intermediate_result = json.load(f)
    else:
        with open('fig/%s_%s/intermediate_result_top0_5.json' % (dataset, temperature), 'r') as f:
            intermediate_result = json.load(f)
else:
    if request_way == 'R1':
        with open('fig/%s_%s/intermediate_result_among5.json' % ('CodeContests', temperature), 'r') as f:
            intermediate_result = json.load(f)
    else:
        with open('fig/%s_%s/intermediate_result_top0_5.json' % ('CodeContests', temperature), 'r') as f:
            intermediate_result = json.load(f)

test_pass_rate = intermediate_result['test_case_pass_rate']

if dataset == 'code_contest':
    if request_way == 'R1':
        with open('fig/structural_similarity/%s_%s_structual_similarity_among5.json' % ('CodeContests', temperature), 'r') as f:
            problem_dic = json.load(f)
    else:
        with open('fig/structural_similarity/%s_%s_structual_similarity_top0_5.json' % ('CodeContests', temperature), 'r') as f:
            problem_dic = json.load(f)
else:
    if request_way == 'R1':
        with open('fig/structural_similarity/%s_%s_structual_similarity_among5.json' % (dataset, temperature), 'r') as f:
            problem_dic = json.load(f)
    else:
        with open('fig/structural_similarity/%s_%s_structual_similarity_top0_5.json' % (dataset, temperature), 'r') as f:
            problem_dic = json.load(f)

    # with open('fig/structural_similarity/%s_%s_structual_similarity_among5.json' % (data, temperature), 'r') as f:
tmp = {'structual_similarity_UnifiedDiff': [], 'structual_similarity_TreeDiff': []}

for key in problem_dic:
    a = problem_dic[key]['structual_similarity']['structual_similarity_UnifiedDiff']
    if a != [-1, -1, -1, -1] and a != [-2, -2, -2, -2] and a != [-3, -3, -3, -3]:
        tmp['structual_similarity_UnifiedDiff'].append(a)
    else:
        tmp['structual_similarity_UnifiedDiff'].append([[0],[0],[0],[0]])
    a = problem_dic[key]['structual_similarity']['structual_similarity_TreeDiff']
    if a != [-1, -1, -1, -1] and a != [-2, -2, -2, -2] and a != [-3, -3, -3, -3]:
        tmp['structual_similarity_TreeDiff'].append(a)
    else:
        tmp['structual_similarity_TreeDiff'].append([[0],[0],[0],[0]])
United_Diff = tmp['structual_similarity_UnifiedDiff']
Tree_Diff = tmp['structual_similarity_TreeDiff']

fig_save_dir = './rerun/%s_%s/' % (dataset, temperature)
if request_way == 'R1':
    with open(fig_save_dir+'rerun_intermediate_result_among5.json', 'r') as f:
        rerun_intermediate_result = json.load(f)
else:
    with open(fig_save_dir+'rerun_intermediate_result_top0_5.json', 'r') as f:
        rerun_intermediate_result = json.load(f)
OER = []
OER_ow = []
LCS = []
Levenshieten = []
if request_way == 'R1':
    Levenshieten.append(intermediate_result['tmp_syntatic_similarity']['Levenshtein_edit_distance'])
for case in rerun_intermediate_result:
    if request_way == 'R1':
        pass
    else:
        OER.append(rerun_intermediate_result[case]['syntatic_similarity']['same_output_between_5'])
        OER_ow.append(rerun_intermediate_result[case]['syntatic_similarity']['same_output_between_5_correct'])
        Levenshieten.append(rerun_intermediate_result[case]['syntatic_similarity']['Levenshtein_edit_distance'])

    LCS.append(rerun_intermediate_result[case]['LCS'])


problem_dic = {}
with open('./log/dataset_%s_model_gpt-3.5-turbo_topn_5_temperature_%s.0.log_%s' % (dataset, temperature, 0), 'r') as f:
    for line in f.readlines():
        tmp = json.loads(line)
        problem_dic[tmp['name']] = {'code_list': []}

for i in range(5):
    with open('./log/dataset_%s_model_gpt-3.5-turbo_topn_5_temperature_%s.0.log_%s' % (dataset, temperature, i), 'r') as f:
        for line in f.readlines():
            tmp = json.loads(line)
            if tmp['index'] == 0:
                problem_dic[tmp['name']]['code_list'].append(response_2_code(tmp['response']))
# test pass rate
# variance
# def generate_mean_var_max_diff(test_pass_rate):
test_pass_rate_var = [np.var(i) for i in test_pass_rate]
test_pass_rate_var_avg = np.mean(test_pass_rate_var)
test_pass_rate_var_above_list = []
for i in range(len(test_pass_rate_var)):
    if test_pass_rate_var[i] > test_pass_rate_var_avg:
        test_pass_rate_var_above_list.append(i)

# max diff
test_pass_rate_max_diff = [max(i)-min(i) for i in test_pass_rate]
test_pass_rate_max_diff_avg = np.mean(test_pass_rate_max_diff)
test_pass_rate_max_diff_above_list = []
for i in range(len(test_pass_rate_max_diff)):
    if test_pass_rate_max_diff[i] > test_pass_rate_max_diff_avg:
        test_pass_rate_max_diff_above_list.append(i)


def print_features(instable_problem):
    instable_problem_length = []
    instable_problem_difficulty = []
    instable_problem_time_limit = []
    instable_problem_cf_tags = []
    instable_problem_cf_rating = []
    for problem in instable_problem:
        instable_problem_length.append(len(problem['description']))
        instable_problem_difficulty.append(problem['difficulty'])
        pattern = re.compile(r'(?<=seconds:=)*\d+')
        time_limit = pattern.findall(problem['time_limit'].split('\n')[0])[0]
        if float(time_limit) < 100:
            instable_problem_time_limit.append(float(time_limit))
        if 'cf_tags' in problem:
            instable_problem_cf_tags.append(problem['cf_tags'])
        instable_problem_cf_rating.append(problem['cf_rating'])

    cf_tags_dic = {}

    for cf_tags in instable_problem_cf_tags:
        for cf_tag in list(set(cf_tags)):
            if cf_tag not in cf_tags_dic:
                cf_tags_dic[cf_tag] = 1
            else:
                cf_tags_dic[cf_tag] += 1


    sorted_dict = dict(sorted(cf_tags_dic.items(), key=lambda item: item[1], reverse=True))
    for key in sorted_dict:
        sorted_dict[key] = sorted_dict[key]/len(instable_problem_cf_tags)
    print('Mean of problem length: %s' % (np.mean(instable_problem_length)))
    print('Mean of problem difficulty: %s' % (np.mean(instable_problem_difficulty)))
    print('Mean of problem time limit: %s' % (np.mean(instable_problem_time_limit)))
    print('Mean of problem CF rating: %s' % (np.mean(instable_problem_cf_rating)))

    print('Distribution problem CF tags: %s' % (sorted_dict))

def key_word_extract(problem_list):
    tmp = []
    count = 0
    key_words = ['Example', 'example']
    # key_words = ['Example', 'example', '>>>', '=', 'return']
    # key_words = ['Example', 'example', '>>>', '=', 'return', 'sample', 'Sample']
    for problem in problem_list:
        flag = True
        for word in key_words:
            if dataset == 'HumanEval':
                if word in problem['prompt']:
                    flag=False
                    break
            else:
                if word in problem['description']:
                    flag=False
                    break
        if flag:
            count += 1
            if dataset == 'HumanEval':
                tmp.append(problem['prompt'])
            else:
                tmp.append(problem['description'])
    print(count)
    return tmp

def get_library(test_pass_rate, problem_dic):
    tmp_code_list = []

    for i in range(len(test_pass_rate)):
        for j in range(len(test_pass_rate[i])):
            if test_pass_rate[i][j] == 0:
                tmp_code_list.append([list(problem_dic.keys())[i],
                                      problem_dic[list(problem_dic.keys())[i]]['code_list'][j]])

    library = []
    for code in tmp_code_list:
        code_list = code[1].split('\n')
        for sentence in code_list:
            if 'import' in sentence:
                library.append(sentence)
    library = list(set(library))
    return library

correlation = {'problem': [],
               'test pass rate mean': [],
               'test pass rate variance': [],
               'test pass rate max diff': [],
               'description length': [],
               'difficulty': [],
               'time_limit': [],
               'cf_rating': []
               }

for i in range(len(problem_list)):
    problem = problem_list[i]


    if dataset == 'HumanEval':
        correlation['problem'].append(problem['task_id'])
        correlation['description length'].append(len(problem['prompt']))

    elif dataset == 'APPS':
        correlation['problem'].append(problem['name'])
        correlation['description length'].append(len(problem['description']))
    else:
        correlation['problem'].append(problem['name'])
        correlation['description length'].append(len(problem['description']))
        correlation['difficulty'].append(problem['difficulty'])

        pattern = re.compile(r'(?<=seconds:=)*\d+')
        time_limit = pattern.findall(problem['time_limit'].split('\n')[0])[0]
        if 'seconds' in problem['time_limit']:
            correlation['time_limit'].append(int(time_limit))
        else:
            correlation['time_limit'].append(3)
        correlation['cf_rating'].append(problem['cf_rating'])

    correlation['test pass rate mean'].append(np.mean(test_pass_rate[i]))
    correlation['test pass rate variance'].append(np.var(test_pass_rate[i]))
    correlation['test pass rate max diff'].append(max(test_pass_rate[i])-min(test_pass_rate[i]))

def old_heatmap():
    correlation['OER'] = OER
    correlation['OER_ow'] = OER_ow

    correlation['LCS mean'] = []
    # correlation['LCS variance'] = []
    correlation['LCS max diff'] = []

    correlation['Levenshieten mean'] = []
    # correlation['Levenshieten variance'] = []
    correlation['Levenshieten max diff'] = []

    correlation['United_Diff mean'] = []
    # correlation['United_Diff variance'] = []
    correlation['United_Diff max diff'] = []

    correlation['Tree_Diff mean'] = []
    # correlation['Tree_Diff variance'] = []
    correlation['Tree_Diff max diff'] = []

    for case in LCS:
        correlation['LCS mean'].append(np.mean(case))
        # correlation['LCS variance'].append(np.var(case))
        correlation['LCS max diff'].append(max(case)-min(case))

    for case in Levenshieten:
        correlation['Levenshieten mean'].append(np.mean(case))
        # correlation['Levenshieten variance'].append(np.var(case))
        correlation['Levenshieten max diff'].append(max(case)-min(case))

    for case in United_Diff:
        correlation['United_Diff mean'].append(np.mean([i[0] for i in case]))
        # correlation['United_Diff variance'].append(np.var([i[0] for i in case]))
        correlation['United_Diff max diff'].append(max([i[0] for i in case])-min([i[0] for i in case]))

    for case in Tree_Diff:
        correlation['Tree_Diff mean'].append(np.mean([i[0] for i in case]))
        # correlation['Tree_Diff variance'].append(np.var([i[0] for i in case]))
        correlation['Tree_Diff max diff'].append(max([i[0] for i in case])-min([i[0] for i in case]))


    correlation_rank = []
    for i in range(1, len(list(correlation.keys()))):
        for j in range(i+1, len(list(correlation.keys()))):
            correlation_rank.append([
                list(correlation.keys())[i],
                list(correlation.keys())[j],
                stats.pearsonr(correlation[list(correlation.keys())[i]], correlation[list(correlation.keys())[j]])
            ])
    # sorted_list = sorted(correlation_rank, key=lambda x: x[2][1])
    high_relavent = []
    problem_features = ['description length', 'difficulty', 'time_limit', 'cf_rating']
    for case in correlation_rank:
        if (case[0] in problem_features or case[1] in problem_features) and case[2][1]<0.05:
            high_relavent.append(case)
            # print('%s & %s\'s correlation: %s' % (list(correlation.keys())[i],
            #                                       list(correlation.keys())[j],
            #                                       stats.pearsonr(correlation[list(correlation.keys())[i]], correlation[list(correlation.keys())[j]])
            #                                       )
            #       )
    correlation_list = []
    # test pass rate
    correlation_list.append(correlation['test pass rate mean'])
    correlation_list.append(correlation['test pass rate variance'])
    correlation_list.append(correlation['test pass rate max diff'])
    # output equivalence rate
    correlation_list.append(correlation['OER'])
    correlation_list.append(correlation['OER_ow'])
    # LCS
    correlation_list.append(correlation['LCS mean'])
    # correlation_list.append(correlation['LCS variance'])
    correlation_list.append(correlation['LCS max diff'])
    # Levenshieten
    correlation_list.append(correlation['Levenshieten mean'])
    # correlation_list.append(correlation['Levenshieten variance'])
    correlation_list.append(correlation['Levenshieten max diff'])
    # United_Diff
    correlation_list.append(correlation['United_Diff mean'])
    # correlation_list.append(correlation['United_Diff variance'])
    correlation_list.append(correlation['United_Diff max diff'])
    # Tree_Diff
    correlation_list.append(correlation['Tree_Diff mean'])
    # correlation_list.append(correlation['Tree_Diff variance'])
    correlation_list.append(correlation['Tree_Diff max diff'])
    # problem features
    correlation_list.append(correlation['description length'])
    correlation_list.append(correlation['difficulty'])
    correlation_list.append(correlation['time_limit'])
    correlation_list.append(correlation['cf_rating'])
    column_names = ['test pass rate mean',
                    'test pass rate variance',
                    'test pass rate max diff',

                    'OER',
                    'OER_ow',

                    'LCS mean',
                    # 'LCS variance',
                    'LCS max diff',

                    'Levenshieten mean',
                    # 'Levenshieten variance',
                    'Levenshieten max diff',

                    'United_Diff mean',
                    # 'United_Diff variance',
                    'United_Diff max diff',

                    'Tree_Diff mean',
                    # 'Tree_Diff variance',
                    'Tree_Diff max diff',

                    'description length',
                    'difficulty',
                    'time_limit',
                    'cf_rating'
                    ]
    column_names_y = ['Mean',
                    'Variance',
                    'Max Diff',
                    'Description Length',
                    'Difficulty',
                    'Time Limit',
                    'CF Rating'
                    ]
    # column_names_x = [
    #                 '       Mean       ',
    #                 '     Variance     ',
    #                 '     Max Diff     ',
    #                 '  Prompt Length ',
    #                 '      Difficulty    ',
    #                 '    Time Limit    ',
    #                 '     CF Rating    '
    #                 ]
    column_names_x = ['M',
                    'V',
                    'MD',
                    'PL',
                    'D',
                    'TL',
                    'CR'
                    ]

    correlation_list = [[row[i] for row in correlation_list] for i in range(len(correlation_list[0]))]

    df = pd.DataFrame(correlation_list, columns=column_names)

    corr_matrix = df.corr()
    p_values = df.corr(method=lambda x, y: stats.pearsonr(x, y)[1])

    empty_df = pd.DataFrame(index=corr_matrix.index, columns=corr_matrix.columns)

    # Use the apply() function to replace the values in the heatmap
    # annotated_data = corr_matrix.applymap(lambda x: replace_value(x, p_values.loc[corr_matrix.columns == x.name, corr_matrix.index == x.name].iloc[0]))

    for i, row in corr_matrix.iterrows():
        for j, val in row.iteritems():
            if p_values.loc[i,j] > 0.05:
                empty_df.loc[i,j] = '-'
            else:
                empty_df.loc[i,j] = round(corr_matrix.loc[i,j],2)

    fig,ax = plt.subplots(figsize=(20, 20))
    fig.subplots_adjust(top=0.95, bottom=0.25, left=0.25)
    p1 = sns.heatmap(corr_matrix, annot=empty_df, cmap='Greys',
                     xticklabels=column_names, yticklabels=column_names, annot_kws={"fontsize": 20}, fmt='')


    cbar = p1.collections[0].colorbar
    # Set the font size of the color bar labels
    cbar.ax.tick_params(labelsize=20)
    #
    p1.set_xticklabels(p1.get_xticklabels(), fontsize=25)
    p1.tick_params(axis='y', labelsize=25)
    # # plt.xticks(rotation=25)
    # # plt.yticks(rotation=0)
    fig_save_dir = './fig/'
    # plt.show()
    # plt.savefig(fig_save_dir+'heatmap_metric.pdf')



correlation['OER'] = OER
correlation['OER_ow'] = OER_ow

correlation['LCS mean'] = []
# correlation['LCS variance'] = []
correlation['LCS min'] = []

correlation['Levenshieten mean'] = []
# correlation['Levenshieten variance'] = []
correlation['Levenshieten max'] = []

correlation['United_Diff mean'] = []
# correlation['United_Diff variance'] = []
correlation['United_Diff min'] = []

correlation['Tree_Diff mean'] = []
# correlation['Tree_Diff variance'] = []
correlation['Tree_Diff min'] = []

for case in LCS:
    correlation['LCS mean'].append(np.mean(case))
    # correlation['LCS variance'].append(np.var(case))
    correlation['LCS min'].append(min(case))

for case in Levenshieten:
    correlation['Levenshieten mean'].append(np.mean(case))
    # correlation['Levenshieten variance'].append(np.var(case))
    correlation['Levenshieten max'].append(max(case))

for case in United_Diff:
    correlation['United_Diff mean'].append(np.mean([i[0] for i in case]))
    # correlation['United_Diff variance'].append(np.var([i[0] for i in case]))
    correlation['United_Diff min'].append(min([i[0] for i in case]))

for case in Tree_Diff:
    correlation['Tree_Diff mean'].append(np.mean([i[0] for i in case]))
    # correlation['Tree_Diff variance'].append(np.var([i[0] for i in case]))
    correlation['Tree_Diff min'].append(min([i[0] for i in case]))

print('LCS min ', np.mean(correlation['LCS min']))
print('Levenshieten max', np.mean(correlation['Levenshieten max']))
print('United_Diff min', np.mean(correlation['United_Diff min']))
print('Tree_Diff min', np.mean(correlation['Tree_Diff min']))


correlation_rank = []
# for i in range(1, len(list(correlation.keys()))):
#     for j in range(i + 1, len(list(correlation.keys()))):
#         correlation_rank.append([
#             list(correlation.keys())[i],
#             list(correlation.keys())[j],
#             stats.pearsonr(correlation[list(correlation.keys())[i]], correlation[list(correlation.keys())[j]])
#         ])
# sorted_list = sorted(correlation_rank, key=lambda x: x[2][1])
high_relavent = []
problem_features = ['description length', 'difficulty', 'time_limit', 'cf_rating']
for case in correlation_rank:
    if (case[0] in problem_features or case[1] in problem_features) and case[2][1] < 0.05:
        high_relavent.append(case)
        # print('%s & %s\'s correlation: %s' % (list(correlation.keys())[i],
        #                                       list(correlation.keys())[j],
        #                                       stats.pearsonr(correlation[list(correlation.keys())[i]], correlation[list(correlation.keys())[j]])
        #                                       )
        #       )
correlation_list = []
# test pass rate
correlation_list.append(correlation['test pass rate mean'])
correlation_list.append(correlation['test pass rate variance'])
correlation_list.append(correlation['test pass rate max diff'])
# output equivalence rate
correlation_list.append(correlation['OER'])
correlation_list.append(correlation['OER_ow'])
# LCS
correlation_list.append(correlation['LCS mean'])
# correlation_list.append(correlation['LCS variance'])
correlation_list.append(correlation['LCS min'])
# Levenshieten
correlation_list.append(correlation['Levenshieten mean'])
# correlation_list.append(correlation['Levenshieten variance'])
correlation_list.append(correlation['Levenshieten max'])
# United_Diff
correlation_list.append(correlation['United_Diff mean'])
# correlation_list.append(correlation['United_Diff variance'])
correlation_list.append(correlation['United_Diff min'])
# Tree_Diff
correlation_list.append(correlation['Tree_Diff mean'])
# correlation_list.append(correlation['Tree_Diff variance'])
correlation_list.append(correlation['Tree_Diff min'])
# problem features
correlation_list.append(correlation['description length'])
if dataset == 'code_contest':
    correlation_list.append(correlation['difficulty'])
    correlation_list.append(correlation['time_limit'])
    correlation_list.append(correlation['cf_rating'])
if dataset == 'code_contest':
    column_names = ['TPR mean value',
                    'TPR mean variance',
                    'TPR mean max diff',

                    'OER mean',
                    'OER (no ex.) mean',

                    'LCS mean',
                    'LCS worst',

                    'LED mean',
                    'LED worst',

                    'United_Diff mean',
                    'United_Diff worst',

                    'Tree_Diff mean',
                    'Tree_Diff worst',

                    'description length',
                    'difficulty',
                    'time_limit',
                    'cf_rating'
                    ]
else:
    column_names = ['TPR mean value',
                    'TPR mean variance',
                    'TPR mean max diff',

                    'OER mean',
                    'OER_ow mean',

                    'LCS mean',
                    'LCS worst',

                    'LED mean',
                    'LED worst',

                    'United_Diff mean',
                    'United_Diff worst',

                    'Tree_Diff mean',
                    'Tree_Diff worst',

                    'description length'
                    ]
# correlation_list = [[row[i] for row in correlation_list] for i in range(len(correlation_list[0]))]

# df = pd.DataFrame(correlation_list, columns=column_names)

# corr_matrix = df.corr()
# p_values = df.corr(method=lambda x, y: stats.pearsonr(x, y)[1])
p_values = []
correlation_values = []
empty_values = []
for i in range(len(column_names)):
    p_tmp = []
    c_tmp = []
    e_tmp = []
    for j in range(len(column_names)):
        p_tmp.append(stats.pearsonr(correlation_list[i], correlation_list[j])[1])
        c_tmp.append(stats.pearsonr(correlation_list[i], correlation_list[j])[0])
        e_tmp.append(0)
    p_values.append(p_tmp)
    correlation_values.append(c_tmp)
    empty_values.append(e_tmp)

for i in range(len(column_names)):
    for j in range(len(column_names)):
        if p_values[i][j] > 0.05:
            empty_values[i][j]  = '-'
        else:
            empty_values[i][j] = round(correlation_values[i][j], 2)

# empty_df = pd.DataFrame(index=corr_matrix.index, columns=corr_matrix.columns)

# Use the apply() function to replace the values in the heatmap
# annotated_data = corr_matrix.applymap(lambda x: replace_value(x, p_values.loc[corr_matrix.columns == x.name, corr_matrix.index == x.name].iloc[0]))

# for i, row in corr_matrix.iterrows():
#     for j, val in row.iteritems():
#         # if p_values.loc[i, j] > 0.05:
#         if p_values[column_names.index(i)][column_names.index(j)] > 0.05:
#             empty_df.loc[i, j] = '-'
#         else:
#             empty_df.loc[i, j] = round(corr_matrix.loc[i, j], 2)

fig, ax = plt.subplots(figsize=(20, 20))
fig.subplots_adjust(top=0.98, bottom=0.18, left=0.18)
p1 = sns.heatmap(correlation_values, annot=empty_values, cmap='Greys',
                 xticklabels=column_names, yticklabels=column_names, annot_kws={"fontsize": 18}, fmt='')

cbar = p1.collections[0].colorbar
# Set the font size of the color bar labels
cbar.ax.tick_params(labelsize=20)
#
p1.set_xticklabels(p1.get_xticklabels(), fontsize=25)
p1.tick_params(axis='y', labelsize=25)
# # plt.xticks(rotation=25)
# # plt.yticks(rotation=0)
fig_save_dir = './fig/'
# plt.show()
plt.savefig(fig_save_dir+'heatmap_metric.pdf')