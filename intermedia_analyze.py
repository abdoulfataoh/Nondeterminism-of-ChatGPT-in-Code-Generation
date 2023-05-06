#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 15/03/2023 12:08
# @Author  : Shuyin Ouyang
# @File    : stability_analyze.py.py
import argparse
import json
import os
import re
import subprocess

import nltk
import pycode_similar
from nltk.translate.bleu_score import sentence_bleu

def response_2_code(response):
    code_template = re.compile('```.*\n([\s\S]+?)\n```', re.M)
    code = code_template.findall(response)
    if len(code) > 0:
        return code[-1]
    else:
        return ''

def solution_evaluation(solution, test_cases, demo_file, time_limit):
    passed_case = []
    case_status = []
    with open(demo_file, 'w') as f:
        f.write(solution)
    for i in range(len(test_cases)):
        try:
            # TODO: timeout value survey
            output = subprocess.run(["python", demo_file], capture_output=True, text=True,
                                    input=test_cases[i]['input'], timeout=time_limit)
        except subprocess.TimeoutExpired as e:
            print(e, flush=True)
            case_status.append('timeout')
            continue
        except Exception as e:
            print(e, flush=True)
            case_status.append('exception')
            continue
        if output.returncode != 0:
            case_status.append('execution error: %s' % output.returncode)
        else:
            case_status.append(output.stdout.strip())
        if test_cases[i]['output'].strip() == output.stdout.strip():
            passed_case.append(i)

    pass_num = len(passed_case)
    print('%s/%s pass.' % (pass_num, len(test_cases)), flush=True)
    return passed_case, case_status

def solution_evaluation_HumanEval(solution, test_cases, demo_file, call_demo_file, entry_point, time_limit):
    passed_case = []
    case_status = []
    with open(demo_file, 'w') as f:
        f.write(solution)
    for i in range(len(test_cases)):
        if test_cases[i]['relation'] == '==':
            with open(call_demo_file, 'w') as f:
                f.write('from %s import %s\nprint(%s(%s))' % (
                    demo_file.split('.')[0],
                    entry_point,
                    entry_point,
                    test_cases[i]['input']
                ))
            try:
                output = subprocess.run(["python", call_demo_file], capture_output=True, text=True, timeout=time_limit)

            except subprocess.TimeoutExpired as e:
                print(e, flush=True)
                case_status.append('Timeout')
                continue
            except Exception as e:
                print(e, flush=True)
                case_status.append('Exception')
                continue
            if output.returncode != 0:
                case_status.append('execution error: %s' % output.returncode)
            else:
                case_status.append(output.stdout.strip())
            if test_cases[i]['output'].strip() == output.stdout.strip():
                passed_case.append(i)
        else:
            if '$input$' in test_cases[i]['relation'] or '$demo$' in test_cases[i]['relation']:
                with open(call_demo_file, 'w') as f:
                    f.write('from %s import %s\n%s' % (
                        demo_file.split('.')[0],
                        entry_point,
                        test_cases[i]['relation'].replace('$input$', str(test_cases[i]['input'])).replace('$demo$', demo_file.split('.')[0])
                    ))
            else:
                with open(call_demo_file, 'w') as f:
                    f.write('from %s import %s\nprint(%s)' % (demo_file.split('.')[0],
                        entry_point,
                        test_cases[i]['relation'].replace('candidate', entry_point)))
                try:
                    output = subprocess.run(["python", call_demo_file], capture_output=True, text=True, timeout=time_limit)

                except subprocess.TimeoutExpired as e:
                    print(e, flush=True)
                    case_status.append('Timeout')
                    continue
                except Exception as e:
                    print(e, flush=True)
                    case_status.append('Exception')
                    continue
                if output.returncode != 0:
                    case_status.append('execution error: %s' % output.returncode)
                else:
                    case_status.append(output.stdout.strip())
                if output.stdout.strip() == 'True':
                    passed_case.append(i)

    pass_num = len(passed_case)
    print('%s/%s pass.' % (pass_num, len(test_cases)), flush=True)
    return passed_case, case_status

def semantic_similarity(problem_dic, name, code_candidates):
    # semantic similarity
    BlEU_score_between_5_list = []
    for i in range(len(problem_dic[name]['code_candidates'])):
        # BLEU score among 5 code candidates
        # the first one is important
        reference_code = code_candidates[:i] + code_candidates[i + 1:]
        BlEU_score_between_5 = sentence_bleu(reference_code, code_candidates[i])
        BlEU_score_between_5_list.append(BlEU_score_between_5)
        # print(BlEU_score_between_5)
    problem_dic[name]['semantic_similarity'] = {
        'BLEU_score_between_5_list': BlEU_score_between_5_list
    }

def syntatic_similarity(problem_dic, name, code_candidates, case_status_list):
    same_output_between_5 = []
    same_output_between_5_correct = []
    same_output_between_5_timeout = []
    same_output_between_5_exception = []
    same_output_between_5_execution_error = []
    Levenshtein_edit_distance = []
    for i in range(len(case_status_list[0])):
        output_set = set()
        for j in range(len(problem_dic[name]['code_candidates'])):
            output_set.add(case_status_list[j][i])
        # print(output_set)
        if len(output_set) == 1:
            same_output_between_5.append(i)
            if list(output_set)[0] == 'timeout':
                same_output_between_5_timeout.append(i)
            elif 'execution error' in list(output_set)[0]:
                same_output_between_5_execution_error.append(i)
            elif list(output_set)[0] == 'exception':
                same_output_between_5_exception.append(i)
            else:
                same_output_between_5_correct.append(i)

    for i in range(len(problem_dic[name]['code_candidates'])):
        if i == 0:
            continue
        Levenshtein_edit_distance.append(nltk.edit_distance(code_candidates[0], code_candidates[i]))
    problem_dic[name]['syntatic_similarity'] = {
        'same_output_between_5': same_output_between_5,
        'same_output_between_5_correct': same_output_between_5_correct,
        'same_output_between_5_timeout': same_output_between_5_timeout,
        'same_output_between_5_exception': same_output_between_5_exception,
        'same_output_between_5_execution_error': same_output_between_5_execution_error,
        'Levenshtein_edit_distance': Levenshtein_edit_distance
    }

def summarize(func_ast_diff_list):
    sum_total_count = sum(func_diff_info.total_count for func_diff_info in func_ast_diff_list)
    sum_plagiarism_count = sum(func_diff_info.plagiarism_count for func_diff_info in func_ast_diff_list)
    if sum_total_count == 0:
        sum_plagiarism_percent = 0
    else:
        sum_plagiarism_percent = sum_plagiarism_count / float(sum_total_count)
    return sum_plagiarism_percent, sum_plagiarism_count, sum_total_count

def modify_code(code):
    prefix = 'def main():\n'
    tab = '    '
    sufix = '\n\nif __name__ == "__main__":\n' + tab + 'main()'
    code_list = code.split('\n')
    code_list = [tab + i for i in code_list]
    res_0 = '\n'.join(i for i in code_list)
    res = prefix + res_0 + sufix
    return res

def structual_similarity(problem_dic, name, code_reference):
    # code_reference = [referenced_code_str, candidate_code_str1, candidate_code_str2, ...]
    if code_reference[0] == '':
        problem_dic[name]['structual_similarity'] = {
            'structual_similarity_UnifiedDiff': [-1, -1, -1, -1],
            'structual_similarity_TreeDiff': [-1, -1, -1, -1]
        }
        return
    try:
        results_UnifiedDiff = pycode_similar.detect(code_reference,
                                    diff_method=pycode_similar.UnifiedDiff, keep_prints=True, module_level=False)
    except pycode_similar.NoFuncException:
        # code_reference[0] = modify_code(code_reference[0])
        for i in range(5):
            if "__main__" not in code_reference[i]:
                code_reference[i] = modify_code(code_reference[i])
        try:
            results_UnifiedDiff = pycode_similar.detect(code_reference,
                                                        diff_method=pycode_similar.UnifiedDiff, keep_prints=True,
                                                        module_level=False)
        except Exception as e:
            problem_dic[name]['structual_similarity'] = {
                'structual_similarity_UnifiedDiff': [-2, -2, -2, -2],
                'structual_similarity_TreeDiff': [-2, -2, -2, -2]
            }
            return
    except Exception as e:
        problem_dic[name]['structual_similarity'] = {
            'structual_similarity_UnifiedDiff': [-2, -2, -2, -2],
            'structual_similarity_TreeDiff': [-2, -2, -2, -2]
        }
        return
    try:
        results_TreeDiff = pycode_similar.detect(code_reference,
                                                 diff_method=pycode_similar.TreeDiff, keep_prints=True,
                                                 module_level=False)
    except pycode_similar.NoFuncException:
        # code_reference[0] = modify_code(code_reference[0])
        for i in range(5):
            if "__main__" not in code_reference[i]:
                code_reference[i] = modify_code(code_reference[i])
        try:
            results_TreeDiff = pycode_similar.detect(code_reference,
                                                     diff_method=pycode_similar.TreeDiff, keep_prints=True,
                                                     module_level=False)
        except Exception as e:
            problem_dic[name]['structual_similarity'] = {
                'structual_similarity_UnifiedDiff': [-3, -3, -3, -3],
                'structual_similarity_TreeDiff': [-3, -3, -3, -3]
            }
            return

    except Exception as e:
        problem_dic[name]['structual_similarity'] = {
            'structual_similarity_UnifiedDiff': [-3, -3, -3, -3],
            'structual_similarity_TreeDiff': [-3, -3, -3, -3]
        }
        return
    structual_similarity_UnifiedDiff = []
    structual_similarity_TreeDiff = []
    for index, func_ast_diff_list in results_UnifiedDiff:
        sum_similarity_percent, sum_similarity_count, sum_total_count = summarize(func_ast_diff_list)
        structual_similarity_UnifiedDiff.append([sum_similarity_percent, sum_similarity_count, sum_total_count])

    for index, func_ast_diff_list in results_TreeDiff:
        sum_similarity_percent, sum_similarity_count, sum_total_count = summarize(func_ast_diff_list)
        structual_similarity_TreeDiff.append([sum_similarity_percent, sum_similarity_count, sum_total_count])
    problem_dic[name]['structual_similarity'] = {
        'structual_similarity_UnifiedDiff': structual_similarity_UnifiedDiff,
        'structual_similarity_TreeDiff': structual_similarity_TreeDiff
    }

def analyze_process(log_file):
    demo_file = 'demo.py'
    problem_dic = {}
    count = 0
    while os.path.exists(demo_file):
        demo_file = 'demo_%s.py' % count
        count += 1
    # initialize the record file
    names = []
    if not os.path.exists('./log/record/%s' % (log_file.split('/')[1])):
        with open('./log/record/%s' % (log_file.split('/')[1]), 'w') as f:
            f.write('')
    else:
        with open('./log/record/%s' % (log_file.split('/')[1]), 'r') as f:
            for line in f.readlines():
                content = json.loads(line)
                names.append(content['name'])

    with open('./tmp2/code_contests_test.json', 'r') as f:
    # with open('dataset/code_contests_test.json', 'r') as f:
        problem_list = json.load(f)
        for i in range(len(problem_list)):
            if not problem_list[i]['name'] in names:
                pattern = re.compile(r'(?<=seconds:=)*\d+')
                time_limit = pattern.findall(problem_list[i]['time_limit'].split('\n')[0])[0]
                problem_dic[problem_list[i]['name']] = {
                    'name': problem_list[i]['name'],
                    'index_num': i,
                    'time_limit': int(time_limit)
                }

    with open(log_file, 'r') as f:
        for line in f.readlines():
            content = json.loads(line)
            name = content['name']
            if name in names:
                continue
            index = content['index']
            response = content['response']
            if index == 0:
                print('----------------------problem name: %s--------------------------------' % (name),
                      flush=True)
            # initialize
            if 'code_candidates' not in problem_dic[name]:
                problem_dic[name]['response_candidates'] = []
                problem_dic[name]['code_candidates'] = []
            print('generate code from response', flush=True)
            # load from code_contest dataset
            problem = problem_list[problem_dic[name]['index_num']]
            test_set = problem['public_tests'] + problem['private_tests'] + problem['generated_tests']
            reference_code = []
            for candidate in problem['solutions']:
                if candidate['language'] == 3:
                    reference_code.append(candidate['solution'].split())

            # get code from response
            code = response_2_code(response)
            # default weight: weights=(0.25, 0.25, 0.25, 0.25)
            if reference_code == []:
                BLEU_score_correct = -1
            else:
                BLEU_score_correct = sentence_bleu(reference_code, code.split())

            # use code to run test cases
            time_limit = problem_dic[name]['time_limit']
            test_case_solved = solution_evaluation(code, test_set, demo_file, time_limit)
            problem_dic[name]['response_candidates'].append(response)
            res = {
                'code': code,
                'index': index,
                'passed_case': test_case_solved[0],
                'case_status': test_case_solved[1],
                'BlEU_score_correct': BLEU_score_correct
            }
            problem_dic[name]['code_candidates'].append(res)

            if index == 4:
                print('%s stability analyze' % (name), flush=True)
                code_candidates = []
                code_reference = []
                case_status_list = []
                for code_res in problem_dic[name]['code_candidates']:
                    code_candidates.append(code_res['code'].split())
                    code_reference.append(code_res['code'])
                    case_status_list.append(code_res['case_status'])
                # semantic similarity
                semantic_similarity(problem_dic, name, code_candidates)
                # syntatic similarity
                # whether 5 output is same
                # syntatic_similarity(problem_dic, name, code_candidates, case_status_list)
                # structural_similarity
                # structual_similarity(problem_dic, name, code_reference)
                print('writing in %s' % (name), flush=True)
                # write in
                json_str = json.dumps(problem_dic[name])
                with open('./log/record/%s' % (log_file.split('/')[1]), 'a') as f:
                    f.write(json_str + '\n')
                problem_dic.pop(name)

def analyze_process_HumanEval(log_file):
    demo_file = 'demo.py'
    call_demo_file = 'call_demo.py'
    count = 0
    problem_dic = {}
    while os.path.exists(demo_file) or os.path.exists(call_demo_file):
        demo_file = 'demo_%s.py' % count
        call_demo_file = 'call_demo_%s.py' % count
        count += 1
    names = []
    if not os.path.exists('./log/record/%s' % (log_file.split('/')[1])):
        with open('./log/record/%s' % (log_file.split('/')[1]), 'w') as f:
            f.write('')
    else:
        with open('./log/record/%s' % (log_file.split('/')[1]), 'r') as f:
            for line in f.readlines():
                content = json.loads(line)
                names.append(content['name'])
    problem_list = []
    with open('HumanEval/HumanEval_new.jsonl', 'r') as f:
        for line in f.readlines():
            problem_list.append(json.loads(line))

    for i in range(len(problem_list)):
        if not problem_list[i]['name'] in names:
            problem_dic[problem_list[i]['name']] = {
                'name': problem_list[i]['name'],
                'index_num': i,
                'time_limit': int(3) # by default
            }
    with open(log_file, 'r') as f:
        for line in f.readlines():
            content = json.loads(line)
            name = content['name']
            if name in names:
                continue
            index = content['index']
            response = content['response']
            if index == 0:
                print('----------------------problem name: %s--------------------------------' % (name),
                      flush=True)
            # initialize
            if 'code_candidates' not in problem_dic[name]:
                problem_dic[name]['response_candidates'] = []
                problem_dic[name]['code_candidates'] = []
            print('generate code from response', flush=True)
            # load from code_contest dataset
            problem = problem_list[problem_dic[name]['index_num']]
            test_set = problem['test_case']
            reference_code = []
            reference_code.append(problem['solution'])

            # get code from response
            code = response_2_code(response)
            # default weight: weights=(0.25, 0.25, 0.25, 0.25)
            if reference_code == []:
                BLEU_score_correct = -1
            else:
                BLEU_score_correct = sentence_bleu(reference_code, code.split())

            # use code to run test cases
            time_limit = problem_dic[name]['time_limit']
            test_case_solved = solution_evaluation_HumanEval(code, test_set, demo_file, call_demo_file, problem['entry_point'], time_limit)
            problem_dic[name]['response_candidates'].append(response)
            res = {
                'code': code,
                'index': index,
                'passed_case': test_case_solved[0],
                'case_status': test_case_solved[1],
                'BlEU_score_correct': BLEU_score_correct
            }
            problem_dic[name]['code_candidates'].append(res)
            if index == 4:
                print('%s stability analyze' % (name), flush=True)
                code_candidates = []
                code_reference = []
                case_status_list = []
                for code_res in problem_dic[name]['code_candidates']:
                    code_candidates.append(code_res['code'].split())
                    code_reference.append(code_res['code'])
                    case_status_list.append(code_res['case_status'])
                # semantic similarity
                semantic_similarity(problem_dic, name, code_candidates)
                # syntatic similarity
                # whether 5 output is same
                # syntatic_similarity(problem_dic, name, code_candidates, case_status_list)
                # structural_similarity
                # structual_similarity(problem_dic, name, code_reference)
                # return problem_dic
                print('writing in %s' % (name), flush=True)
                # write in
                json_str = json.dumps(problem_dic[name])
                with open('./log/record/%s' % (log_file.split('/')[1]), 'a') as f:
                    f.write(json_str + '\n')
                problem_dic.pop(name)

def analyze_process_APPS(log_file):
    path = './APPS/test/'
    demo_file = 'demo.py'
    problem_dic = {}
    count = 0
    while os.path.exists(demo_file):
        demo_file = 'demo_%s.py' % count
        count += 1
    # initialize the record file
    names = []
    if not os.path.exists('./log/record/%s' % (log_file.split('/')[1])):
        with open('./log/record/%s' % (log_file.split('/')[1]), 'w') as f:
            f.write('')
    else:
        with open('./log/record/%s' % (log_file.split('/')[1]), 'r') as f:
            for line in f.readlines():
                content = json.loads(line)
                names.append(content['name'])

    for dirpath, dirnames, filenames in os.walk(path):
        # iterating for every problem
        for i in range(len(dirnames[:500])):
            if dirnames[:500][i] not in names:
                problem_dic[dirnames[:500][i]] = {
                    'name': dirnames[:500][i],
                    'index_num': i,
                    'time_limit': int(3)
                }
    with open(log_file, 'r') as f:
        for line in f.readlines():
            content = json.loads(line)
            name = content['name']
            if name in names:
                continue
            index = content['index']
            response = content['response']
            if index == 0:
                print('----------------------problem name: %s--------------------------------' % (name),
                      flush=True)
            # initialize
            if 'code_candidates' not in problem_dic[name]:
                problem_dic[name]['response_candidates'] = []
                problem_dic[name]['code_candidates'] = []
            print('generate code from response', flush=True)
            # load from code_contest dataset

            with open(path + name + '/input_output.json', 'r', encoding='utf-8') as f:
                test_case = json.load(f)
            test_set = []
            for i in range(len(test_case['inputs'])):
                test_set.append({
                    'input': test_case['inputs'][i],
                    'output': test_case['outputs'][i]
                })
            reference_code = []
            if os.path.exists(path + name + '/solutions.json',):
                with open(path + name + '/solutions.json', 'r', encoding='utf-8') as f:
                    solutions = json.load(f)
            else:
                solutions = []
            for candidate in solutions:
                reference_code.append(candidate.split())
            code = response_2_code(response)
            if reference_code == []:
                BLEU_score_correct = -1
            else:
                BLEU_score_correct = sentence_bleu(reference_code, code.split())
            time_limit = problem_dic[name]['time_limit']
            test_case_solved = solution_evaluation(code, test_set, demo_file, time_limit)
            problem_dic[name]['response_candidates'].append(response)
            res = {
                'code': code,
                'index': index,
                'passed_case': test_case_solved[0],
                'case_status': test_case_solved[1],
                'BlEU_score_correct': BLEU_score_correct
            }
            problem_dic[name]['code_candidates'].append(res)
            if index == 4:
                print('%s stability analyze' % (name), flush=True)
                code_candidates = []
                code_reference = []
                case_status_list = []
                for code_res in problem_dic[name]['code_candidates']:
                    code_candidates.append(code_res['code'].split())
                    code_reference.append(code_res['code'])
                    case_status_list.append(code_res['case_status'])
                # semantic similarity
                semantic_similarity(problem_dic, name, code_candidates)
                # syntatic similarity
                # whether 5 output is same
                # syntatic_similarity(problem_dic, name, code_candidates, case_status_list)
                # structural_similarity
                # structual_similarity(problem_dic, name, code_reference)
                # return problem_dic
                print('writing in %s' % (name), flush=True)
                # write in
                json_str = json.dumps(problem_dic[name])
                with open('./log/record/%s' % (log_file.split('/')[1]), 'a') as f:
                    f.write(json_str + '\n')
                problem_dic.pop(name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-f",
        "--file",
        type=str,
        help="Choose file",
        required=True,
    )
    args = parser.parse_args()
    if 'code_contest' in args.file:
        analyze_process(args.file)
    elif 'HumanEval' in args.file:
        analyze_process_HumanEval(args.file)
    elif 'APPS' in args.file:
        analyze_process_APPS(args.file)
    # problem_dic = analyze_process_APPS('log/dataset_APPS_model_gpt-3.5-turbo_topn_5_temperature_1.0.log_0')
