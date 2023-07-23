import math
import time

import nltk
import openai
import re
import os
import json
import subprocess
import argparse
from nltk.corpus import stopwords

openai.api_key = ''

def description_2_code(description, model, topn, temperature):
    prompt = 'Generate Python3 code (Markdown):\n'
    completion = openai.ChatCompletion.create(
        model=model,
        n=topn,
        temperature=temperature,
        messages=[{"role": "user",
                   "content": prompt + description},
                  ]
    )
    response_list = []
    # code_list = []
    for i in completion['choices']:
        response_list.append(i['message']['content'])
    # code_template = re.compile('```.*\n([\s\S]+?)\n```', re.M)
    # for response in response_list:
    #     code = code_template.findall(response)
    #     if len(code) > 0:
    #         code_list.append(code[-1])
    #     else:
    #         code_list.append('')
    # return code_list, response_list
    return response_list

def code_contest_experiment(dataset, option, model, sequence, topn=1, temperature=1.0):
    if option == 'original':
        log_file = './log/dataset_%s_model_%s_topn_%s_temperature_%s.log_%s' % \
                   (dataset, model, topn, temperature, sequence)
    else:
        log_file = './log/%s_dataset_%s_model_%s_topn_%s_temperature_%s.log_%s' % \
                   (option, dataset, model, topn, temperature, sequence)

    with open('./tmp2/code_contests_test.json', 'r') as f:
    # with open('./dataset/code_contests_test.json', 'r') as f:
        problem_list = json.load(f)
    names = set()
    if os.path.exists(log_file):
        with open(log_file, 'r') as f:
            for line in f:
                content = json.loads(line)
                names.add(content['name'])

    for problem in problem_list:
        if problem['name'] in names:
            continue
        print('----------------------problem name: %s--------------------------------' % (problem['name']), flush=True)
        print('using %s to generate response' % (model), flush=True)
        description = problem['description']
        try:
            response_list = description_2_code(description, model, topn, temperature)
        except Exception as e:
            print('%s---------%s' % (problem['name'], e), flush=True)
            continue
        for i in range(len(response_list)):
            res = {
                'name': problem['name'],
                'index': i,
                'response': response_list[i],
            }
            print('response %s is writting into file' % (i), flush=True)
            json_str = json.dumps(res)
            with open(log_file, 'a') as f:
                f.write(json_str+'\n')
        print('%s finish!' % (problem['name']), flush=True)
    print('Done!', flush=True)

def APPS_experiment(dataset, option, model, sequence, topn=1, temperature=1.0):
    path = './APPS/test/'

    if option == 'original':
        log_file = './log/dataset_%s_model_%s_topn_%s_temperature_%s.log_%s' % \
                   (dataset, model, topn, temperature, sequence)
    else:
        log_file = './log/%s_dataset_%s_model_%s_topn_%s_temperature_%s.log_%s' % \
                   (option, dataset, model, topn, temperature, sequence)

    names = set()
    if os.path.exists(log_file):
        with open(log_file, 'r') as f:
            for line in f:
                content = json.loads(line)
                names.add(content['name'])

    # better, worse description
    for dirpath, dirnames, filenames in os.walk(path):
        # iterating for every problem
        for dirname in dirnames[:500]:
            if dirname in names:
                continue
            print('----------------------problem name: %s--------------------------------' % (dirname),
                  flush=True)
            print('using %s to generate code' % (model), flush=True)
            # description
            with open(path + dirname + '/question.txt', 'r', encoding='utf-8') as f:
                description = f.read()
            try:
                response_list = description_2_code(description, model, topn, temperature)

            except Exception as e:
                print('%s---------%s' % (dirname, e), flush=True)
                # response_list = ['', '', '', '', '']
            for i in range(len(response_list)):
                res = {
                    'name': dirname,
                    'index': i,
                    'response': response_list[i],
                }
                print('response %s is writting into file' % (i), flush=True)
                json_str = json.dumps(res)
                with open(log_file, 'a') as f:
                    f.write(json_str + '\n')
            print('%s finish!' % (dirname), flush=True)
    print('Done!', flush=True)

def HumanEval_experiment(dataset, option, model, sequence, topn=1, temperature=1.0):
    if option == 'original':
        log_file = './log/dataset_%s_model_%s_topn_%s_temperature_%s.log_%s' % \
                   (dataset, model, topn, temperature, sequence)
    else:
        log_file = './log/%s_dataset_%s_model_%s_topn_%s_temperature_%s.log_%s' % \
                   (option, dataset, model, topn, temperature, sequence)
    problem_list = []
    with open('./HumanEval/HumanEval.jsonl', 'r') as f:
        for line in f.readlines():
            problem_list.append(json.loads(line))
    names = set()
    if os.path.exists(log_file):
        with open(log_file, 'r') as f:
            for line in f:
                content = json.loads(line)
                names.add(content['name'])

    for problem in problem_list:
        if problem['task_id'] in names:
            continue
        print('----------------------problem name: %s--------------------------------' % (problem['task_id']), flush=True)
        print('using %s to generate response' % (model), flush=True)
        description = problem['prompt']
        try:
            response_list = description_2_code(description, model, topn, temperature)
        except Exception as e:
            print('%s---------%s' % (problem['task_id'], e), flush=True)
            continue
        for i in range(len(response_list)):
            res = {
                'name': problem['task_id'],
                'index': i,
                'response': response_list[i],
            }
            print('response %s is writting into file' % (i), flush=True)
            json_str = json.dumps(res)
            with open(log_file, 'a') as f:
                f.write(json_str + '\n')
        print('%s finish!' % (problem['task_id']), flush=True)
    print('Done!', flush=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        choices=['APPS', 'code_contest', 'HumanEval'],
        help="Choose dataset",
        required=True,
    )
    parser.add_argument(
        "-m",
        "--model",
        help="Openai Model",
        type=str,
        required=True,
    )
    parser.add_argument(
        "-n",
        "--topn",
        type=int,
        help="Top N candidates",
        required=True,
    )
    parser.add_argument(
        "-t",
        "--temperature",
        type=float,
        help="Set the temperature",
        required=True,
    )
    # parser.add_argument(
    #     "-o",
    #     "--option",
    #     type=str,
    #     choices=['original', 'rephrase', 'extractive_summarize', 'abstractive_summarize'],
    #     help="Choose the mode of the experiment",
    #     required=True,
    #     default='original'
    # )
    parser.add_argument(
        "-s",
        "--sequence",
        type=str,
        help="Choose the order of the experiment",
        default='0'
    )
    args = parser.parse_args()
    if args.dataset == 'code_contest':
        code_contest_experiment(args.dataset, 'original', args.model, args.sequence, args.topn, args.temperature)
    elif args.dataset == 'APPS':
        APPS_experiment(args.dataset, 'original', args.model, args.sequence, args.topn, args.temperature)
    elif args.dataset == 'HumanEval':
        HumanEval_experiment(args.dataset, 'original', args.model, args.sequence, args.topn, args.temperature)
