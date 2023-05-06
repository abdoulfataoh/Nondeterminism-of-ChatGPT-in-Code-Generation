#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 08/03/2023 09:46
# @Author  : Shuyin Ouyang
# @File    : generate_response.py

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


def solution_evaluation(solution, test_cases, demo_file):
    passed_case = []
    with open(demo_file, 'w') as f:
        f.write(solution)
    for i in range(len(test_cases)):
        try:
            output = subprocess.run(["python", demo_file], capture_output=True, text=True, input=test_cases[i]['input'], timeout=11)
        except Exception as e:
            print(e)
            continue
        if test_cases[i]['output'].strip() == output.stdout.strip():
            passed_case.append(i)
    pass_num = len(passed_case)
    print('%s/%s pass.' % (pass_num, len(test_cases)), flush=True)
    return (pass_num, len(test_cases), passed_case)

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

def description_2_code_modify(description, model, topn, temperature):
    prompt = 'Generate Python3 code (Markdown):\n'
    completion = openai.ChatCompletion.create(
        model=model,
        n=topn,
        temperature=temperature,
        max_tokens=3000,
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


def codex_description_2_code(description, model, topn, temperature):
    # prompt = 'Generate Python3 code only (Markdown):\n'
    prompt = 'Return the correct Python3 code for the following description (Markdown):\n'
    completion = openai.ChatCompletion.create(
        model=model,
        n=topn,
        temperature=temperature,
        messages=[{"role": "user",
                   "content": prompt + description},
                  ]
    )
    response_list = []
    code_list = []
    for i in completion['choices']:
        response_list.append(i['message']['content'])
    code_template = re.compile('```.*\n([\s\S]+?)\n```', re.M)
    for response in response_list:
        code = code_template.findall(response)
        if len(code) > 0:
            code_list.append(code[-1])
        else:
            code_list.append('')
    return code_list, response_list

def description_rephrase(model, description):
    # prompt = 'Rephrase the following description, and make it better to generate AC code:\n'
    prompt = 'Rephrase the following description:\n'
    completion = openai.ChatCompletion.create(
        model=model,
        temperature=0,
        messages=[{"role": "user",
                   "content": prompt + description},
                  ]
    )
    rephrased_description = completion['choices'][0]['message']['content']
    return rephrased_description

def summerization_extractive(description, percentage):
    word2count = {}

    if '\ninput' in description.lower():
        index = description.lower().index('\ninput')
    else:
        index = -1
    problem_description = description[:index]
    problem_else = description[index + 1:]
    word_list = nltk.word_tokenize(problem_description)
    stopword_list = stopwords.words('english')
    for i in word_list:
        if i not in stopword_list:
            if i not in word2count:
                word2count[i] = 1
            else:
                word2count[i] += 1
    for key in word2count.keys():
        word2count[key] = word2count[key] / max(word2count.values())
    # print(word2count)
    # calculate the score of the sentence
    sent2score = {}
    sentence_list = nltk.sent_tokenize(problem_description)
    for sentence in sentence_list:
        for word in nltk.word_tokenize(sentence):
            if word not in stopword_list:
                if sentence not in sent2score.keys():
                    sent2score[sentence] = word2count[word]
                else:
                    sent2score[sentence] += word2count[word]

    # sort the sentence by score
    # keep the original order of the sentence
    sorted_dic = sorted([(k, v) for k, v in sent2score.items()], reverse=True, key=lambda x: x[1])[:math.ceil(len(sentence_list)*percentage)]
    sorted_dic = [i[0] for i in sorted_dic]
    # print(sorted_dic)
    final_result = []
    for sentence in sentence_list:
        if sentence in sorted_dic:
            final_result.append(sentence)
    return ' '.join(i for i in final_result) + "/n" + problem_else

def summerization_abstractive(description):
    if '\ninput' in description.lower():
        index = description.lower().index('\ninput')
    else:
        index = -1
    problem_description = description[:index]
    problem_else = description[index+1:]
    summarizer = pipeline("summarization", model="philschmid/bart-large-cnn-samsum")
    return summarizer(problem_description)[0]['summary_text'] + problem_else

def experiment(dataset, option, model, sequence, topn=1, temperature=1.0):
    openai.api_key = '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'
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

def APPS_experiment(dataset, option, model, sequence,  topn=1, temperature=1.0):
    path = './APPS/test/'
    openai.api_key = ''
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
        for dirname in dirnames:
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
                response_list = ['', '', '', '', '']
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

def HumanEval_experiment(dataset, option, model, sequence,  topn=1, temperature=1.0):
    openai.api_key = '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'
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
    parser.add_argument(
        "-o",
        "--option",
        type=str,
        choices=['original', 'rephrase', 'extractive_summarize', 'abstractive_summarize'],
        help="Choose the mode of the experiment",
        required=True,
        default='original'
    )
    parser.add_argument(
        "-s",
        "--sequence",
        type=str,
        help="Choose the order of the experiment",
        default='0'
    )
    args = parser.parse_args()
    if args.option == 'abstractive_summarize':
        from transformers import pipeline
    if args.dataset == 'code_contest':
        experiment(args.dataset, args.option, args.model, args.sequence, args.topn, args.temperature)
    elif args.dataset == 'APPS':
        APPS_experiment(args.dataset, args.option, args.model, args.sequence, args.topn, args.temperature)
    elif args.dataset == 'HumanEval':
        HumanEval_experiment(args.dataset, args.option, args.model, args.sequence, args.topn, args.temperature)
