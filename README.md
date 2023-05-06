# Stability-of-ChatGPT-in-Code-Generation

## Datasets
Below are the links of datasets we used in our experiments.

[HumanEval](https://github.com/openai/human-eval) We use script [Modify_HumanEval.py](https://github.com/CodeHero0/Stability-of-ChatGPT-in-Code-Generation/blob/main/Modify_HumanEval.py) to reshape the HumanEval dataset.

[APPS](https://github.com/hendrycks/apps)

[CodeContests](https://github.com/deepmind/code_contests) For those who may not know how to conver riegeli files into JSON files, here is the [script](https://github.com/deepmind/code_contests/pull/21)

## Full results

'./heatmaps' stores all the heatmaps of three datasets.

## Conduct Experiments
Our experiments mainly contains three steps:

### 1. Generating response

We use generate_response.py to generate response with three code problem datasets, and store it into JSON files in '.\log\'.

### 2. Intermedia result processing

We use intermedia_analyze.py to generate intermedia based on the test cases and responses we get in the first step, intermedia result will be stored into '.\log\record\' 

We then use Structural similarity.py and Syntactic Similarity & OER.py to generate the final results we need in semantic similarity, syntactic similarity, and structural similarity.

### 3. Result display

We use draw_plot.py to display our results in the answer of RQ1.

We use draw_heatmap.py to display our result in the answer of RQ4.
