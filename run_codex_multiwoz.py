import os
import json
import argparse
import copy
from collections import defaultdict
from tqdm import tqdm
import re
import numpy as np
from config_multiwoz import CONFIG
from sklearn.metrics import f1_score
from evaluate_metrics import BLEUScorer
from codex_completion_multiwoz import codex_completion
from prompting_multiwoz import get_prompt_multiwoz, get_prompt_w_rule
from utils.helper import SpeedLimitTimer
# from retriever.code.embed_based_retriever import EmbeddingRetriever
# from evaluate_metrics import evaluate
from datareaders_node_answer import filter_dataset, NextActionUtteranceDataset
# input arguments
parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str)
parser.add_argument("--task_train", type=str, choices=["domain-transfer", "task-transfer"])
parser.add_argument("--num_tasks", type=int, default=1)
parser.add_argument("--schema_path", type=str)
parser.add_argument("--use_schema", action="store_true")
parser.add_argument("--w_explanation", action="store_true")
parser.add_argument("--num_examples", type=int, default=0)
parser.add_argument('--output_dir', type=str, default="./expts", help="directory to save running log and configs")
args = parser.parse_args()

# create the output folder
os.makedirs(args.output_dir, exist_ok=True)

with open(os.path.join(args.output_dir, "exp_config.json"), 'w') as f:
    json.dump(vars(args), f, indent=4)


def run(args, exp_setting=None):

    # Data readers
    test_dataset = json.load(open(args.data_path))

    timer = SpeedLimitTimer(second_per_step=3.1)  # openai limitation 20 queries/min

    # result_dict = defaultdict(list)  # use to record the accuracy

    # start experiment
    all_result = []
    output_tests = []
    n_total = 0
    for data_item in tqdm(test_dataset):
        # print("test data_item:\n", data_item)
        n_total += 1

        completion = ""
        if args.use_schema:
            # print("demo ex:\n", CONFIG["demo_example"])
            prompt_text = get_prompt_multiwoz(
                args.schema_path, data_item, CONFIG["demo_example"], exp_setting, args.num_examples)
        
        # record the prompt
        data_item['prompt'] = prompt_text
        # print("prompt:\n", prompt_text)
        total_completion_list = []
        # codex completion
        complete_flag = False
        try_count = 0
        while not complete_flag:
            try:
                completion = codex_completion(prompt_text)
                # print("completion:\n", completion) # (1) [user] hello. [sep] action label: 0:hello
                if "action" and "system :" in completion:
                    total_completion_list.append(completion)
                    print("completion:\n", completion)
                    try_count += 1
                    # complete_flag = True

            except Exception as e:
                if e.user_message.startswith("This model's maximum context length"):
                    # print("prompt overlength")
                    prompt_text = get_prompt_multiwoz(
                        args.schema_path, data_item, CONFIG["demo_example"], exp_setting, 1)
                else:
                    # throughput too high
                    timer.sleep(10)
            if try_count >1:
                complete_flag = True

        # limit query speed
        timer.step()

        data_item['generated_belief_response'] = total_completion_list
        all_result.append(data_item)
        output_tests.append(total_completion_list)

    return all_result, output_tests


if __name__ == "__main__":


    domains = ['ride', 'trip', 'plane', 'spaceship', 'meeting', 'weather', 'party', 'doctor', 'trivia', 'apartment', 'restaurant', 'hotel', 'bank']
    tasks = ['restaurant']

    scores = []
    # Use old scores if experiment crashes.
    old_scores = []
    orig_output_dir = args.output_dir

    # # ZERO-SHOT TASK TRANSFER EXPERIMENTS
    if args.task_train == "task-transfer":
        tasks = tasks[:args.num_tasks]     
        for i,task in enumerate(tasks):
            print("TASK", task)
            exp_setting = {"task": task, "data_type": "happy"}

            args.output_dir = orig_output_dir + "/" + task + "/"

            all_results = run(args, exp_setting)[0]
            os.makedirs(args.output_dir, exist_ok=True)
            with open(os.path.join(args.output_dir, "running_log.json"), 'w') as f:
                json.dump(all_results, f, indent=4)

            output_tests = run(args, exp_setting)[1]
            json.dump(output_tests, open(os.path.join(args.output_dir, "generated_responses.json"),'w'), indent=2)

    # # # ZERO-SHOT DOMAIN TRANSFER EXPERIMENTS
    # if args.task_train == "domain-transfer":
    #     domains = domains[:args.num_tasks]
    #     for i,task in enumerate(domains):
    #         print("DOMAIN", task)
    #         exp_setting = {"domain": task, "data_type": "happy"}
    #         args.output_dir = orig_output_dir + "/" + task + "/"

    #         all_results = run(args, exp_setting)[0]
    #         os.makedirs(args.output_dir, exist_ok=True)
    #         with open(os.path.join(args.output_dir, "running_log.json"), 'w') as f:
    #             json.dump(all_results, f, indent=4)
            
    #         output_tests = run(args, exp_setting)[1]
    #         json.dump(output_tests, open(os.path.join(args.output_dir, "generated_responses.json"),'w'), indent=2)
    #         # with open(os.path.join(args.output_dir, "evaluation_file.txt"), "w") as writer:
    #         #     writer.write(report)
