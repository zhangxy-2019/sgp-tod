import os
import json
import argparse
import copy
from collections import defaultdict
from tqdm import tqdm
import re
import numpy as np
from config import CONFIG_e2e
from sklearn.metrics import f1_score
# from evaluate_metrics import BLEUScorer
from codex_completion import codex_completion, codex_completion_gpt3, codex_completion_chatgpt
from prompting_debug import get_prompt
from utils.helper import SpeedLimitTimer
# from retriever.code.embed_based_retriever import EmbeddingRetriever
# from evaluate_metrics import evaluate
from datareaders_ic import filter_dataset, NextActionUtteranceDataset
# input arguments
parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str)
parser.add_argument("--task_train", type=str, choices=["domain-transfer", "task-transfer"])
parser.add_argument("--num_tasks", type=int, default=1)
parser.add_argument("--schema_path", type=str)
parser.add_argument("--use_schema", action="store_true")
parser.add_argument("--w_explanation", action="store_true")
parser.add_argument("--num_examples", type=int, default=2)
parser.add_argument('--output_dir', type=str, default="./expts", help="directory to save running log and configs")
parser.add_argument("--model_type", default="codex", type=str, help="The model architecture to be fine-tuned.")
args = parser.parse_args()

# create the output folder
os.makedirs(args.output_dir, exist_ok=True)

with open(os.path.join(args.output_dir, "exp_config.json"), 'w') as f:
    json.dump(vars(args), f, indent=4)

if args.model_type == "codex":
    codex_completion = codex_completion
elif args.model_type == "gpt3":
    codex_completion = codex_completion_gpt3
else:
    codex_completion = codex_completion_chatgpt

CONFIG = CONFIG_e2e

def run(args, exp_setting=None):

    # Data readers
    dataset_initializer = NextActionUtteranceDataset
    
    dataset = dataset_initializer(args.data_path, args.schema_path)

    if exp_setting is not None:
        if "domain" in exp_setting:
            data_type = exp_setting.get("data_type")
            train_dataset = filter_dataset(dataset,
                                            data_type=data_type,
                                            percentage=1.0,
                                            domain=exp_setting.get("domain"),
                                            exclude=True,
                                            train=True)

            test_dataset = filter_dataset(dataset,
                                            data_type=data_type,
                                            percentage=1.0,
                                            domain=exp_setting.get("domain"),
                                            exclude=False,
                                            train=False)
        elif "task" in exp_setting:
            data_type = exp_setting.get("data_type")
            train_dataset = filter_dataset(dataset,
                                            data_type=data_type,
                                            percentage=1.0,
                                            task=exp_setting.get("task"),
                                            exclude=True,
                                            train=True)

            test_dataset = filter_dataset(dataset,
                                            data_type=data_type,
                                            percentage=1.0,
                                            task=exp_setting.get("task"),
                                            exclude=False,
                                            train=False)
        else:
            data_type = exp_setting.get("data_type") # standard happy
            train_dataset = filter_dataset(dataset,
                                            data_type=data_type,
                                            percentage=0.8,
                                            train=True)

            test_dataset = filter_dataset(dataset,
                                            data_type=data_type,
                                            percentage=0.2,
                                            train=False)

    timer = SpeedLimitTimer(second_per_step=3.1)  # openai limitation 20 queries/min

    # result_dict = defaultdict(list)  # use to record the accuracy

    # start experiment
    all_result = []
    pred_action = []
    pred_wizard = []
    true_action = []
    true_wizard = []
    input_history = []
    n_total = 0
    for data_item in tqdm(test_dataset):
        # print("test data_item:\n", data_item)
        n_total += 1

        completion = ""
        if args.use_schema:
            # print("demo ex:\n", CONFIG["demo_example"])
            prompt_text = get_prompt(
                data_item, CONFIG["demo_example"], exp_setting, args.num_examples)

        # record the prompt
        data_item['prompt'] = prompt_text
        print("prompt:\n", prompt_text)

        # codex completion
        complete_flag = False
        error_count = 0
        while not complete_flag:
            try:
                completion = codex_completion(prompt_text)
                print("completion:\n", completion) # (1) [user] hello. [sep] action label: 0:hello 
                pre = completion.split("action:")[-1].strip()
                pre = pre.split("[wizard]")[0].strip()
                # print("pre", pre)
                if str(pre) in data_item['task_action_list']:
                    complete_flag = True
                else:
                    error_count += 1
                # print("completion:\n", completion)
            except Exception as e:
                if e.user_message.startswith("This model's maximum context length"):
                    # print("prompt overlength")
                    prompt_text = get_prompt(
                        data_item, CONFIG["demo_example"], exp_setting, 1)
                else:
                    # throughput too high
                    timer.sleep(10)
            if error_count >=5:
                complete_flag = True
                completion = completion + "1000:none"
        # limit query speed
        timer.step()

        # record the predictions
        data_item['completion'] = completion
        all_result.append(data_item)

        # print the result
        # print("completion: ", completion) # 0:hello [wizard] hello, how can i help? 
        if "1000" in completion:
            if ")" in completion:
                completion_num = completion.split(")")[0].split("(")[-1]
                completion_num = int(completion_num) -1
                if "task_action_label_list" in data_item.keys():
                    value_action = data_item["task_action_label_list"][completion_num]
            else:
                value_action = "1000"
        else:
            value_action = completion.split("action:")[-1].strip()
            value_action = value_action.split("[wizard]")[0].strip()
        value_action = value_action.split(":")[0]
        pred_action += [int(value_action)]
        true_action += [int(data_item["action"])]
        input_history += [data_item['orig_history']]

    assert len(pred_action) == len(true_action) == len(input_history)
    # assert len(pred_wizard) == len(true_wizard)
    # perform evaluation
    # action
    acc = sum(p == t for p,t in zip(pred_action, true_action))/len(pred_action)
    f1 = f1_score(true_action, pred_action, average='weighted')
    print("accuracy: ", acc, "\n")
    print("f1 score: ", f1, "\n")

    # bleu 
    # bscorer = BLEUScorer()
    # bleu_score = bscorer.score(pred_wizard, true_wizard)
    # print("corpus BLEU: ", bleu_score, "\n")
    # start analysis
    id_map = test_dataset.action_label_to_id
    label_map = sorted(id_map, key=id_map.get)
    print("INCORRECT ==========================================================")
    for i in range(len(true_action)):
        if pred_action[i] != true_action[i]:
            if pred_action[i] < 1000:
                print("dialog history: ", input_history[i] + "\n", "true label: ", label_map[true_action[i]], "predicted label: ", label_map[pred_action[i]])
            else:
                print("dialog history: ", input_history[i] + "\n", "true label: ", label_map[true_action[i]], "predicted label: ", str(pred_action[i]))
    print("CORRECT ==========================================================")
    for i in range(len(true_action)):
        if pred_action[i] == true_action[i]:
            print("dialog history: ", input_history[i] + "\n", "true label: ", label_map[true_action[i]], "predicted label: ", label_map[pred_action[i]])

    report = "***** Eval results *****\n"
    report += "corpus accuracy = {:2.2f}".format(acc) + "\n"
    report += "corpus f1 score = {:2.2f}".format(f1) + "\n"
    # report += "corpus bleu = {:2.2f}".format(bleu_score)
    
    return f1, acc, all_result, report


if __name__ == "__main__":


    domains = ['ride', 'trip', 'plane', 'spaceship', 'meeting', 'weather', 'party', 'doctor', 'trivia', 'apartment', 'restaurant', 'hotel', 'bank']
    tasks = ['hotel_service_request', 'bank_balance', 'weather', 'bank_fraud_report', 'party_rsvp', 'apartment_search', 'trivia', 'ride_book', 'apartment_schedule', 'hotel_book', 'ride_status', 'restaurant_search', 'doctor_schedule', 'doctor_followup', 'restaurant_book', 'plane_search', 'meeting_schedule',  'party_plan', 'plane_book', 'spaceship_access_codes', 'hotel_search', 'trip_directions']
    training_examples = [2]
    # scores = []
    # # Use old scores if experiment crashes.
    old_scores = []
    orig_dir = args.output_dir
    orig_output_dir = args.output_dir

    # # ZERO-SHOT TASK TRANSFER EXPERIMENTS
    if args.task_train == "task-transfer":
        for j, num_trainex in enumerate(training_examples):
            print("number of training examples: ", str(num_trainex))
            scores = []
            old_scores = []
            for i,task in enumerate(tasks):
                print("TASK", task)
                exp_setting = {"task": task, "data_type": "happy"}

                args.action_output_dir = orig_dir + "/" + str(num_trainex) + "/" + task + "/"
                args.output_dir = orig_output_dir + "/" + str(num_trainex) + "/" + task + "/"

                if i < len(old_scores):
                    scores.append(old_scores[i])
                else:
                    score = run(args, exp_setting)[:2]
                    scores.append(score)
                print(scores)

                print("f1", np.mean([e[0] for e in scores]))
                print("acc", np.mean([e[1] for e in scores]))

                all_results = run(args, exp_setting)[2]
                os.makedirs(args.output_dir, exist_ok=True)
                with open(os.path.join(args.output_dir, "running_log.json"), 'w') as f:
                    json.dump(all_results, f, indent=4)

            report = ""
            report += "number of training examples: " + str(num_trainex) + "\n"
            report += "task: " + task + "\n"
            report += "scores: " + str(scores) + "\n"
            report += "f1 " + str(np.mean([e[0] for e in scores])) + "\n"
            report += "acc " + str(np.mean([e[1] for e in scores]))
            out_dir = orig_output_dir + str(num_trainex) + "/"
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
            with open(os.path.join(out_dir, "evaluation_file.txt"), "w") as writer:
                writer.write(report)

    # # ZERO-SHOT DOMAIN TRANSFER EXPERIMENTS
    if args.task_train == "domain-transfer":
        for j, num_trainex in enumerate(training_examples):
            print("number of training examples: ", str(num_trainex))
            scores = []
            old_scores = []
            for i,task in enumerate(domains):
                print("DOMAIN", task)
                exp_setting = {"domain": task, "data_type": "happy"}

                args.action_output_dir = orig_dir + "/" + str(num_trainex) + "/" + task + "/"
                args.output_dir = orig_output_dir + "/" + str(num_trainex) + "/" + task + "/"

                if i < len(old_scores):
                    scores.append(old_scores[i])
                else:
                    score = run(args, exp_setting)[:2]
                    scores.append(score)
                print(scores)

                print("f1", np.mean([e[0] for e in scores]))
                print("acc", np.mean([e[1] for e in scores]))

                all_results = run(args, exp_setting)[2]
                os.makedirs(args.output_dir, exist_ok=True)
                with open(os.path.join(args.output_dir, "running_log.json"), 'w') as f:
                    json.dump(all_results, f, indent=4)

            report = ""
            report += "number of training examples: " + str(num_trainex) + "\n"
            report += "domain: " + task + "\n"
            report += "scores: " + str(scores) + "\n"
            report += "f1 " + str(np.mean([e[0] for e in scores])) + "\n"
            report += "acc " + str(np.mean([e[1] for e in scores]))
            out_dir = orig_output_dir + str(num_trainex) + "/"
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
            with open(os.path.join(out_dir, "evaluation_file.txt"), "w") as writer:
                writer.write(report)
