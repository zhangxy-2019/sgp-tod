
import copy
import json
import random


def get_prompt(data_item, example_path, exp_setting=None, n_examples=2, in_domain=False, w_explanation=False):
    # print("example_path:\n", example_path)
    total_demonstration_examples = json.load(open(example_path))
    end2end_prompt = "Generate appropriate wizard actions based on the history, following the most relevant task rule."
    # current_belief_rule = "belief instruction: wizard needs to collect the hotel name, user's name, room number, service request and service time sequentially before querying: hotelname = shadyside inn, old town inn, hilton hotel, etc.; roomnumber = 450, 483, etc. ; time = 5 pm, 8 pm, etc. ; customername = ben, alexis, mark, etc. ; customerrequest = towels, cheeseburger and fries with 2 brewskis, etc."
    demo_examples = []
    demo_text = ""
    demo_text += end2end_prompt
    if n_examples > 0:
        # for demo in total_demonstration_examples:
        #     if "tasks" == "ride_book" or "doctor_schedule":
        #         demo_examples.append(demo)

        if "domain" in exp_setting:
            for demo in total_demonstration_examples:
                if exp_setting["domain"] != demo["domains"]:
                    demo_examples.append(demo)
        elif "task" in exp_setting:
            for demo in total_demonstration_examples:
                if exp_setting["task"] != demo["tasks"]:
                    demo_examples.append(demo)

        demo_examples = random.sample(demo_examples, n_examples)
        # print("demo_examples")
        # demo_examples = total_demonstration_examples
        for i, demo_ex in enumerate(demo_examples):
            # ex_text = "example #" + str(int(i+1)) + " task: " + demo_ex["tasks"] + " " + demo_ex["instructions"] + " task rules: " + demo_ex["task_rules"] + demo_ex["dialog_examples"] + " " + demo_ex["extra_ex"] + " " + demo_ex["explanation"]
            ex_text = "example #" + str(int(i+1)) + " task: " + demo_ex["tasks"] + " task rules: " + demo_ex["task_rules"] + " " + demo_ex["dialog_examples"] + " answer: " + demo_ex["explanation"]
            demo_text += " " + ex_text
        demo_text += " test: " + data_item["input_text"]
        # demo_text += " test: " + data_item["input_text"].split("task rules")[0] + current_belief_rule + " task rules" + data_item["input_text"].split("task rules")[-1] + "select * from hotel_service_request"
    else:
        demo_text += data_item["input_text"]
    demo_text = demo_text.lower()

    return demo_text

