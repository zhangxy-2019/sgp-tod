
import copy
import json
import random

table_prompt = """
CREATE TABLE hotel(
  name text,
  pricerange text CHECK (pricerange IN (dontcare, cheap, moderate, expensive)),
  type text CHECK (type IN (hotel, guest house)),
  parking text CHECK (parking IN (dontcare, yes, no)),
  book_stay int,
  book_day text,
  book_people int,
  area text CHECK (area IN (dontcare, centre, east, north, south, west)),
  stars int CHECK (stars IN (dontcare, 0, 1, 2, 3, 4, 5)),
  internet text CHECK (internet IN (dontcare, yes, no))
)
/*
4 example rows:
SELECT * FROM hotel LIMIT 4;
name  pricerange  type  parking book_stay book_day  book_people area  stars internet
a and b guest house moderate  guest house  dontcare  3 friday  5 east  4 yes
ashley hotel  expensive hotel yes 2 thursday  5 north 5 yes
el shaddia guest house  cheap guest house  yes 5 friday  2 centre  dontcare  no
express by holiday inn cambridge  dontcare  guest house yes 3 monday  2 east  dontcare  no
*/

CREATE TABLE train(
  destination text,
  departure text,
  day text,
  book_people int,
  leaveat text,
  arriveby text
)
/*
3 example rows:
SELECT * FROM train LIMIT 3;
destination departure day book_people leaveat arriveby
london kings cross  cambridge monday  6 dontcare 05:51
cambridge stansted airport  dontcare  1 20:24 20:52
peterborough  cambridge saturday  2  12:06  12:56
*/

CREATE TABLE attraction(
  name text,
  area text CHECK (area IN (dontcare, centre, east, north, south, west)),
  type text CHECK (type IN (architecture, boat, church, cinema, college, concert hall, entertainment, hotspot, multiple sports, museum, nightclub, park, special, swimming pool, theatre))
)
/*
4 example rows:
SELECT * FROM attraction LIMIT 4;
name area type
abbey pool and astroturf pitch  centre  swimming pool
adc theatre centre  theatre
all saints church dontcare  architecture
castle galleries  centre  museum
*/

CREATE TABLE restaurant(
  name text,
  food text,
  pricerange text CHECK (pricerange IN (dontcare, cheap, moderate, expensive)),
  area text CHECK (area IN (centre, east, north, south, west)),
  book_time text,
  book_day text,
  book_people int
)
/*
5 example rows:
SELECT * FROM restaurant LIMIT 5;
name  food  pricerange  area  book_time book_day  book_people
pizza hut city centre italian dontcare centre  13:30 wednesday 7
the missing sock  international moderate  east  dontcare dontcare  2
golden wok chinese moderate north 17:11 friday 4
cambridge chop house  dontcare  expensive  center 08:43 monday  5
darrys cookhouse and wine shop  modern european expensive center  11:20 saturday  8
*/

CREATE TABLE taxi(
  destination text,
  departure text,
  leaveat text,
  arriveby text
)
/*
3 example rows:
SELECT * FROM taxi LIMIT 3;
destination departure leaveat arriveby
copper kettle royal spice 14:45 15:30
magdalene college  university arms hotel dontcare  15:45
lovell lodge  da vinci pizzeria 11:45 dontcare
*/

-- Using valid SQLite, answer the following multi-turn conversational questions for the tables provided above.

"""
# end2end_prompt = "Generate appropriate wizard actions and responses according to the demonstrated examples and task-specific rules:"

def conversion(prompt, reverse=False):
    conversion_dict = {"leaveat": "depart_time", "arriveby": "arrive_by_time",
                       "book_stay": "book_number_of_days",
                       "food": "food_type"}
    reverse_conversion_dict = {v: k for k, v in conversion_dict.items()}
    used_dict = reverse_conversion_dict if reverse else conversion_dict

    for k, v in used_dict.items():
        prompt = prompt.replace(k, v)
    return prompt

def get_prompt_multiwoz(instruction_path, data_item, example_path, exp_setting=None, n_examples=0, in_domain=False):

    total_instructions = json.load(open(instruction_path))
    total_demonstration_examples = json.load(open(example_path))
    current_instruct = total_instructions
    end2end_prompt = "Following the rules, generate appropriate system response based on the history."
    demo_examples = []
    demo_text = ""
    demo_text += end2end_prompt
    user_system_graph_list = []
    current_graph = ""
    for i_en, node in enumerate(list(current_instruct["graph"])):
        if node in current_instruct["replies"]:
            node_graph = current_instruct["replies"][node]
        # elif node in current_instruct["beliefs"]:
        #     node_graph = current_instruct["beliefs"][node]
        current_graph += "(" + str(int(i_en+1))+ ")"+ node_graph + "action: " + current_instruct["replies"][current_instruct["graph"][node]] + "[eos]"

    if n_examples > 0:
        # for demo in total_demonstration_examples:
        #     demo_examples.append(demo)
        demo_examples = total_demonstration_examples
        # demo_examples = random.sample(demo_examples, n_examples)
        # print("demo_examples")
        for i, demo_ex in enumerate(demo_examples):
            ex_text = "example #" + str(int(i+1)) + " task: " + demo_ex["task"] + " " + demo_ex["belief_instructions"] + " task rules: "  + demo_ex["new_rule"] + " history: " + " ".join(demo_ex["history"]) + " " + demo_ex["belief"] + " " + demo_ex["explanation_1"] + " " + demo_ex["reply"] + "[eos]"
            demo_text += " " + ex_text
        demo_text += " test: " + "task : " + current_instruct["task"] + " " + current_instruct["belief_instructions"] + " task rules: " + current_graph + " history: " + " ".join(data_item["history"]) + " SQL: select * from train where"
    else:
        demo_text += "task instructions: " + current_instruct["task"] + " " + current_instruct["general_instructions"] + " rules: " + current_graph + " history: " + " ".join(data_item["history"][-1:])

    return demo_text

def get_prompt(data_item, example_path, exp_setting=None, n_examples=0, in_domain=False, w_explanation=False):
    # print("example_path:\n", example_path)
    total_demonstration_examples = json.load(open(example_path))
    end2end_prompt = "Generate appropriate wizard actions based on the history, following the most relevant task rule:"
    demo_examples = []
    demo_text = ""
    demo_text += end2end_prompt
    if n_examples > 0:
        if "domain" in exp_setting:
            for demo in total_demonstration_examples:
                if exp_setting["domain"] not in list(demo["domains"]):
                    demo_examples.append(demo)
        elif "task" in exp_setting:
            for demo in total_demonstration_examples:
                if exp_setting["task"] not in list(demo["tasks"]):
                    demo_examples.append(demo)
        else:
            if in_domain:
                for demo in total_demonstration_examples:
                    if data_item['tasks'][0] in list(demo["tasks"]):
                        demo_examples.append(demo)
                        
        demo_examples = random.sample(demo_examples, n_examples)
        # print("demo_examples")
        for i, demo_ex in enumerate(demo_examples):
            ex_text = "example #" + str(int(i+1)) + " task: " + demo_ex["tasks"] + " task rules: " + demo_ex["task_rules"] + demo_ex["dialog_examples"] + " answer: " + demo_ex["explanation"]
            demo_text += " " + ex_text
        demo_text += " test: " + data_item["input_text"]
    else:
        demo_text += data_item["input_text"]

    return demo_text

def get_prompt_before(data_item, example_path, exp_setting=None, n_examples=0, in_domain=False, w_explanation=False):
    # print("example_path:\n", example_path)
    total_demonstration_examples = json.load(open(example_path))
    end2end_prompt = "Generate appropriate wizard actions according to the demonstrated examples, task-specific rules and the history:"
    demo_examples = []
    demo_text = ""
    demo_text += end2end_prompt
    if n_examples > 0:
        if "domain" in exp_setting:
            for demo in total_demonstration_examples:
                if exp_setting["domain"] not in demo["domains"]:
                    demo_dialog = random.sample(demo["dialog_examples"], 1)
                    # if not w_explanation:
                    #     demo_dialog = [i.split("[wizard]")[0] for i in demo_dialog]
                    demo["demo_examples"] = demo_dialog
                    demo_examples.append(demo)
        elif "task" in exp_setting:
            for demo in total_demonstration_examples:
                if exp_setting["task"] not in demo["tasks"]:
                    demo_dialog = random.sample(demo["dialog_examples"], 1)
                    # if not w_explanation:
                    #     demo_dialog = [i.split("[wizard]")[0] for i in demo_dialog]
                    demo["demo_examples"] = demo_dialog
                    demo_examples.append(demo)
        else:
            if in_domain:
                for demo in total_demonstration_examples:
                    if data_item['tasks'][0] in demo["tasks"]:
                        demo_dialog = random.sample(demo["dialog_examples"], 1)
                        # if not w_explanation:
                        #     demo_dialog = [i.split("[wizard]")[0] for i in demo_dialog]
                        demo["demo_examples"] = demo_dialog
                        demo_examples.append(demo)
                        
        demo_examples = random.sample(demo_examples, n_examples)
        # print("demo_examples")
        for i, demo_ex in enumerate(demo_examples):
            ex_text = "example #" + str(int(i+1)) + " task: " + demo_ex["tasks"] + " task-specific rules: " + demo_ex["dialog_examples"][0]
            demo_text += ex_text
        demo_text += "test: " + data_item["input_text"]
    else:
        demo_text += data_item["input_text"]

    return demo_text



