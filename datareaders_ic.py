import copy
import json
import numpy as np
import os
import pickle
import torch
import random
import copy
from collections import defaultdict
from torch.utils.data import Dataset
from tqdm import tqdm

def filter_dataset(dataset,
                   data_type="happy", # happy, unhappy, multitask
                   domain=None,
                   task=None,
                   exclude=False,
                   percentage=1.0,
                   train=True):
    """
    Split the dataset according to the criteria

    - data_type:
        - happy: Only the happy dialogs
        - unhappy: Only the happy + unhappy dialogs (no multitask)
        - multitask: All the dialogs

    - domain:
        - Requirements:
            - task should be None
            - data_type should be happy/unhappy
        - If exclude = True
            - Exclude dialog of this domain
        - If exclude = False
            - Include ONLY dialogs of this domain

    - task:
        - Requirements:
            - domain should be None
            - data_type should be happy/unhappy
        - If exclude = True
            - Exclude dialog of this domain
        - If exclude = False
            - Include ONLY dialogs of this domain

    - percentage:
        - Take only a certain percentage of the available data (after filters)
        - If train = True
            - Take the first [percentage]% of the data
        - If train = False:
            - Take the last [percentage]% of the data
    """
    examples = dataset.examples

    # Filter based on happy/unhappy/multitask
    if data_type == "happy":
        examples = [ex for ex in examples if ex.get("happy")]
    elif data_type == "unhappy":
        examples = [ex for ex in examples if not ex.get("multitask")]

    # Filter based on domain
    if domain is not None:
        assert data_type in ["happy", "unhappy"], "For zero-shot experiments, can only use single-task data"
        assert task is None, "Can filter by domain OR task, not both"

        if exclude:
            examples = [ex for ex in examples if ex["domains"][0] != domain]
        else:
            examples = [ex for ex in examples if ex["domains"][0] == domain]

    # Filter based on task
    if task is not None:
        assert data_type in ["happy", "unhappy"], "For zero-shot experiments, can only use single-task data"
        assert domain is None, "Can filter by domain OR task, not both"

        if exclude:
            examples = [ex for ex in examples if ex["tasks"][0] != task]
        else:
            examples = [ex for ex in examples if ex["tasks"][0] == task]

    # Split based on percentage
    all_dialog_ids = sorted(list(set([ex['dialog_id'] for ex in examples])))
    if train:
        selected_ids = all_dialog_ids[:int(len(all_dialog_ids)*percentage)]
    else:
        selected_ids = all_dialog_ids[-int(len(all_dialog_ids)*percentage):]

    selected_ids = set(selected_ids)
    examples = [ex for ex in examples if ex['dialog_id'] in  selected_ids]

    # Filter out only the relevant keys for each example (so that DataLoader doesn't complain)
    # keys = ["input_ids", "attention_mask", "token_type_ids", "action", "tasks", "history", "response"]
    keys = ["input_text", "output_text", "action", "tasks", "task_action_list", 'orig_history']
    examples = [{k:v for k,v in ex.items() if k in keys} for ex in examples]
    for ex in examples:
        ex["tasks"] = ex["tasks"][0]

    # Return new dataset
    new_dataset = copy.deepcopy(dataset)
    new_dataset.examples = examples
    return new_dataset

class NextActionSchema(Dataset):
    def __init__(self,
                 data_path,
                 tokenizer,
                 max_seq_length,
                 action_label_to_id,
                 vocab_file_name):
        # Check if cached pickle file exists
        data_dirname = os.path.dirname(os.path.abspath(data_path))
        # print("data_dirname_schema: \n", data_dirname)
        cached_path = os.path.join(data_dirname, "schema_action_cached")
        if os.path.exists(cached_path):
            with open(cached_path, "rb") as f:
                self.examples, self.action_to_response = pickle.load(f)

            return None

        # Read all of the JSON files in the data directory
        tasks = [
            json.load(open(data_path + fn + "/" + fn + ".json")) for fn in os.listdir(data_path)
        ]
        # print("task", task)
        self.action_to_response = {}

        # Iterate over the schemas and get (1) the prior states and (2) the 
        # next actions.
        node_to_utt = {}
        self.examples = []
        for task in tqdm(tasks):
            # Get the graph
            graph = task['graph']

            # For every edge in the graph, get examples of transfer to each action
            for prev,action in graph.items():
                if False and prev in task['r_graph']:
                    sys_utt = '[wizard] ' + task['replies'][task['r_graph'][prev]] + ' [SEP] '
                    utterance = '[user] ' + sys_utt + task['replies'][prev] + ' [SEP]'
                else:
                    utterance = '[user] ' + task['replies'][prev] + ' [SEP]'

                node_to_utt[prev] = utterance 

                # For next action prediction, we can normalize the diff query types
                #if action in ['query_check', 'query_book']:
                #    action = 'query'

                if action not in action_label_to_id:
                    continue

                action_label = action_label_to_id[action]
                self.action_to_response[action_label] = task['replies'][action] 
                encoded = tokenizer.encode(utterance)
                self.examples.append({
                    "input_ids": np.array(encoded.ids)[-max_seq_length:],
                    "attention_mask": np.array(encoded.attention_mask)[-max_seq_length:],
                    "token_type_ids": np.array(encoded.type_ids)[-max_seq_length:],
                    "action": action_label,
                    "task": task['task'], # "apartment_schedule"
                    "node_utterance": utterance
                })

        with open(cached_path, "wb+") as f:
            pickle.dump([self.examples, self.action_to_response], f)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]

class NextActionDataset(Dataset):
    def __init__(self,
                 data_path,
                 tokenizer,
                 max_seq_length,
                 vocab_file_name):
        # Check if cached pickle file exists
        data_dirname = os.path.dirname(os.path.abspath(data_path))
        cached_path = os.path.join(data_dirname, "action_cached")
        if os.path.exists(cached_path):
            with open(cached_path, "rb") as f:
                self.action_label_to_id, self.examples = pickle.load(f)

            return None

        # Read all of the JSON files in the data directory
        conversations = [
            json.load(open(data_path + fn)) for fn in os.listdir(data_path)
        ]

        # Iterate over the conversations and get (1) the dialogs and (2) the 
        # actions for all wizard turns.
        self.examples = []
        self.action_label_to_id = {}
        for conv in tqdm(conversations):
            # History (so far) for this dialog
            # history_list = []
            history = ""
            for i,utt in enumerate(conv['Events']):
                # NOTE: Ground truth action labels only exist when wizard picks suggestion. 
                # We skip all custom utterances for action prediction.
                if utt['Agent'] == 'Wizard' and utt['Action'] in ['query', 'pick_suggestion']:
                    # Tokenize history
                    processed_history = ' '.join(history.strip().split()[:-1])
                    # print("processed history: \n", processed_history)
                    encoded_history  = tokenizer.encode(processed_history)

                    # Convert action label to id
                    query_label = 'query'
                    if 'ActionLabel' not in utt:
                        query_check = 'Check' in [e['RequestType'][1:-1] for e in utt['Constraints'] if 'RequestType' in e]
                        query_book = 'Book' in [e['RequestType'][1:-1] for e in utt['Constraints'] if 'RequestType' in e]
                        # In case of a bug, if both book and check are on - we treat it as a check.
                        if query_check:
                            query_label = 'query_check' 
                        elif query_book:
                            query_label = 'query_book' 

                    action_label = utt['ActionLabel'] if 'ActionLabel' in utt else query_label #  "spaceship_inform_outcome"
                    if action_label not in self.action_label_to_id:
                        self.action_label_to_id[action_label] = len(self.action_label_to_id)
                    action_label_id = self.action_label_to_id[action_label]

                    # Include metadata 
                    domains = conv['Scenario']['Domains'] #  [
                                                        # "spaceship"
                                                        # ],
                    tasks = [e['Task'] for e in conv['Scenario']['WizardCapabilities']] #  ["spaceship_access_codes"]
                    happy = conv['Scenario']['Happy'] #  true,
                    multitask = conv['Scenario']['MultiTask'] #  false,

                    # Add to data
                    self.examples.append({
                        "input_ids": np.array(encoded_history.ids)[-max_seq_length:],
                        "attention_mask": np.array(encoded_history.attention_mask)[-max_seq_length:],
                        "token_type_ids": np.array(encoded_history.type_ids)[-max_seq_length:],
                        "action": action_label_id,
                        "dialog_id": conv['DialogueID'],
                        "domains": domains,
                        "tasks": tasks,
                        "happy": happy,
                        "multitask": multitask,
                        "orig_history": processed_history,
                        "orig_action": action_label,
                    })

                # Process and concatenate to history
                if utt['Agent'] in ['User', 'Wizard', 'KnowledgeBase']:
                    utt_text = ""

                    # If there's text, just use it directly
                    if utt['Action'] in ['pick_suggestion', 'utter']:
                        utt_text = utt['Text']

                    # If it's a knowledge base query, format it as a string
                    if utt['Action'] == 'query':
                        utt_text = "[QUERY] "
                        for constraint in utt['Constraints']:
                            key = list(constraint.keys())[0]
                            val = constraint[key]

                            utt_text += "{} = {} ; ".format(key, val)

                    # If it's a knowledge base item, format it as a string
                    if utt['Action'] == 'return_item':
                        utt_text = "[RESULT] "
                        if 'Item' not in utt:
                            utt_text += "NO RESULT"
                        else:
                            for key,val in utt['Item'].items():
                                utt_text += "{} = {} ; ".format(key, val)

                    if utt_text != "":
                        history += "[{}] {} [SEP] ".format(utt['Agent'], utt_text.strip())

        # Write to cache
        with open(cached_path, "wb+") as f:
            pickle.dump([self.action_label_to_id, self.examples], f)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]

class NextActionUtteranceDataset(Dataset):
    def __init__(self,
                 data_path,
                 schema_path):

        max_seq_len = 512
        data_dirname = os.path.dirname(os.path.abspath(data_path))
        cached_path = os.path.join(data_dirname, "/home/xiaoying/schema_codex/zeroshot_schema_ms/ic_star/codex_schema_cached_ic_cot")
        if os.path.exists(cached_path):
            with open(cached_path, "rb") as f:
                self.action_to_response, self.action_label_to_id, self.examples = pickle.load(f)
                # print("self.examples \n", self.examples)

            return None

        # Read all of the JSON files in the data directory
        conversations = [
            json.load(open(data_path + fn)) for fn in os.listdir(data_path)
        ]

        # Iterate over the conversations and get (1) the dialogs and (2) the 
        # actions for all wizard turns.
        self.ori_examples = []
        self.action_label_to_id = {}
        for conv in tqdm(conversations):
            # History (so far) for this dialog
            history = ""
            history_list = []
            action_label = ''
            action_label_id = ''
            # previous_action_label = ''
            wizard_action_label_list = []
            # previous_action_label_id = ''

            for i,utt in enumerate(conv['Events']):
                # NOTE: Ground truth action labels only exist when wizard picks suggestion. 
                # We skip all custom utterances for action prediction.
                if utt['Agent'] == 'Wizard' and utt['Action'] in ['query', 'pick_suggestion']:
                    # Tokenize history
                    # processed_history = ' '.join(history.strip().split()[:-1])
                    # history_last_two = history_list[-2] + history_list[-1]
                    history_last_two = ' '.join(history_list[-4:])
                    # print("history list:\n", history_list) #  ['[User] hello [SEP] ']
                    # print("history:\n", history_last_two) # [User] hello [SEP] 
                    processed_history = ' '.join(history_last_two.strip().split())
                
                    # processed_history = ' '.join(history_last_two.strip().split()[:-1])
                
                    # print("processed history: \n", processed_history)
                    # encoded_history  = tokenizer.encode(processed_history)

                    # add wizard utterance
                    if utt['Action'] in ['pick_suggestion']:
                        wizard_uttr = "[wizard] " + utt["Text"].lower() + " [eos]"
                    else:
                        wizard_uttr = '[wizard] query [eos]'

                    # Convert action label to id
                    query_label = 'query'
                    if 'ActionLabel' not in utt:
                        query_check = 'Check' in [e['RequestType'][1:-1] for e in utt['Constraints'] if 'RequestType' in e]
                        query_book = 'Book' in [e['RequestType'][1:-1] for e in utt['Constraints'] if 'RequestType' in e]
                        # In case of a bug, if both book and check are on - we treat it as a check.
                        if query_check:
                            query_label = 'query_check' 
                        elif query_book:
                            query_label = 'query_book' 

                    # if action_label:
                    #     previous_action_label = action_label
                    #     previous_action_label_id = action_label_id
                    action_label = utt['ActionLabel'] if 'ActionLabel' in utt else query_label #  "spaceship_inform_outcome"
                    if action_label not in self.action_label_to_id:
                        self.action_label_to_id[action_label] = len(self.action_label_to_id)
                    action_label_id = self.action_label_to_id[action_label]
                    # Add wizard_action_label_id to the list
                    wizard_action_label_list.append(action_label)

                    if len(wizard_action_label_list) > 1:
                        previous_action_label = wizard_action_label_list[-2]
                    else:
                        previous_action_label = None # indicates no previous wizard action
                    # Include metadata 
                    domains = conv['Scenario']['Domains'] #  [
                                                        # "spaceship"
                                                        # ],
                    tasks = [e['Task'] for e in conv['Scenario']['WizardCapabilities']] #  ["spaceship_access_codes"]
                    happy = conv['Scenario']['Happy'] #  true,
                    multitask = conv['Scenario']['MultiTask'] #  false,
                    if happy:
                        # Add to data
                        self.ori_examples.append({
                            # "input_ids": np.array(encoded_history.ids)[-max_seq_length:],
                            # "attention_mask": np.array(encoded_history.attention_mask)[-max_seq_length:],
                            # "token_type_ids": np.array(encoded_history.type_ids)[-max_seq_length:],
                            "action": action_label_id,
                            "dialog_id": conv['DialogueID'],
                            "domains": domains,
                            "tasks": tasks,
                            "happy": happy,
                            "multitask": multitask,
                            "orig_history": processed_history,
                            "orig_action": action_label, # 'domains': ['bank', 'restaurant', 'trivia'], 'tasks': ['restaurant_book', 'bank_balance', 'bank_fraud_report', 'trivia']
                            "pre_action_label": previous_action_label,
                            "wizard_utterance": wizard_uttr
                        })
                        # print("self.ori \n", self.ori_examples)
                # Process and concatenate to history
                if utt['Agent'] in ['User', 'Wizard', 'KnowledgeBase']:
                    utt_text = ""

                    # If there's text, just use it directly
                    if utt['Action'] in ['pick_suggestion', 'utter']:
                        utt_text = utt['Text']

                    # If it's a knowledge base query, format it as a string
                    if utt['Action'] == 'query':
                        utt_text = "[QUERY] "
                        for constraint in utt['Constraints']:
                            key = list(constraint.keys())[0]
                            val = constraint[key]

                            utt_text += "{} = {} ; ".format(key, val)

                    # If it's a knowledge base item, format it as a string
                    if utt['Action'] == 'return_item':
                        utt_text = "[RESULT] "
                        if 'Item' not in utt:
                            utt_text += "NO RESULT"
                        else:
                            for key,val in utt['Item'].items():
                                utt_text += "{} = {} ; ".format(key, val)

                    if utt_text != "":
                        history += "[{}] {} [SEP] ".format(utt['Agent'].lower(), utt_text.strip().lower())
                        history_item = "[{}] {} [SEP] ".format(utt['Agent'].lower(), utt_text.strip().lower())
                        # history += "[{}] {} ".format(utt['Agent'], utt_text.strip())
                        # history_item = "[{}] {} ".format(utt['Agent'], utt_text.strip())
                        history_list.append(history_item)
        # Check if cached pickle file exists
        schema_data_dirname = os.path.dirname(os.path.abspath(schema_path))
        # print("self.action_label_to_id\n", self.action_label_to_id) 
        #  {'hello': 0, 'party_ask_venue': 1, 'ask_name': 2, 'party_ask_day': 3, 'party_ask_starting_time': 4, 'weather_ask_location': 5, 'query': 6, 'weather_inform_forecast': 7, 'party_ask_number_of_guests': 8, 'query_check': 9, 'party_ask_confirm_booking': 10, 'query_book': 11, 'party_booking_successful': 12, 'anything_else': 13, 'goodbye_1': 14, 'hotel_unavailable': 15, 'hotel_ask_confirm_booking': 16, 'restaurant_ask_restaurant': 17, 'restaurant_ask_size': 18, 'restaurant_ask_confirm_booking': 19, 'doctor_inform_booking_available': 20, 'doctor_inform_booking_successful': 21, 'hotel_ask_hotel': 22, 'hotel_ask_room_number': 23, 'hotel_ask_time': 24, 'hotel_inform_service_request_successful': 25, 'plane_flight_unavailable': 26, 'plane_flight_available': 27, 'plane_reservation_failed': 28, 'plane_reservation_succeeded': 29, 'goodbye_2': 30, 'bank_inform_balance': 31, 'trivia_ask_question_number': 32, 'trivia_inform_answer_incorrect_ask_next': 33, 'trivia_ask_question': 34, 'trivia_inform_answer_2_ask_next': 35, 'trivia_bye': 36, 'ride_ask_confirm_booking': 37, 'ride_confirm_booking': 38, 'ride_bye': 39, 'bank_ask_account_number': 40, 'bank_ask_fraud_details': 41, 'bank_ask_pin': 42, 'bank_inform_fraud_report_submitted': 43, 'restaurant_ask_time': 44, 'restaurant_inform_unavailable': 45, 'restaurant_inform_booking_failed': 46, 'apartment_ask_application_fee_paid': 47, 'apartment_ask_day': 48, 'apartment_inform_booking_successful': 49, 'hotel_ask_service_request': 50, 'hotel_inform_service_request_failed': 51, 'apartment_inform_viewing_unavailable': 52, 'apartment_inform_viewing_available': 53, 'party_ask_end_time': 54, 'trivia_inform_answer_correct_ask_next': 55, 'bank_ask_dob': 56, 'bank_ask_mothers_maiden_name': 57, 'bank_ask_childhood_pets_name': 58, 'restaurant_inform_booking_successful': 59, 'out_of_scope': 60, 'meeting_ask_end_time': 61, 'meeting_inform_unavailable_ask_different_time': 62, 'meeting_inform_confirmed': 63, 'doctor_ask_doctor_name': 64, 'doctor_ask_day': 65, 'doctor_ask_symptoms': 66, 'doctor_inform_booking_unavailable': 67, 'plane_ask_flight_id': 68, 'trip_ask_travel_mode': 69, 'ask_departure_location': 70, 'trip_ask_arrival_location': 71, 'trip_inform_simple_step_ask_proceed': 72, 'hotel_ask_date_from': 73, 'hotel_inform_nothing_found': 74, 'ride_ask_destination': 75, 'ride_ask_departure': 76, 'custom': 77, 'apartment_inform_search_criteria': 78, 'apartment_inform_nothing_found': 79, 'apartment_inform_search_result': 80, 'apartment_ask_start_time': 81, 'trip_inform_last_step_and_done': 82, 'hotel_provide_search_result': 83, 'hotel_reservation_succeeded': 84, 'weather_ask_day': 85, 'ride_inform_changes_successful': 86, 'party_ask_host': 87, 'party_ask_arrival_time': 88, 'party_ask_parking_needed': 89, 'plane_ask_date': 90, 'plane_inform_nothing_found': 91, 'meeting_ask_guest_name': 92, 'spaceship_ask_rank': 93, 'spaceship_ask_code': 94, 'spaceship_ask_code_type': 95, 'spaceship_inform_outcome': 96, 'spaceship_bye': 97, 'hotel_inform_name': 98, 'party_inform_food_drink_criteria': 99, 'bank_inform_cannot_authenticate': 100, 'ride_provide_driver_details': 101, 'spaceship_ask_lock_manufacturer': 102, 'party_confirm_rsvp': 103, 'doctor_inform_doctors_instructions': 104, 'trip_inform_detailed_step': 105, 'trip_ask_departure_time': 106, 'meeting_ask_day': 107, 'meeting_ask_reason': 108, 'apartment_ask_apartment_name': 109, 'hotel_inform_price': 110, 'hotel_inform_rating': 111, 'apartment_ask_num_bedrooms': 112, 'apartment_ask_nearby_pois': 113, 'spaceship_ask_colour_second_cable': 114, 'spaceship_ask_colour_top_cable': 115, 'hotel_reservation_failed': 116, 'hotel_inform_search_criteria': 117, 'hotel_ask_date_to': 118, 'apartment_ask_search_more': 119, 'doctor_ask_start_time': 120, 'plane_inform_flight_details': 121, 'meeting_inform_nothing_found': 122, 'ride_ask_booking_number': 123, 'party_no_venue_available': 124, 'trip_instructions_done': 125, 'party_ask_dietary_restrictions': 126, 'ride_ask_change': 127, 'party_ask_drinks': 128, 'restaurant_inform_search_criteria': 129, 'restaurant_ask_food_type': 130, 'restaurant_ask_location': 131, 'restaurant_ask_delivery': 132, 'restaurant_ask_continue_searching': 133, 'hotel_ask_search_more': 134, 'hotel_ask_price': 135, 'restaurant_ask_takes_reservations': 136, 'restaurant_ask_rating': 137, 'hotel_inform_location': 138, 'plane_ask_arrival_city': 139, 'ride_inform_search_criteria': 140, 'party_venue_not_available': 141, 'hotel_ask_name': 142, 'hotel_ask_rating': 143, 'apartment_ask_price': 144, 'apartment_ask_balcony': 145, 'apartment_ask_elevator': 146, 'hotel_ask_location': 147, 'weather_inform_nothing_found': 148, 'ride_inform_changes_failed': 149, 'plane_request_optional': 150, 'ride_inform_nothing_found': 151, 'apartment_ask_floor': 152, 'plane_ask_more_questions': 153, 'apartment_ask_custom_message': 154, 'meeting_ask_start_time': 155, 'party_booking_failed': 156, 'restaurant_inform_nothing_found': 157, 'party_ask_food': 158, 'apartment_ask_end_time': 159, 'trip_inform_nothing_found': 160, 'hotel_ask_customer_request': 161, 'spaceship_inform_nothing_found': 162, 'doctor_inform_nothing_found': 163, 'bank_inform_nothing_found': 164, 'trivia_inform_nothing_found': 165}

        # Read all of the JSON files in the data directory
        tasks = [
            json.load(open(schema_path + fn + "/" + fn + ".json")) for fn in os.listdir(schema_path)
        ]
        # print("task\n", tasks)
        task_responses = [
            json.load(open(schema_path + fn + "/" + "responses.json")) for fn in os.listdir(schema_path)
        ]
        self.action_to_response = {}


        # Iterate over the schemas and get (1) the prior states and (2) the 
        # next actions.
        # self.demonstrated_examples = []
        node_to_utt = {}
        self.schema_examples = []
        for id_, task in enumerate(tasks):
            ind = 1
            # Get the graph
            graph = task['graph']
            full_graph_utterance = ""
            task_action_list = []
            task_action_label_list = []
            task_node_action_dict = {}
            # task_node_action_dict_retrieval = {}
            node_previous_dict = {}
            node_previous_action_list = []
            task_node_action_list = []
            only_node_utterance = ""
            # For every edge in the graph, get examples of transfer to each action
            for prev,action in graph.items():
                if False and prev in task['r_graph']:
                    sys_utt = '[wizard] ' + task['replies'][task['r_graph'][prev]] + ' [SEP] '
                    utterance = '[user] ' + sys_utt + task['replies'][prev] + ' [SEP]'
                else:
                    utterance = '[user] ' + task['replies'][prev] + ' [SEP]'
                    if prev in task['r_graph']:
                        node_previous_action_list.append(task['r_graph'][prev])
                        node_previous_dict[utterance] = node_previous_action_list[-3:]
                        # node_previous_dict[utterance] = task['r_graph'][prev] #  {'[user] We need your help. The life support is failing! [SEP]': 'hello', '[user] The lock manufacturer is [LOCK_MANUFACTURER] [SEP]': 'spaceship_ask_lock_manufacturer', '[user] The colour of the top cable is [COLOUR] [SEP]': 'spaceship_ask_colour_top_cable', '[user] The colour of the second cable is [COLOUR] [SEP]': 'spaceship_ask_colour_second_cable', '[user] [RESULT] APIName = spaceship_life_support ; Message = Successful! Life support was recovered. ;  [SEP]': 'query', '[user] Thanks [SEP]': 'anything_else'}


                node_to_utt[prev] = utterance.lower()
                # print("node to utterance:\n", node_to_utt)

                # For next action prediction, we can normalize the diff query types
                #if action in ['query_check', 'query_book']:
                #    action = 'query'

                if action not in self.action_label_to_id:
                    continue

                action_label = self.action_label_to_id[action]
                self.action_to_response[action_label] = task['replies'][action] 
                # encoded = tokenizer.encode(utterance)
                current_task_responses = task_responses[id_]
                if utterance:
                    task_node_action_dict[utterance] = action_label 
                    # task_node_action_dict_retrieval[utterance] = str(action_label) + "）" + str(action)
                    
                    if str(action) in current_task_responses.keys():
                        full_graph_utterance += "(" + str(ind) + ") " + utterance + " action: " + str(action_label) + ":" + str(action) + " [wizard] " + current_task_responses[str(action)] + "[eos] "
                    else:
                        full_graph_utterance += "(" + str(ind) + ") " + utterance + " action: " + str(action_label) + ":" + str(action) + "[eos] "
                    node_action = utterance + " action: " + str(action_label) + ":" + str(action) + "[eos]"
                    task_node_action_list.append(node_action)
                    # only_node_utterance += utterance
                    action_part = str(action_label) + ":" + str(action)
                    action_part = action_part.lower()
                    # action_part = str(action_label) + ":" + tokenizer.eos_token
                    task_action_list.append(action_part)
                    task_action_label_list.append(int(action_label))
                    ind += 1

            task_action_list = list(set(task_action_list))
            self.schema_examples.append({"full_graph":full_graph_utterance, "tasks": task['task'], "task_action_list": task_action_list, "task_node_action_dict": task_node_action_dict, "node_previous_dict": node_previous_dict, "task_action_label_list": task_action_label_list, "task_node_action_list": task_node_action_list})

        self.examples = []
        for i, ex in enumerate(self.ori_examples):
            input_text = ""
            output_text = ""
            for schema_ex in self.schema_examples:
                if schema_ex["tasks"] in ex['tasks']:
                    input_text = "task: " + schema_ex['tasks'] + " task rules: " + schema_ex["full_graph"] + " history: " + ex["orig_history"]
                    output_text = str(ex["action"]) + ":" + str(ex["orig_action"]) + "[eos]"
                    ex["input_text"] = input_text.lower()
                    ex["output_text"] = output_text.lower() # 'output_text': '79:apartment_inform_nothing_found[eos]'
                    ex["task_action_list"] = schema_ex["task_action_list"]
                    ex["task_action_label_list"] = schema_ex["task_action_label_list"]
                    self.examples.append(ex)


        # json.dump(self.schema_examples, open("./schema_examples.json", 'w'), indent=2)   
        # Write to cache
        with open(cached_path, "wb+") as f:
            pickle.dump([self.action_to_response, self.action_label_to_id, self.examples], f)
        
    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]

def collate_fn(data, tokenizer):
    # print("data:\n", data)
    batch_data = {}
    for key in data[0]:
        batch_data[key] = [d[key] for d in data]
    # print("batch data:\n", batch_data)
    input_batch = tokenizer(batch_data["input_text"], padding=True, return_tensors="pt", add_special_tokens=False, verbose=False)
    batch_data["encoder_input"] = input_batch["input_ids"]
    batch_data["attention_mask"] = input_batch["attention_mask"]
    output_batch = tokenizer(batch_data["output_text"], padding=True, return_tensors="pt", add_special_tokens=False, return_attention_mask=False, truncation=True, max_length=50)
    # replace the padding id to -100 for cross-entropy
    output_batch['input_ids'].masked_fill_(output_batch['input_ids']==tokenizer.pad_token_id, -100)
    batch_data["decoder_output"] = output_batch['input_ids']

    return batch_data

def gpt_collate_fn(data,tokenizer):
    batch_data = {}
    for key in data[0]:
        batch_data[key] = [d[key] for d in data]

    output_batch = tokenizer(batch_data["output_text"], padding=True, return_tensors="pt", add_special_tokens=False, return_attention_mask=False, truncation=True, max_length=1000)
    batch_data["input_ids"] = output_batch['input_ids']
    return batch_data

