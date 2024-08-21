#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import re


# In[2]:


# reading dataset 
def parse_dataset(file_path):
    dataset = []
    current_sentence = None
    dependacy = None

    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()

            if line.startswith('# sent_id'):
                # Start of a new sentence
                if current_sentence:
                    current_sentence['dependacy'] = dependacy
                    dataset.append(current_sentence)
                current_sentence = {'sent_id': line.split('=')[1].strip(), 'text': ''}
            elif line.startswith('# text'):
                # Text of the sentence
                current_sentence['text'] = line.split('=')[1].strip()
                current_sentence['pos_tag'] = []
                dependacy = dict()
            elif line:
                # Details about each word in the sentence
                parts = line.split()
                if '-' in parts[0] or '.' in parts[0]:
                    continue

                head_value = parts[4] if parts[4] != '_' else None

                parts[0] = int(parts[0]) if parts[0] is not None else None

                dependacy[parts[0]] = {
                        'index': parts[0],
                        'original_word': parts[1],
                        'word': parts[2],
                        'pos': parts[3],
                        'head': int(head_value) if head_value is not None else None,
                        'dependency_relation': parts[5]
                    }
                current_sentence['pos_tag'].append(parts[3])

    # Append the last sentence to the dataset
    if current_sentence:
        current_sentence['dependacy'] = dependacy
        dataset.append(current_sentence)

    return dataset


# In[3]:


# reading dataset train
file_path = 'train.txt'
parsed_dataset_train = parse_dataset(file_path)

# Print the parsed dataset for inspection
for sentence in parsed_dataset_train[:4]:
    print(sentence)


# In[4]:


# get text in a sentence
def get_text(sentence_data):
  return [ row['original_word'] for row in sentence_data['dependacy'].values()]

print(f"Original text {get_text(parsed_dataset_train[6])}")
print(f"Pos Tag {parsed_dataset_train[6]['pos_tag']}")


# In[5]:


# configuration class and oracle
class Configuration:
    def __init__(self, sentence_info):
        self.sentence_info = sentence_info
        self.original_sentence = " ".join(get_text(sentence_info))
        self.dependency = sentence_info['dependacy']
        self.stack = []
        self.buffer = list(sentence_info['dependacy'].values())   # get_text(sentence_info)
        self.arcs = []

    def do_left_arc(self):
        if len(self.stack) < 1 and len(self.buffer) < 1:
            raise ValueError("Cannot perform LEFT-ARC: Stack or Buffer has fewer than 1 item")
        dependent = self.stack.pop()
        head = self.buffer[0]
        self.arcs.append((head, dependent))

    def do_right_arc(self):
        if len(self.stack) < 1 and len(self.buffer) < 1:
            raise ValueError("Cannot perform RIGHT-ARC: Stack or Buffer has fewer than 1 item")
        head = self.stack[-1]
        dependent = self.buffer.pop(0)
        self.stack.append(dependent)
        self.arcs.append((head, dependent))

    def do_reduce(self):
        if len(self.stack) < 1:
            raise ValueError("Cannot perform REDUCE: Stack is empty")
        head = self.stack[-1]
        for arc in self.arcs:
            if arc[1]['original_word'] == head['original_word']:
                self.stack.pop()
                return
        raise ValueError("Cannot perform REDUCE: stack top head is still unknown")

    def do_shift(self):
        if len(self.buffer) < 1:
            raise ValueError("Cannot perform SHIFT: Buffer is empty")
        word = self.buffer.pop(0)
        self.stack.append(word)

    def get_head(self, text):
        for v in self.dependency.values():
            if v['head'] == 0:
                return 0
            else:
                return self.dependency[v['head']]['original_word']

    def curr_config(self):
        print("\nCurrent stack:", [ row['original_word'] for row in self.stack])
        print("Current buffer:", [ row['original_word'] for row in self.buffer])
        print("Current arcs:", [ (head['original_word'], dep['original_word']) for head, dep in self.arcs])

    # get oracle transaction
    def oracle_transition(self):
        #if buffer is empty
        if len(self.buffer) == 0:
            return "END"
        
        # if stack empty 
        if len(self.stack) == 0:
            return "SHIFT"
            
        # Rule 1: LEFT-ARC
        if self.buffer and self.stack:
            top_s = self.stack[-1]
            first_b = self.buffer[0]
            # top_s -> first_b and not top_s -> any in current arc 
            if first_b['index'] == top_s['head'] and \
               (top_s['head'] != 0 or not any(dep['index'] == top_s['index'] for head, dep in self.arcs)):
                return "LEFT-ARC"

        # Rule 2: RIGHT-ARC
        if self.buffer and self.stack:
            top_s = self.stack[-1]
            first_b = self.buffer[0]
            # top_s <- first_b
            if top_s['index'] == first_b['head']:
                return "RIGHT-ARC"

        # Rule 3: REDUCE
        if self.stack:
            top_s = self.stack[-1]
            first_b = self.buffer[0]
            # dependency_list = [ (row['head'],row['index']) for row in self.dependency.values() if row['head'] != 0 ]
            arc = [ (head['index'],dep['index']) for  head, dep in self.arcs ]
            # w -> first_b or w <- first_b and w != top_s and top_s <- any in current arc 
            if any(head['index'] == top_s['head'] for head, dep in self.arcs):
                for word in self.stack:
                    if word != top_s:
                        # if (word['index'], first_b['index']) in dependency_list or \
                        #    (first_b['index'], word['index']) in dependency_list:
                        if ((word['index'], first_b['index']) in arc) or ((first_b['index'], word['index']) in arc):
                            return "REDUCE"

        # Rule 4: SHIFT
        # if nothing works shift
        return "SHIFT"


# In[6]:


# testing:
# sentence_info = {'sent_id': 'GUM_academic_art-1', 'text': 'Aesthetic Appreciation and Spanish Art:', 'pos_tag': ['JJ', 'NN', 'CC', 'JJ', 'NN', ':'], 'dependacy': {'1': {'original_word': 'Aesthetic', 'word': 'aesthetic', 'pos': 'JJ', 'head': 2, 'dependency_relation': 'amod'}, '2': {'original_word': 'Appreciation', 'word': 'appreciation', 'pos': 'NN', 'head': 0, 'dependency_relation': 'root'}, '3': {'original_word': 'and', 'word': 'and', 'pos': 'CC', 'head': 5, 'dependency_relation': 'cc'}, '4': {'original_word': 'Spanish', 'word': 'Spanish', 'pos': 'JJ', 'head': 5, 'dependency_relation': 'amod'}, '5': {'original_word': 'Art', 'word': 'art', 'pos': 'NN', 'head': 2, 'dependency_relation': 'conj'}, '6': {'original_word': ':', 'word': ':', 'pos': ':', 'head': 2, 'dependency_relation': 'punct'}}}
sentence_info = parsed_dataset_train[0]
config = Configuration(sentence_info)

config.curr_config()
# Perform transitions
print(config.oracle_transition())
config.do_shift()
print(config.oracle_transition())
# config.curr_config()
config.do_left_arc()
print(config.oracle_transition())
# config.curr_config()
config.do_shift()
print(config.oracle_transition())
# config.curr_config()
config.do_shift()
print(config.oracle_transition())
# config.curr_config()
config.do_shift()
print(config.oracle_transition())
# config.curr_config()
config.do_left_arc()
print(config.oracle_transition())
# config.curr_config()
config.do_left_arc()
print(config.oracle_transition())
# config.curr_config()
config.do_right_arc()
print(config.oracle_transition())
# config.curr_config()
config.do_reduce()
print(config.oracle_transition())
# config.curr_config()
config.do_right_arc()
print(config.oracle_transition())
config.curr_config()


# In[7]:


from collections import Counter
# pos_tag count for P_size
# Initialize a Counter to store the frequency of each POS tag
pos_tag_frequency = Counter()

# Iterate through the dataset to collect unique POS tags and their frequency counts
for sentence_data in parsed_dataset_train:
    pos_tags = sentence_data['pos_tag']
    pos_tag_frequency.update(pos_tags)


# In[8]:


# Print the total number of unique POS tags
print("Total unique POS tags:", len(pos_tag_frequency))
# Print the unique POS tags and their frequency counts
# print("Unique POS tags and their frequency counts:")
# for tag, count in pos_tag_frequency.items():
#     print(f"{tag}: {count}")


# In[9]:


# normalized_tokens count for V_size
unique_words= Counter()
# Iterate through the dataset to collect unique POS tags
for sentence_data in parsed_dataset_train:
    words = get_text(sentence_data)
    unique_words.update(words)


# In[10]:


# Print the total number of unique POS tags
print("Total unique words:", len(unique_words))


# In[11]:


# normalized_tokens count for V_size
total_sentence_in_training = len(parsed_dataset_train)
# Get normalized tokens
normalized_tokens = [(k, v) for k, v in unique_words.items() if v < total_sentence_in_training]
# Sort based on frequency of occurrence (v)
normalized_tokens = sorted(normalized_tokens, key=lambda x: x[1], reverse=True)[:1000]
# Extract only the tokens without frequencies
normalized_tokens = [token for token, _ in normalized_tokens]
len(normalized_tokens)


# In[12]:


# Dependency_relation count for R_size
Dependency_relation = set()
root_list = []
# Iterate through the dataset to collect unique POS tags and their frequency counts
for sentence_data in parsed_dataset_train:
    for v in sentence_data['dependacy'].values():
        if v['dependency_relation'] == 'root':
            root_list.append(v['original_word'])
        Dependency_relation.add(v['dependency_relation'])


# In[13]:


# Print the total number of Dependency_relation
print("Total Dependency relation:", len(Dependency_relation))


# In[14]:


def get_feature_vector(conf, transition, vocabulary = normalized_tokens, pos_tags = pos_tag_frequency, dependency_relations = Dependency_relation):
    # Initialize the feature vector with zeros
    pos_tag = list(pos_tags.keys())
    dependency_relation = list(dependency_relations)
    V_size = len(vocabulary)
    R_size = len(dependency_relation)
    P_size = len(pos_tag)
    
    feature_vector_size = 4 * (2 * V_size + 3 * P_size + 4 * R_size)
    f = np.zeros(feature_vector_size, dtype=int)
    S, B, A = conf.stack, conf.buffer, conf.arcs
    
    # Helper function to update feature vector for a given feature index 
    def set_feature (idx, value=1):
        if 0 <= idx < feature_vector_size:
            f[idx] = value

    def update_top_S_DEP(top_S, offset):
        head_index = top_S['index']
        if head_index != 0:  # If head is not root
            dep_relation = top_S['dependency_relation']
            # Update TOP.DEP
            dep_index = offset + V_size + P_size + dependency_relation.index(dep_relation)
            # print("top_dep")
            set_feature(dep_index)

            # Update TOP.LDEP
            left_dep = get_left_most_dependency(head_index)
            if left_dep:
                left_dep_relation = left_dep['dependency_relation']
                left_dep_index = offset + V_size + P_size + R_size + dependency_relation.index(left_dep_relation)
                # print("top_l_dep")
                set_feature(left_dep_index)

            # Update TOP.RDEP
            right_dep = get_right_most_dependency(head_index)
            if right_dep:
                right_dep_relation = right_dep['dependency_relation']
                right_dep_index = offset + V_size + P_size + 2 * R_size + dependency_relation.index(right_dep_relation)
                # print("top_r_dep")
                set_feature(right_dep_index)
    
    def update_buffer_DEP(first_B, offset):
        head_index = first_B['index']
        if head_index != 0:  # If head is not root
            # Update FIRST.LDEP
            left_dep = get_left_most_dependency(head_index)
            if left_dep:
                left_dep_relation = left_dep['dependency_relation']
                left_dep_index = offset + 2 * V_size + 2 * P_size + 3 * R_size + dependency_relation.index(left_dep_relation)
                # print("1st_l_dep")
                set_feature(left_dep_index)
    
    def get_left_most_dependency(head_index):
        left_most = None
        for arc in A:
            if arc[1]['head'] == head_index:
                if not left_most or arc[1]['index'] < left_most['index']:
                    left_most = arc[1]
        return left_most
    
    def get_right_most_dependency(head_index):
        right_most = None
        for arc in A:
            if arc[1]['head'] == head_index:
                if not right_most or arc[1]['index'] > right_most['index']:
                    right_most = arc[1]
        return right_most

    
    # Feature indices based on the transition 
    transition_offset = {'LEFT-ARC': 0, 'RIGHT-ARC': 1, 'REDUCE': 2, 'SHIFT': 3}[transition] * (2 * V_size + 3 * P_size + 4 * R_size)
    
    # TOP feature
    if S:
        top_S = S[-1]  # Last stack item is the top
        # TOP
        top_token = top_S['original_word']  # Normalized token
        if top_token in vocabulary:
            # print("top_voc")
            set_feature(vocabulary.index(top_token) + transition_offset)
        
        # TOP.POS
        top_POS = top_S['pos']  # POS_tag for TOS
        if top_POS in pos_tag:
            # print("top_pos")
            set_feature(V_size + pos_tag.index(top_POS) + transition_offset)
        
        update_top_S_DEP(top_S, transition_offset)
        
    # FIRST feature
    if B:
        first_B = B[0]  # First buffer item
        # FIRST
        first_token = first_B['original_word']  # Normalized token
        if first_token in vocabulary:
            # print("1st_voc")
            set_feature(V_size + P_size + 3 * R_size + vocabulary.index(first_token) + transition_offset)
        
        # FIRST.POS
        first_POS = first_B['pos']  # POS_tag for first (Buffer)
        if first_POS in pos_tag:
            # print("1st_pos")
            set_feature(2 * V_size + P_size + 3 * R_size + pos_tag.index(first_POS) + transition_offset)
        
        update_buffer_DEP(first_B, transition_offset)
    
    # LOOK.POS
    if len(B) >= 2:
        second_pos = B[1]
        look_POS = second_pos['pos']
        if look_POS in pos_tag:
            # print("2nd_pos")
            set_feature(2 * V_size + 2 * P_size + 4 * R_size + pos_tag.index(look_POS) + transition_offset)
    
    return f


# In[15]:


import copy
# to generate training data
def get_training_data(sentences):
    training_instances = []
    error = 0
    for index, sentence in enumerate(sentences):
        # Initial configuration for the sentence
        config = Configuration(sentence)
        is_this_sentence_error = False
        training_instances_per_sentence = []
        while len(config.buffer) >= 1: # Continue until buffer is empty
            # Determine the gold-standard action using the oracle
            gold_action = config.oracle_transition()
            # Encode current configuration and gold action into feature vector
            training_instances_per_sentence.append((copy.deepcopy(config), gold_action))
            # Update configuration based on the gold action
            try:
                if gold_action == 'LEFT-ARC':
                    config.do_left_arc()
                elif gold_action == 'RIGHT-ARC':
                    config.do_right_arc()
                elif gold_action == 'REDUCE':
                    config.do_reduce()
                elif gold_action == 'SHIFT':
                    config.do_shift()
                else:
                    print("Invalid action for sentence index:", index)
                    print("Invalid action:", gold_action)
                    break
            except:
                error += 1
                # print(index)
                is_this_sentence_error = True # just to ensure there is no error in original gold tag data
                break
        if not is_this_sentence_error:
            training_instances.extend(training_instances_per_sentence)
    return training_instances, error


# In[16]:


train_data, error = get_training_data(parsed_dataset_train)


# In[17]:


print(f"number of training data = {len(train_data)}")
print(f"number of error = {error} out of {len(parsed_dataset_train)}")


# In[18]:


import random
from tqdm import tqdm
# to train the weights
def training_classifier(train_data, epochs, vocabulary = normalized_tokens, pos_tags = pos_tag_frequency, dependency_relations = Dependency_relation):
    # Initialize weights with a vector of all 1s
    pos_tag = list(pos_tags.keys())
    dependency_relation = list(dependency_relations)
    V_size = len(vocabulary)
    R_size = len(dependency_relation)
    P_size = len(pos_tag)
    
    feature_vector_size = 4 * (2 * V_size + 3 * P_size + 4 * R_size)  # Assuming all configurations have the same feature vector size
    w = np.ones(feature_vector_size)
    
    for epoch in tqdm(range(epochs)):
        # random.shuffle(train_data) # we can on this to get better acuuracy
        for config, gold_action in train_data:
            
            t_star = None
            max_score = float('-inf')
            for transition in ['LEFT-ARC', 'RIGHT-ARC', 'REDUCE', 'SHIFT']:
                score = get_score(config, transition, w)
                if score > max_score:
                    max_score = score
                    t_star = transition
            
            t_gold = gold_action  # Gold-standard action
            # update w if gold and star is not same
            if t_star != t_gold:
                w += get_feature_vector(config, t_gold) - get_feature_vector(config, t_star)
                
    return w

def get_score(config, transition, weights):
    # Calculate the score for a given transition and configuration
    feature_vector = get_feature_vector(config, transition)
    return np.dot(weights, feature_vector)


# In[19]:


w = training_classifier(train_data, 5)


# In[20]:


Counter(w)


# In[21]:


# Name of the file to store the array
file_name = "dependency_model_on.npy"
# Save the array to a .npy file
np.save(file_name, w)
print(f"vector weight stored in '{file_name}'")


# In[22]:


# Name of the file to store the array
file_name = "dependency_model_on.npy"
# Load the array from the file
w = np.load(file_name)


# In[23]:


# read test.txt to make dataset
file_path = 'test.txt'
parsed_dataset_test = parse_dataset(file_path)

# Print the parsed dataset for inspection
for sentence in parsed_dataset_test[:4]:
    print(sentence)


# In[24]:


def get_score(config, transition, weights):
    # Calculate the score for a given transition and configuration
    feature_vector = get_feature_vector(config, transition)
    return np.dot(weights, feature_vector)

# calculate UAS score
def calculate_uas(gold_heads, predicted_heads):
    total_tokens = len(gold_heads.keys())
    correctly_predicted = 0
    
    for key in gold_heads.keys():
        g_head, _ = gold_heads[key]
        # print(g_head)
        if predicted_heads.get(key) is None:
            continue
        p_head, _ = predicted_heads[key] 
        if p_head == g_head:
            correctly_predicted += 1
    return correctly_predicted / total_tokens

# def calculate_uas(gold_arcs, predicted_arcs):
#     correct_arcs = 0
#     for arc in gold_arcs:
#         if arc in predicted_arcs:
#             correct_arcs += 1
#     return correct_arcs / len(gold_arcs)


# In[25]:


def gold_arcs(sentence, weights):
    config = Configuration(sentence)
    error = False
    while len(config.buffer) >= 1: # Continue until buffer is empty
        # Determine the gold-standard action using the oracle
        gold_action = config.oracle_transition()
        # Update configuration based on the gold action
        try:
            if gold_action == 'LEFT-ARC':
                config.do_left_arc()
            elif gold_action == 'RIGHT-ARC':
                config.do_right_arc()
            elif gold_action == 'REDUCE':
                config.do_reduce()
            elif gold_action == 'SHIFT':
                config.do_shift()
            else:
                print("Invalid action for sentence index:", index)
                print("Invalid action:", gold_action)
                break
        except:
            error = True # just to ensure there is no error in original gold tag data
            break
    dependency_dict = {dep['index']: (head['index'],dep['original_word']) for head, dep in config.arcs}
    return dependency_dict, error 


# In[26]:


def predict_arcs(sentence, weights):
    config = Configuration(sentence)
    while len(config.buffer) >= 1: # Continue until buffer is empty
        # Determine the gold-standard action using the oracle
        t_star = None
        max_score = float('-inf')
        for transition in ['LEFT-ARC', 'RIGHT-ARC', 'REDUCE', 'SHIFT']:
            score = get_score(config, transition, weights)
            if score > max_score:
                max_score = score
                t_star = transition

        # Update configuration based on the gold action
        try:
            if len(config.stack) == 0: # heuristic
                config.do_shift()
            elif t_star == 'LEFT-ARC':
                if any(dep['index'] == config.stack[-1]['index'] for head, dep in config.arcs): # heuristic
                    config.do_reduce()
                else:
                    config.do_left_arc()
            elif t_star == 'RIGHT-ARC':
                config.do_right_arc()
            elif t_star == 'REDUCE':
                if not any(dep['index'] == config.stack[-1]['index'] for head, dep in config.arcs): # heuristic
                    config.do_left_arc()
                else:
                    config.do_reduce()
            elif t_star == 'SHIFT':
                config.do_shift()
            else:
                print("Invalid action for sentence index:", index)
                print("Invalid action:", t_star)
                break
        except:
            
            break
    # dependency_list = [ (head['original_word'],dep['original_word']) for head, dep in config.arcs ]
    dependency_dict = {dep['index']: (head['index'],dep['original_word']) for head, dep in config.arcs}
    return dependency_dict


# In[27]:


def predict_test(sentences, weights):
    # Predict transitions for each sentence in the test data
    predicted_arcs = []
    error = 0
    score = 0
    total_len = 0
    predictions_output = []
    for index, sentence in enumerate(sentences):
        
        gold_heads, error = gold_arcs(sentence, weights)
        predicted_heads = predict_arcs(sentence, weights)
        # print(gold_heads, predicted_heads)
        if not error: # just to ensure there is no error in original gold tag data
            total_len += 1
            score += calculate_uas(gold_heads, predicted_heads)
            # Save predictions to the list
            predictions_output.append((sentence['sent_id'], predicted_heads))
    return score / total_len , predictions_output , total_len


# In[28]:


# predicting on test data
test_acc, predictions_output, data_len = predict_test(parsed_dataset_test, w)


# In[29]:


print(f"Accuracy for {data_len} test data: {test_acc*100:.4f}%")


# In[30]:


# output file write 
output_file_path =  "dependency_predictions_on.tsv"
with open(output_file_path, "w", encoding='utf-8') as output_file:
    # Write the header
    output_file.write("sent_id\ttoken_number\tword\tpredicted_head_index\n")

    # Write the predictions
    for sent_id, pred_head_dict in predictions_output:
        for token, arc in pred_head_dict.items():
            head, word = arc
            output_file.write(f"{sent_id}\t{token}\t{word}\t{head}\n")

    output_file.close()
    print(f"Predictions saved to {output_file_path}")


# In[ ]:




