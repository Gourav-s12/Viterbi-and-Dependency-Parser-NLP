#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import re


# In[2]:


#read dataset function
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
                if '-' in parts[0]:
                    continue

                head_value = parts[4] if parts[4] != '_' else None

                dependacy[parts[0]] = {
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


file_path = 'train.txt'
parsed_dataset_train = parse_dataset(file_path)

# Print the parsed dataset for inspection
for sentence in parsed_dataset_train[:4]:
    print(sentence)


# In[4]:


# get all the post tags
from collections import Counter

# Initialize a Counter to store the frequency of each POS tag
pos_tag_frequency = Counter()

# Iterate through the dataset to collect unique POS tags and their frequency counts
for sentence_data in parsed_dataset_train:
    pos_tags = sentence_data['pos_tag']
    pos_tag_frequency.update(pos_tags)


# In[5]:


# Print the total number of unique POS tags
print("Total unique POS tags:", len(pos_tag_frequency))
# Print the unique POS tags and their frequency counts
print("Unique POS tags and their frequency counts:")
for tag, count in pos_tag_frequency.items():
    print(f"{tag}: {count}")


# In[6]:


# get text in a sentance
def get_text(sentence_data):
  return [ row['original_word'] for row in sentence_data['dependacy'].values()]

print(f"Original text {get_text(parsed_dataset_train[6])}")
print(f"Pos Tag {parsed_dataset_train[6]['pos_tag']}")


# In[7]:


# get unique words
unique_words= set()
# Iterate through the dataset to collect unique POS tags
for sentence_data in parsed_dataset_train:
    words = get_text(sentence_data)
    unique_words.update(words)


# In[8]:


# Print the total number of unique POS tags
print("Total unique POS tags:", len(unique_words))

# Print the unique POS tags
# print("Unique words:", unique_words)


# In[9]:


import collections
from collections import defaultdict


# In[10]:


# transition matrix
def compute_transition_probabilities(parsed_dataset):
    # Initialize a dictionary to store transition counts
    transition_counts = defaultdict(int)

    # Count transitions in the dataset
    for sentence_data in parsed_dataset:
        pos_tags = sentence_data['pos_tag']
        for i in range(len(pos_tags) - 1):
            transition_counts[(pos_tags[i], pos_tags[i + 1])] += 1

    # Compute transition probabilities
    transition_probabilities = {transition: count / pos_tag_frequency[transition[0]] for transition, count in transition_counts.items()}

    return transition_probabilities


# In[11]:


# emission matrix
def compute_emission_probabilities(parsed_dataset):
    # Initialize a dictionary to store emission counts
    emission_counts = defaultdict(int)

    # Count emissions in the dataset
    for sentence_data in parsed_dataset:
        pos_tags = sentence_data['pos_tag']
        words = get_text(sentence_data)   # [token_data['pos'] for token_data in sentence_data.values() if isinstance(token_data, dict)]
        for pos_tag, word in zip(pos_tags, words):
            emission_counts[(pos_tag, word)] += 1

    # Compute emission probabilities
    emission_probabilities = {emission: (count + 1) / (pos_tag_frequency[emission[0]] + len(unique_words))  for emission, count in emission_counts.items()}
    # emission_probabilities = {emission: count / pos_tag_frequency[emission[0]] for emission, count in emission_counts.items()}

    return emission_probabilities


# In[12]:


# start transition matrix
def compute_start_probabilities(parsed_dataset):
    # Initialize a dictionary to store start counts
    start_counts = defaultdict(int)

    # Count starts in the dataset
    for sentence_data in parsed_dataset:
        start_tag = sentence_data['pos_tag'][0]
        start_counts[start_tag] += 1

    # Compute start probabilities
    total_sentences = len(parsed_dataset)
    start_probabilities = {tag: count / total_sentences for tag, count in start_counts.items()}

    return start_probabilities


# In[13]:


# generating all the matrix
transition_probabilities = compute_transition_probabilities(parsed_dataset_train)
emission_probabilities = compute_emission_probabilities(parsed_dataset_train)
start_probabilities = compute_start_probabilities(parsed_dataset_train)


# In[14]:


# sum(emission_probabilities.values()), sum(transition_probabilities.values()), sum(start_probabilities.values())


# In[15]:


# Print the computed probabilities
print("Transition Probabilities:", transition_probabilities)
print("Emission Probabilities:", emission_probabilities)
print("Start Probabilities:", start_probabilities)


# In[16]:


len(transition_probabilities), len(emission_probabilities), len(start_probabilities)


# In[17]:


# reading text file:
file_path = 'test.txt'
parsed_dataset_test = parse_dataset(file_path)

# Print the parsed dataset for inspection
for sentence in parsed_dataset_test[:4]:
    print(sentence)


# In[18]:


import math
# viterbi algo funtion which return best pos tag path and smooth_words (unseen words in training)
def viterbi_algorithm(sentence_data, transition_probabilities, emission_probabilities, start_probabilities):
    pos_tags = pos_tag_frequency.keys()  # sentence_data['pos_tag']
    words = get_text(sentence_data)
    value_for_unseen_transition = 1e-10
    value_for_unseen_emission = (1/len(unique_words))   # 1e-10
    smooth_words = set()

    # Initialize the Viterbi matrix
    viterbi_matrix = [{} for _ in range(len(words))]
    backpointer_matrix = [{} for _ in range(len(words))]

    # Initialization step
    for tag in pos_tags:
        start_prob = start_probabilities.get(tag, value_for_unseen_transition)  # Use a default value, e.g., 1e-10
        emission_prob = emission_probabilities.get((tag, words[0]), value_for_unseen_emission)  # Use a default value, e.g., 1e-10
        viterbi_matrix[0][tag] = math.log(start_prob) + math.log(emission_prob)

    # Recursion and termination steps
    for t in range(1, len(words)):
        max_emission_prob = value_for_unseen_emission
        # if words[t] not in unique_words:
        #     smooth_words.add(words[t])
             # print(words[t])
       

        for current_tag in pos_tags:
            max_prob = float('-inf')
            best_prev_tag = None

            for prev_tag in pos_tags:
                transition_prob = transition_probabilities.get((prev_tag, current_tag), value_for_unseen_transition)  
                emission_prob = emission_probabilities.get((current_tag, words[t]), value_for_unseen_emission)  

                current_prob = viterbi_matrix[t - 1][prev_tag] + math.log(transition_prob) + math.log(emission_prob)

                if current_prob > max_prob:
                    max_prob = current_prob
                    best_prev_tag = prev_tag

                if max_emission_prob < emission_prob:
                    max_emission_prob = emission_prob 

            viterbi_matrix[t][current_tag] = max_prob
            backpointer_matrix[t][current_tag] = best_prev_tag
            
        if max_emission_prob == value_for_unseen_emission:
            smooth_words.add(words[t])

    # Find the best path
    best_path = []
    max_prob_last_tag = max(viterbi_matrix[-1].values())
    for tag, prob in viterbi_matrix[-1].items():
        if prob == max_prob_last_tag:
            best_last_tag = tag
            break

    best_path.append(best_last_tag)
    for t in range(len(words) - 1, 0, -1):
        best_prev_tag = backpointer_matrix[t][best_last_tag]
        best_path.insert(0, best_prev_tag)
        best_last_tag = best_prev_tag

    return best_path, smooth_words


# In[19]:


# calculate accuracy
def calculate_accuracy(true_tag, pred_tag):
    print(len(true_tag), len(pred_tag))
    if len(true_tag) != len(pred_tag):
        raise ValueError("Input lists must have the same length")

    correct_predictions = sum(1 for true, pred in zip(true_tag, pred_tag) if true == pred)
    total_predictions = len(true_tag)

    accuracy = correct_predictions / total_predictions

    return accuracy


# In[20]:


from sklearn import metrics
# calculate all other metrics
def calculate_metrics(true_tags, pred_tags):
    # Calculate accuracy
    accuracy = metrics.accuracy_score(true_tags, pred_tags)

    # Calculate precision, recall, and F1-score
    precision = metrics.precision_score(true_tags, pred_tags, average='weighted', zero_division=1)
    recall = metrics.recall_score(true_tags, pred_tags, average='weighted', zero_division=1)
    f1_score = metrics.f1_score(true_tags, pred_tags, average='weighted', zero_division=1)

    return accuracy, precision, recall, f1_score


# In[21]:


# output file write 
def write_tsv(output_file_path):
    with open(output_file_path, "w", encoding='utf-8') as output_file:
        # Write the header
        output_file.write("sent_id\ttoken_number\tword\tpredicted_POS_tag\n")
    
        # Write the predictions
        for sent_id, word, pred_tag in predictions_output:
            for i in range(len(pred_tag)):
                output_file.write(f"{sent_id}\t{i}\t{word[i]}\t{pred_tag[i]}\n")
    
        output_file.close()
        print(f"Predictions saved to {output_file_path}")


# ## on Training

# In[22]:


true_tags = []
pred_tags = []
# Initialize a list to store the predictions
predictions_output = []
unseen_words = []


# In[23]:


# Apply Viterbi algorithm to each sentence in the train dataset
for sentence_data_train in parsed_dataset_train:
    best_path_train, unseen_word = viterbi_algorithm(sentence_data_train, transition_probabilities, emission_probabilities, start_probabilities)
    # print(f"Sentence: {sentence_data_train['text']}")
    # print(f"Original POS Tags: {sentence_data_train['pos_tag']}")
    # print(f"Predicted POS Tags: {best_path_train}\n")


    true_tags.extend(sentence_data_train['pos_tag'])
    pred_tags.extend(best_path_train)
    unseen_words.extend(unseen_word)

    # Save predictions to the list
    predictions_output.append((sentence_data_train['sent_id'], get_text(sentence_data_train), best_path_train))


# In[24]:


accuracy = calculate_accuracy(true_tags, pred_tags)
print(f"Training Accuracy: {accuracy * 100:.2f}%")


# In[25]:


accuracy, precision, recall, f1_score = calculate_metrics(true_tags, pred_tags)
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-score: {f1_score:.4f}")


# In[26]:


print(f"trains smooth words {len(unseen_words)} should be 0")


# In[27]:


# Save the predictions to a file
output_file_path = "viterbi_predictions_train.tsv"
write_tsv(output_file_path)


# ## on Testing

# In[28]:


true_tags = []
pred_tags = []
# Initialize a list to store the predictions
predictions_output = []
smooth_words = set()


# In[29]:


# Apply Viterbi algorithm to each sentence in the test dataset
for sentence_data_test in parsed_dataset_test:
    best_path_test , smooth_word = viterbi_algorithm(sentence_data_test, transition_probabilities, emission_probabilities, start_probabilities)
    # print(f"Sentence: {sentence_data_test['text']}")
    # print(f"Original POS Tags: {sentence_data_test['pos_tag']}")
    # print(f"Predicted POS Tags: {best_path_test}\n")


    true_tags.extend(sentence_data_test['pos_tag'])
    pred_tags.extend(best_path_test)

    # Save predictions to the list
    predictions_output.append((sentence_data_test['sent_id'], get_text(sentence_data_test), best_path_test))
    smooth_words.update(smooth_word)


# In[30]:


accuracy = calculate_accuracy(true_tags, pred_tags)
print(f"Test Accuracy: {accuracy * 100:.2f}%")


# In[31]:


accuracy, precision, recall, f1_score = calculate_metrics(true_tags, pred_tags)
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-score: {f1_score:.4f}")


# In[32]:


# Save the predictions to a file
output_file_path = "viterbi_predictions_test.tsv"
write_tsv(output_file_path)


# In[33]:


unique_words_test= set()
# Iterate through the dataset to collect unique POS tags
for sentence_data in parsed_dataset_test:
    words = get_text(sentence_data)
    unique_words_test.update(words)


# In[34]:


# Print the total number of unique POS tags
print("Total unique words:", len(unique_words_test))

# Print the unique POS tags
# print("Unique words:", unique_words_test)


# In[35]:


print(f"test smooth words {len(smooth_words)}")
print(smooth_words)


# In[ ]:




