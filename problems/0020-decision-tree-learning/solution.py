import math
from collections import Counter

def calculate_entropy(labels: list) -> float:
    """Calculate the entropy of a list of labels."""
    n = len(labels)
    count_dict = Counter(labels)
    entropy = 0 
    for i, c in count_dict.items():
        entropy -= (c/n)*math.log(c/n, 2)
    return entropy

def calculate_information_gain(examples: list[dict], attr: str, target_attr: str) -> float:
    """Calculate the information gain of splitting on attr."""
    n_sample = len(examples)
    labels = [i[target_attr] for i in examples]
    parent_entropy = calculate_entropy(labels)
    att_val_count = Counter([i[attr] for i in examples])
    att_val_count = sorted(att_val_count.items())
    entropy_current = 0
    for val, count in att_val_count:
        child_labels = [i[target_attr]  for i in examples if i[attr] == val]
        child_entropy = calculate_entropy(child_labels)
        entropy_current += (count/n_sample)* child_entropy
    return parent_entropy - entropy_current

def majority_class(examples: list[dict], target_attr: str) -> str:
    """Return the majority class. Break ties alphabetically."""
    labels = [i[target_attr] for i in examples]
    return Counter(labels).most_common()[0][0]


def learn_decision_tree(examples: list[dict], attributes: list[str], target_attr: str) -> dict:
    """Build a decision tree using the ID3 algorithm."""
    best_i_gain = -1
    best_attr = None
    unique_labels = set([i[target_attr] for i in examples])

    if len(unique_labels) == 1:
        return list(unique_labels)[0]

    if len(attributes) == 0:
        return majority_class(examples, target_attr)
    
    for att in attributes:
        i_gain = calculate_information_gain(examples, att, target_attr)
        if i_gain > best_i_gain:
            best_i_gain = i_gain
            best_attr = att

    att_val_count = Counter([i[best_attr] for i in examples])
    att_val_count = sorted(att_val_count.items())
    entropy_current = 0
    attributes.remove(best_attr)
    decision_tree = {}
    for val, count in att_val_count:
        examples_child = [i  for i in examples if i[best_attr] == val]
        child_decision_tree = learn_decision_tree(examples_child, attributes.copy(), target_attr)
        # print(child_decision_tree, examples_child, attributes)
        decision_tree[val] = child_decision_tree
    final_decision_tree = {
        best_attr : decision_tree
    }
    return final_decision_tree
    
