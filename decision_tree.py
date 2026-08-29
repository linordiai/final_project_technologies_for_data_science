import math
from collections import defaultdict
from course_utils import majority_class


def entropy(data, class_attr):
    counts = defaultdict(int)
    for row in data:
        counts[row[class_attr]] += 1
    total = len(data)
    if total == 0:
        return 0.0
    return -sum((c / total) * math.log2(c / total) for c in counts.values() if c > 0)


def information_gain(data, attr, class_attr):
    total = len(data)
    if total == 0:
        return 0.0
    partitions = defaultdict(list)
    for row in data:
        partitions[row[attr]].append(row)
    weighted = sum(len(sub) / total * entropy(sub, class_attr) for sub in partitions.values())
    return entropy(data, class_attr) - weighted


def id3(data, attributes, class_attr, attr_values):
    unique_classes = {row[class_attr] for row in data}

    if len(unique_classes) == 1:
        return {'leaf': True, 'class': unique_classes.pop()}
    if not attributes or not data:
        return {'leaf': True, 'class': majority_class(data, class_attr)}

    gains = {a: information_gain(data, a, class_attr) for a in attributes}
    if max(gains.values()) == 0:
        return {'leaf': True, 'class': majority_class(data, class_attr)}

    best = max(gains, key=gains.get)
    remaining = [a for a in attributes if a != best]
    children = {}

    for val in attr_values[best]:
        subset = [row for row in data if row[best] == val]
        if subset:
            children[val] = id3(subset, remaining, class_attr, attr_values)
        else:
            children[val] = {'leaf': True, 'class': majority_class(data, class_attr)}

    return {'leaf': False, 'attr': best, 'values': attr_values[best], 'children': children}


def predict(tree, instance, default):
    if tree['leaf']:
        return tree['class']
    child = tree['children'].get(instance.get(tree['attr']))
    return predict(child, instance, default) if child else default


def write_tree(tree, filepath):
    with open(filepath, 'w') as f:
        _write_node(tree, f, depth=0)


def _write_node(node, f, depth):
    if node['leaf']:
        return
    attr = node['attr']
    for val in node['values']:
        child = node['children'][val]
        prefix = '\t' * depth + ('|' if depth > 0 else '')
        if child['leaf']:
            f.write(f'{prefix}{attr}={val}:{child["class"]}\n')
        else:
            f.write(f'{prefix}{attr}={val}\n')
            _write_node(child, f, depth + 1)
