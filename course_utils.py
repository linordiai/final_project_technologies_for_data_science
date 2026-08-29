from collections import defaultdict


def read_data(filename):
    data, headers = [], None
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split('\t')
            if headers is None:
                headers = parts
            else:
                data.append(dict(zip(headers, parts)))
    return headers, data


def get_attr_values(data, attributes):
    result = {}
    for attr in attributes:
        seen, visited = [], set()
        for row in data:
            v = row[attr]
            if v not in visited:
                seen.append(v)
                visited.add(v)
        result[attr] = seen
    return result


def majority_class(data, class_attr):
    counts = defaultdict(int)
    for row in data:
        counts[row[class_attr]] += 1
    return max(counts, key=lambda k: (counts[k], k))
