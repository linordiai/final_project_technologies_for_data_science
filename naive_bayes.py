import math
from collections import defaultdict


class NaiveBayes:
    def fit(self, data, attributes, class_attr, attr_values):
        self.attributes = attributes
        self.attr_values = attr_values
        self.class_counts = defaultdict(int)
        self.feat_counts = {a: defaultdict(lambda: defaultdict(int)) for a in attributes}
        self.classes = []
        seen = set()

        for row in data:
            cls = row[class_attr]
            if cls not in seen:
                self.classes.append(cls)
                seen.add(cls)
            self.class_counts[cls] += 1
            for attr in attributes:
                self.feat_counts[attr][row[attr]][cls] += 1

        self.total = len(data)

    def predict(self, instance):
        best_cls, best_log = None, float('-inf')
        for cls in self.classes:
            log_p = math.log(self.class_counts[cls] / self.total)
            for attr in self.attributes:
                val = instance.get(attr)
                n_vals = len(self.attr_values[attr])
                count = self.feat_counts[attr][val][cls] + 1
                denom = self.class_counts[cls] + n_vals
                log_p += math.log(count / denom)
            if log_p > best_log:
                best_log, best_cls = log_p, cls
        return best_cls
