import numpy as np

from judgenet.utils.file import write_json


class Results():
    
    def __init__(self):
        self.results = {}

    def update_results(self, exp_type, stats):
        if exp_type not in self.results:
            self.results[exp_type] = {}
        for stat_name, stat_val in stats.items():
            if stat_name not in self.results[exp_type]:
                self.results[exp_type][stat_name] = []
            self.results[exp_type][stat_name].append(stat_val)

    def finalize_results(self, path):
        orig_keys = list(self.results.keys())
        for exp_type in orig_keys:
            self.results[f"{exp_type}_avg"] = {}
            for stat_name, stat_vals in self.results[exp_type].items():
                self.results[f"{exp_type}_avg"][stat_name] = np.mean(stat_vals)
        write_json(self.results, path)
