import os
import time

class UIPRMDResults:
    def __init__(self, experiment_type, base_dir='.'):
        self.base_dir = base_dir
        self.results_dict = {}
        for i in range(1, 6, 1):
            exercise_id = 'Es'+str(i)
            self.results_dict[exercise_id] = []
        self.ts = str(int(time.time()))
        self.write_path = None
        self.write_path = os.path.join(self.base_dir, 'results_' + experiment_type + '_' + self.ts + '.txt')


    def write_results_exercise(self, exercise_id):
        with open(self.write_path, 'a') as f:
            f.write('######################################################\n')
            for MAE, RMSE in self.results_dict[exercise_id]:
                results_str = 'MAE = '+str(MAE)+' RMSE = '+str(RMSE)+'\n'
                f.write(results_str)
            f.write('######################################################\n')


    def write_embeddor_type(self, embeddor_type):
        with open(self.write_path, 'a') as f:
            f.write(embeddor_type+'\n')


# results = UIPRMDResults()
# results.results_dict['e02'].append((10, 20))
# results.results_dict['e02'].append((1, 2))
# # lst = results.results_dict.get('e02')
# # lst.append((10, 20))
# # results.results_dict['e02'] = lst
# print(results.results_dict['e02'])