import os

codes = ['KNN', 'neural_network', 'logisticRegression', 'decisionTree']
if not os.path.exists('images'):
    os.makedirs('images')
for i in codes:
    os.system(f'python3 {i}.py')