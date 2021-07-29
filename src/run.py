import os

codes = ['KNN', 'neural_network', 'logisticRegression', 'decisionTree']
if not os.path.exists('images'):
    os.makedirs('images')
for i in codes:
    print(f'-------------------{i}: -------------------')
    os.system(f'python3 {i}.py')
