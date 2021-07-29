import os
import sys

if __name__ == "__main__":
    codes = ['KNN', 'neural_network', 'logisticRegression', 'decisionTree']
    if not os.path.exists('images'):
        os.makedirs('images')
    if (len(sys.argv) != 1):
        if (sys.argv.count('r') == 0): codes.remove('logisticRegression')
        if (sys.argv.count('k') == 0): codes.remove('KNN')
        if (sys.argv.count('t') == 0): codes.remove('decisionTree')
        if (sys.argv.count('n') == 0): codes.remove('neural_network')
    for i in codes:
        print(f'-------------------{i}: -------------------')
        os.system(f'python3 {i}.py')
