import os
import sys
import decisionTree
import KNN
import logisticRegression
import neural_network
import EDA
import main as m

ip = 'Invalid #parameter'
isf = 'Invalid shortcut for <function>'


if __name__ == "__main__":
    if not os.path.exists('images'):
        os.makedirs('images')
    if not os.path.exists('eda'):
        os.makedirs('eda')
    
    if (len(sys.argv) == 1):
        print('Usage: python3 run.py <file> <function> <parameters>')
        print('Shortcut for <file> as follow:')
        print('> main: Run the main file\n> knn: KNN functions\n> tree: DecisionTree functions\n> log: LogisticRegression functions\n> net: Run NeuralNetwork\n> eda: Exploratory-Data-Analysis for Santander Bank')
    elif (len(sys.argv) == 2):
        if sys.argv[1] == 'main':
            m.main()
        elif sys.argv[1] == 'knn':
            print('Shortcut for <function> as follow:\n> find_best_k_value\n> train_KNN_model\n> K_value_tuning <X_validation> <Y_validation>')
        elif sys.argv[1] == 'tree':
            print('Shortcut for <function> as follow:\n> train_decisionTree_model\n> find_best_depth\n> test_decisionTree\n> depth_tuning <X_validation> <Y_validation>')
        elif sys.argv[1] == 'log':
            print('Shortcut for <function> as follow:\n> train_lgr_model\n> find_best_solver\n> find_best_C\n> c_value_tuning <X_validation> <Y_validation>')
        elif sys.argv[1] == 'net':
            neural_network.main()
        elif sys.argv[1] == 'eda':
            EDA.main()
        else:
            print('Invalid shortcut for <file>')
    elif (len(sys.argv) > 2):
        if sys.argv[1] == 'knn':
            if sys.argv[2] == 'find_best_k_value':
                KNN.find_best_k_value()
            elif sys.argv[2] == 'train_KNN_model':
                KNN.train_KNN_model()
            elif sys.argv[2] == 'K_value_tuning':
                if len(sys.argv) > 4:
                    KNN.K_value_tuning(sys.argv[3], sys.argv[4])
                else:
                    print(ip)
            else:
                print(isf)
        if sys.argv[1] == 'tree':
            if sys.argv[2] == 'train_decisionTree_model':
                decisionTree.train_decisionTree_model()
            elif sys.argv[2] == 'find_best_depth':
                decisionTree.find_best_depth()
            elif sys.argv[2] == 'depth_tuning':
                if len(sys.argv) > 4:
                    decisionTree.depth_tuning(sys.argv[3], sys.argv[4])
                else:
                    print(ip)
            else:
                print(isf)
        if sys.argv[1] == 'log':
            if sys.argv[2] == 'train_lgr_model':
                logisticRegression.train_lgr_model()
            elif sys.argv[2] == 'find_best_solver':
                logisticRegression.find_best_solver()
            elif sys.argv[2] == 'find_best_C':
                logisticRegression.find_best_C()
            elif sys.argv[2] == 'c_value_tuning':
                if len(sys.argv) > 4:
                    logisticRegression.c_value_tuning(sys.argv[3], sys.argv[4])
                else:
                    print(ip)
            else:
                print(isf)
        else:
            print('Invalid shortcut')
