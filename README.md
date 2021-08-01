# customer_satisfaction_machine_learning
# Dependencies
1) Python3 and pip3 are required to run the code.
2) Install the dependencies by typing 'pip3 install -r requirements.txt'.

## Jupyter notebooks
because we are using deep learning we included  jupyter notebooks that have cached results for Neural Network, KNN, Logistic regression, DecisionTree and the main jupyter notebook that essemble different models.

## What are the different files and what they do
1) There are seperate files and cached jupyter notebooks for each model which are called: 
   * neural_network.py
   * KNN.py
   * decisionTree.py
   * logisticRegression.py 
   * neural_network.ipynb 
   * decisionTree.ipynb
   * logisticRegression.ipynb
   * KNN.ipynb 
2) There is a file called main.py. This file trains all of the models and fit them based on the hyperparameters that we found running the model files.

## How to run 

1) first change to src folder 
> cd src
2) First run data_preprocess.py
> python3 data_preprocess.py  
3) run main.py
> python3 main.py 
4) to run the model files seperately you can run 
> python3 neural_network.py

> python3 decisionTree.py   

> python3 logisticRegression.py



