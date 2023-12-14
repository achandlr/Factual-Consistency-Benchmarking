'''
This code is designed to select the top_n functions (aka LLM prompts) to use for a given experiment.

It is a naive way of deciding which functions are a best, where we experimently calculate ensembling scores for different weight combinations .
'''


# Based on https://towardsdatascience.com/understanding-the-importance-of-diversity-in-ensemble-learning-34fb58fd2ed0
class VarianceStatistics:
    def __init__(self):
        pass

    @staticmethod
    def coefficients(preds):
        A = np.asarray(preds[:, 0], dtype=bool)
        B = np.asarray(preds[:, 1], dtype=bool)

        a = np.sum(A * B)           # A right, B right
        b = np.sum(~A * B)          # A wrong, B right
        c = np.sum(A * ~B)          # A right, B wrong
        d = np.sum(~A * ~B)         # A wrong, B wrong

        return a, b, c, d

    @staticmethod
    def disagreement(preds, i,j):
        L = preds.shape[1]
        a, b, c, d = VarianceStatistics.coefficients(preds[:, [i, j]])
        disagreement =  float(b + c) / (a + b + c + d)
        return disagreement

    @staticmethod
    def paired_q(preds, i, j):
        L = preds.shape[1]
        # div = np.zeros((L * (L - 1)) // 2)
        a, b, c, d = VarianceStatistics.coefficients(preds[:, [i, j]])
        paired_q =  float(a * d - b * c) / ((a * d + b * c) + 10e-24) 
        return paired_q
    
    @staticmethod
    def entropy(selected_functions, function_in_consideration):
        preds = np.column_stack((selected_functions, function_in_consideration))
        L = preds.shape[1]
        tmp = np.sum(preds, axis=1)
        tmp = np.minimum(tmp, L - tmp)
        ent = np.mean((1.0 / (L - np.ceil(0.5 * L))) * tmp)
        return ent
    # , selected_functions, function_in_consideration, ground_truth)




import pandas as pd
import numpy as np
import random
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from deap import creator, base, tools, algorithms
import sys


def avg(l):
    """
    Returns the average between list elements
    """
    return (sum(l)/float(len(l)))


def getFitness(individual, X, y):
    """
    Feature subset fitness function
    """

    if(individual.count(0) != len(individual)):
        # get index with value 0
        cols = [index for index in range(
            len(individual)) if individual[index] == 0]

        # get features subset
        X_parsed = X.drop(X.columns[cols], axis=1)
        X_subset = pd.get_dummies(X_parsed)

        # apply classification algorithm
        clf = LogisticRegression()

        return (avg(cross_val_score(clf, X_subset, y, cv=5)),)
    else:
        return(0,)


def geneticAlgorithm(X, y, n_population, n_generation):
    """
    Deap global variables
    Initialize variables to use eaSimple
    """
    # create individual
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    # create toolbox
    toolbox = base.Toolbox()
    toolbox.register("attr_bool", random.randint, 0, 1)
    toolbox.register("individual", tools.initRepeat,
                     creator.Individual, toolbox.attr_bool, len(X.columns))
    toolbox.register("population", tools.initRepeat, list,
                     toolbox.individual)
    toolbox.register("evaluate", getFitness, X=X, y=y)
    toolbox.register("mate", tools.cxOnePoint)
    toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
    toolbox.register("select", tools.selTournament, tournsize=3)

    # initialize parameters
    pop = toolbox.population(n=n_population)
    hof = tools.HallOfFame(n_population * n_generation)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("min", np.min)
    stats.register("max", np.max)

    # genetic algorithm
    pop, log = algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2,
                                   ngen=n_generation, stats=stats, halloffame=hof,
                                   verbose=True)

    # return hall of fame
    return hof


def bestIndividual(hof, X, y):
    """
    Get the best individual
    """
    maxAccurcy = 0.0
    for individual in hof:
        if(individual.fitness.values > maxAccurcy):
            maxAccurcy = individual.fitness.values
            _individual = individual

    _individualHeader = [list(X)[i] for i in range(
        len(_individual)) if _individual[i] == 1]
    return _individual.fitness.values, _individual, _individualHeader

'''
https://github.com/renatoosousa/GeneticAlgorithmForFeatureSelection/blob/master/gaFeatureSelection.py
'''
def getArguments():
    """
    Get argumments from command-line
    If pass only dataframe path, pop and gen will be default
    """
    dfPath = sys.argv[1]
    if(len(sys.argv) == 4):
        pop = int(sys.argv[2])
        gen = int(sys.argv[3])
    else:
        pop = 10
        gen = 2
    return dfPath, pop, gen
if __name__ == '__main__':
    # get dataframe path, population number and generation number from command-line argument
    dataframePath, n_pop, n_gen = getArguments()
    # read dataframe from csv
    df = pd.read_csv(dataframePath, sep=',')

    # encode labels column to numbers
    le = LabelEncoder()
    le.fit(df.iloc[:, -1])
    y = le.transform(df.iloc[:, -1])
    X = df.iloc[:, :-1]

    # get accuracy with all features
    individual = [1 for i in range(len(X.columns))]
    print("Accuracy with all features: \t" +
          str(getFitness(individual, X, y)) + "\n")

    # apply genetic algorithm
    hof = geneticAlgorithm(X, y, n_pop, n_gen)

    # select the best individual
    accuracy, individual, header = bestIndividual(hof, X, y)
    print('Best Accuracy: \t' + str(accuracy))
    print('Number of Features in Subset: \t' + str(individual.count(1)))
    print('Individual: \t\t' + str(individual))
    print('Feature Subset\t: ' + str(header))

    print('\n\ncreating a new classifier with the result')

    # read dataframe from csv one more time
    df = pd.read_csv(dataframePath, sep=',')

    # with feature subset
    X = df[header]

    clf = LogisticRegression()

    scores = cross_val_score(clf, X, y, cv=5)
    print("Accuracy with Feature Subset: \t" + str(avg(scores)) + "\n")
    
import numpy as np
from sklearn.metrics import balanced_accuracy_score, confusion_matrix

def select_best_functions(scores, gold_values, n, weights):
    # Weights format: {'balanced_accuracy': w1, 'variance_added': w2, 'TPFP_ratio': w3, 'TNFN_ratio': w4}

    def calculate_balanced_accuracy(predictions, gold):
        # Calculate balanced accuracy using sklearn
        return balanced_accuracy_score(gold, predictions)

    def calculate_variance_added(selected_functions, candidate_function, gold):
        # Implement Kohavi's variance added calculation considering the selected functions and the candidate
        pass  # Placeholder for the actual implementation


    def calculate_TPFP_ratio(predictions, gold):
        # Calculate TP/FP ratio using confusion matrix
        tn, fp, fn, tp = confusion_matrix(gold, predictions).ravel()
        return tp / fp if fp != 0 else 0  # Avoid division by zero

    def calculate_TNFN_ratio(predictions, gold):
        # Calculate TN/FN ratio using confusion matrix
        tn, fp, fn, tp = confusion_matrix(gold, predictions).ravel()
        return tn / fn if fn != 0 else 0  # Avoid division by zero

    selected_functions = []
    while len(selected_functions) < n:
        best_score = -np.inf
        best_function_index = -1

        for i in range(scores.shape[1]):
            if i not in selected_functions:
                predictions = scores[:, i]
                score = weights['balanced_accuracy'] * calculate_balanced_accuracy(predictions, gold_values)
                if len(selected_functions) > 0:
                    score += weights['variance_added'] * calculate_variance_added(selected_functions, i, gold_values)
                score += weights['TPFP_ratio'] * calculate_TPFP_ratio(predictions, gold_values)
                score += weights['TNFN_ratio'] * calculate_TNFN_ratio(predictions, gold_values)

                if score > best_score:
                    best_score = score
                    best_function_index = i

        selected_functions.append(best_function_index)

    return selected_functions

# Example usage
# scores = np.array([[...]]) # 2D array of scores
# gold_values = np.array([...]) # Array of gold binary values
# n = 5 # Number of best functions to select
# weights = {'balanced_accuracy': 1, 'variance_added': 1, 'TPFP_ratio': 1, 'TNFN_ratio':
