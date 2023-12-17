from src.models.ModelBaseClass import ModelBaseClass

def ConsensusMethod(ModelBaseClass):

    def __init__(self, cardinality=2, verbose=True):
        # instantiate a DecisionTreeClassifier with Grid search
        # instantiate a WeightedMajorityVotingClassifier
        # instantiate a SVCSKLearnModel

        pass

    def _train(self, X_train, Y_train):
        # train consensus and stage 2
        '''
        Prompt pool already exists, but now this code 
        Train the prompt combinations 
        '''
        # train your consensus method
        # train your stage 2 model (Decision tree over all training data using the best set of prompts determined)
        # split train into train and val
        # train all methods on train
        # select the best method of these for val
        # retrain the best selected stage 2 method on both train and val
        '''
        For stage 2: 
        Step 1: split train into train_train, and train_val (70-30)
        Step 2: Using our X different methods for prompt selection, select different prompt pools.
            We have Devesh's method
            We have 2 other methods mrmr_classif, and rfe_svc
            Devesh's method - select 4 different prompt pools - size (some may not be possible with current experiment config) 15, 9, 5, 3 (remember that whatever the size is must be less than our current prompt pool for current experiment config)
            Alex's 2 methods - select best of size whatever so that a total of max 8 different selected, 4 for each method
        Step 3:
            total number of models to train here is n_models * n_prompt_pools, this is a lot of training, but these models train extremely quickly. 
            For each prompt pool-model combination, train a model on train_train, and test on train_val
            store the validation accuracies in some data structure
        Step 4: Select the best model-prompt pool combination determined by balanced_accuracy on train_val 
        Step 5: retrain best model-prompt pool combination on all of train (train_train and train_val) and make this our stage 2 model used in predict
        '''
        raise NotImplementedError()

    def predict(self, X):
        # predict consensus and stage 2
        raise NotImplementedError()
    
    def report_trained_parameters(self):
        # Report prompt combinations, selected stage 2 model parameters
        raise NotImplementedError()
