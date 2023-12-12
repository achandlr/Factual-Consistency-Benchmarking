from src.models.ModelBaseClass import ModelBaseClass


def ConsensusMethod(ModelBaseClass):

    def __init__(self, cardinality=2, verbose=True):
        pass

    def train(self, L_train, Y_dev=None, n_epochs=500, log_freq=100, seed=123):
        # train consensus and stage 2
        '''
        Prompt pool already exists, but now this code 
        Train the prompt combinations 
        '''
        raise NotImplementedError()

    def predict(self, L):
        # predict consensus and stage 2
        raise NotImplementedError()
    
    def report_trained_parameters(self):
        # Report prompt combinations, selected stage 2 model parameters
        raise NotImplementedError()
