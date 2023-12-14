from src.models.ModelBaseClass import ModelBaseClass
from src.models.WRENCHModels.DawidSkeneModel import DawidSkeneModel
from src.models.AlexEnsemblingMethods.ConditionalLR import ConditionalLR
from src.models.WRENCHModels.SnorkelModels import SnorkelLabelModel, SnorkelMajorityLabelVoter
from src.models.LGBModels import LGBMSKLearnModel, LGBMSKLearnModel2, LGBMSKLearnModel3
from src.models.SKLearnModels.SkLearnModels import instantiate_sk_learn_models
from src.models.PyTorchModels.PyTorchModels import load_pytorch_models

def load_models(use_dawid_skene_models=True, use_sklearn_models=True, use_alex_models=True, use_snorkel_models=True, use_lgb_models = True, use_pytorch_models = True):
    """
    Returns a list of models to be used in the benchmarking process.
    """
    models_in_use = []
    if use_dawid_skene_models:
        models_in_use.append(DawidSkeneModel()) 
    if use_sklearn_models:
        models_in_use.extend(instantiate_sk_learn_models())
    if use_alex_models:
        models_in_use.append(ConditionalLR())
    if use_snorkel_models:
        models_in_use.extend([SnorkelLabelModel(), SnorkelMajorityLabelVoter()])
    if use_lgb_models:
        models_in_use.extend([LGBMSKLearnModel3(), LGBMSKLearnModel(), LGBMSKLearnModel2()])
    if use_pytorch_models:
        pytorch_models = load_pytorch_models()
        models_in_use.extend(pytorch_models)

    return models_in_use