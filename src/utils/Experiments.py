from enum import Enum, auto
from typing import List, Union, Optional, Set

class DatasetOrigin(Enum):
    AGGREFACCT_SOTA_XSUM_TEST = "AGGREFACCT_SOTA_XSUM_TEST" 
    AGGREFACCT_SOTA_XSUM_VAL = "AGGREFACCT_SOTA_XSUM_VAL"
    AGGREFACCT_SOTA_CNN_DM_TEST = "AGGREFACCT_SOTA_CNN_DM_TEST"
    AGGREFACCT_SOTA_CNN_DM_VAL = "AGGREFACCT_SOTA_CNN_DM_VAL"
    HALU_EVAL_SUMMARIZATION = "HALU_EVAL_SUMMARIZATION"


class DataTypeOrigin(Enum):
    Confident_Train     = "Confident_Train"  
    Unsure_Train        = "Unsure_Train"
    Confident_Test      = "Confident_Test"
    Unsure_Test         = "Unsure_Test"
    HALU_EVAL_SUMMARIZATION = "HALU_EVAL_SUMMARIZATION"


class Experiment:
    def __init__(self, 
                 train_origin: Union[DatasetOrigin, List[DatasetOrigin]], 
                 test_origin: Union[DatasetOrigin, List[DatasetOrigin]], 
                 skip_rows_with_null_values: bool = True, 
                 prompt_columns_in_use: Optional[List[str]] = None):
        """
        Initializes an Experiment instance.
        :param train_origin: DatasetOrigin or list of DatasetOrigins for training data.
        :param test_origin: DatasetOrigin or list of DatasetOrigins for testing data.
        :param skip_rows_with_null_values: Boolean indicating whether to skip rows with null values.
        :param prompt_columns_in_use: List of columns to use as prompts. Defaults to ["prompt_1", "prompt_2"].
        """
        self.train_origin = self._ensure_dataset_origin_list(train_origin)
        self.test_origin = self._ensure_dataset_origin_list(test_origin)
        self.skip_rows_with_null_values = skip_rows_with_null_values
        self.prompt_columns_in_use = prompt_columns_in_use or ["prompt_1", "prompt_2"]
        self.verify_configuration()

    def _ensure_dataset_origin_list(self, origins: Union[DatasetOrigin, List[DatasetOrigin]]) -> List[DatasetOrigin]:
        return list(origins) if isinstance(origins, list) else [origins]

    def verify_configuration(self):
        """
        Verifies that the training and testing datasets do not overlap.
        Raises ValueError if there is an intersection.
        """
        train_set = set(self.train_origin)
        test_set = set(self.test_origin)

        if train_set.intersection(test_set):
            raise ValueError("Intersection of train_origin and test_origin should be empty.")



def load_experiment_configs() -> List[Experiment]:

    from data.imported.devesh_top_prompts import devesh_selected_prompt_combinations
    from data.imported.alex_top_prompts import best_rfe_prompt_sets,best_mrmr_prompt_sets 

    desired_prompt_sets = devesh_selected_prompt_combinations + best_rfe_prompt_sets + best_mrmr_prompt_sets
    # Experiment configurations
    train_test_combinations = [
        ([DatasetOrigin.AGGREFACCT_SOTA_XSUM_VAL, DatasetOrigin.AGGREFACCT_SOTA_CNN_DM_VAL], [DatasetOrigin.AGGREFACCT_SOTA_CNN_DM_TEST, DatasetOrigin.AGGREFACCT_SOTA_XSUM_TEST]),
        ([DatasetOrigin.AGGREFACCT_SOTA_XSUM_VAL], [DatasetOrigin.AGGREFACCT_SOTA_XSUM_TEST]),
        ([DatasetOrigin.AGGREFACCT_SOTA_CNN_DM_VAL], [DatasetOrigin.AGGREFACCT_SOTA_XSUM_TEST]),
        ([DatasetOrigin.AGGREFACCT_SOTA_XSUM_VAL, DatasetOrigin.AGGREFACCT_SOTA_CNN_DM_VAL], [DatasetOrigin.HALU_EVAL_SUMMARIZATION]),
        ([DatasetOrigin.AGGREFACCT_SOTA_XSUM_TEST, DatasetOrigin.AGGREFACCT_SOTA_CNN_DM_TEST], [DatasetOrigin.HALU_EVAL_SUMMARIZATION]),
        ([DatasetOrigin.AGGREFACCT_SOTA_XSUM_TEST, DatasetOrigin.AGGREFACCT_SOTA_CNN_DM_TEST, DatasetOrigin.AGGREFACCT_SOTA_XSUM_VAL, DatasetOrigin.AGGREFACCT_SOTA_CNN_DM_VAL], [DatasetOrigin.HALU_EVAL_SUMMARIZATION])
    ]

    # Note: This config was used to determine the best stage 2 consensus method
    # train_test_combinations = [
    #     ([DataTypeOrigin.Confident_Train, DataTypeOrigin.Unsure_Train], [DataTypeOrigin.Unsure_Test]),
    #     ([ DataTypeOrigin.Unsure_Train], [DataTypeOrigin.Unsure_Test]),
    #     ([DataTypeOrigin.Confident_Train], [DataTypeOrigin.Unsure_Test])
    # ]

    train_test_combinations_values = [([origin.value for origin in train_set], [origin.value for origin in test_set]) for train_set, test_set in train_test_combinations]

    # Create configurations
    configs = []
    for train_set, test_set in train_test_combinations_values:
        for prompt_set in desired_prompt_sets:
            configs.append(Experiment(train_origin=train_set, test_origin=test_set, skip_rows_with_null_values=True, prompt_columns_in_use=prompt_set))

    return configs

def load_config_for_feature_selection():


    # got from df.columns and manually removed the ones that are not prompts
    desired_prompt_sets = [
        'Devesh_Prompt_1 Alex Output Modiciation',
        'Detailed Inferential Analysis',
        'Contextual Analysis with Detailed Guidance',
        'Specific Details and Nuances',
        'Devesh_Prompt_1.1 Alex Output Modiciation 2 ', 'Devesh Prompt 3',
        'HALU-EVAl Like Prompt',
        'Contradiction Highlighted Inferential Analysis',
        'Strict factual consistency analysis Variant 1 - Targeted Error Detection',
        'NLI_GPT_PROMPT1119',
        'Comprehensive factual consistency analysis Variant 3.2',
        'Extraction NLI', ' Strict factual consistency analysis Variant 3',
        'Contextual Analysis with Focus on Contradictory Information',
        'Luo 23 Zero Shot CoT FactCC',
        'Comprehensive factual consistency analysis Variant 3.1',
        'Strict factual consistency analysis', 'GPT_improved_nli_style_prompt',
        'Strict factual consistency analysis Variant 3 - Strict 2',
        'Comprehensive factual consistency analysis Variant 3.2 - Strict 7',
        'Comprehensive factual consistency analysis Variant 3.2 - Strict 8'],

    train_test_combinations = [
        # ([DatasetOrigin.AGGREFACCT_SOTA_XSUM_VAL, DatasetOrigin.AGGREFACCT_SOTA_CNN_DM_VAL], [DatasetOrigin.AGGREFACCT_SOTA_CNN_DM_TEST, DatasetOrigin.AGGREFACCT_SOTA_XSUM_TEST]),
        # ([DatasetOrigin.AGGREFACCT_SOTA_XSUM_VAL], [DatasetOrigin.AGGREFACCT_SOTA_XSUM_TEST]),
        # ([DatasetOrigin.AGGREFACCT_SOTA_CNN_DM_VAL], [DatasetOrigin.AGGREFACCT_SOTA_XSUM_TEST]),
        # ([DatasetOrigin.AGGREFACCT_SOTA_XSUM_VAL, DatasetOrigin.AGGREFACCT_SOTA_CNN_DM_VAL], [DatasetOrigin.HALU_EVAL_SUMMARIZATION]),
        ([DatasetOrigin.AGGREFACCT_SOTA_XSUM_TEST, DatasetOrigin.AGGREFACCT_SOTA_CNN_DM_TEST], [DatasetOrigin.HALU_EVAL_SUMMARIZATION])
    ]

    train_test_combinations_values = [([origin.value for origin in train_set], [origin.value for origin in test_set]) for train_set, test_set in train_test_combinations]

    # Create configurations
    configs = []
    for train_set, test_set in train_test_combinations_values:
        for prompt_set in desired_prompt_sets:
            configs.append(Experiment(train_origin=train_set, test_origin=test_set, skip_rows_with_null_values=True, prompt_columns_in_use=prompt_set))

    assert len(configs) == 1

    return configs[0]
    # flatten the devesh_selected_prompt_combinations to just be a pool of prompts:


