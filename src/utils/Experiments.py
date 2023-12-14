from enum import Enum, auto
from typing import List, Union, Optional, Set

class DatasetOrigin(Enum):
    AGGREFACCT_SOTA_XSUM_TEST = "AGGREFACCT_SOTA_XSUM_TEST"  # TODO: Change these to what they are in devesh's code
    AGGREFACCT_SOTA_XSUM_VAL = "AGGREFACCT_SOTA_XSUM_VAL"
    AGGREFACCT_SOTA_CNN_DM_TEST = "AGGREFACCT_SOTA_CNN_DM_TEST"
    AGGREFACCT_SOTA_CNN_DM_VAL = "AGGREFACCT_SOTA_CNN_DM_VAL"
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
    # Define the prompt column sets
    # TODO: We need to pass in our columns here!
    prompt_columns_all_ours = ['Devesh_Prompt_1 Alex Output Modiciation',
       'Detailed Inferential Analysis',
       'Contextual Analysis with Detailed Guidance',
       'Specific Details and Nuances',
       'Devesh_Prompt_1.1 Alex Output Modiciation 2 ', 
       'Devesh Prompt 3',
       'HALU-EVAl Like Prompt',
       'Contradiction Highlighted Inferential Analysis',
       'Strict factual consistency analysis Variant 1 - Targeted Error Detection',
       'NLI_GPT_PROMPT1119',
       'Comprehensive factual consistency analysis Variant 3.2',
       'Extraction NLI', 
       ' Strict factual consistency analysis Variant 3',
       'Contextual Analysis with Focus on Contradictory Information',
       'Luo 23 Zero Shot CoT FactCC',
       'Comprehensive factual consistency analysis Variant 3.1',
       'Strict factual consistency analysis', 
       'GPT_improved_nli_style_prompt',
       'Strict factual consistency analysis Variant 3 - Strict 2',
       'Comprehensive factual consistency analysis Variant 3.2 - Strict 7',
       'Comprehensive factual consistency analysis Variant 3.2 - Strict 8',]
    
    from data.imported.datasets.devesh_top_prompts import devesh_selected_prompt_combinations
    desired_prompt_sets = devesh_selected_prompt_combinations
    # # TODO:: Have method that selects best n prompts
    # prompt_columns_top_3_var_1 = prompt_columns_all_ours[0:3] # TODO: add code to select the best 3 promtps to use, ["col1", "col2", "col3"]
    # prompt_columns_top_3_var_2 = prompt_columns_all_ours[0:3] # TODO: add different variations of the top_n prompts based on our selection criteria (bal. acc, TPFP/TNFN ratios, variance, etc.)
    # prompt_columns_top_5 = prompt_columns_all_ours[0:5]  # ["col1", "col2", "col3", "col4", "col5"]
    # prompt_columns_top_9 = prompt_columns_all_ours[0:9] #  ["col1", "col2", "col3", "col4", "col5", "col6", "col7", "col8", "col9"]
    # prompt_columns_top_15 = prompt_columns_all_ours[0:15]#   ["col1", "col2", "col3", "col4", "col5", "col6", "col7", "col8", "col9", "col10", "col11", "col12", "col13", "col14", "col15"]

    # desired_prompt_sets = [prompt_columns_all_ours, prompt_columns_top_3_var_1, prompt_columns_top_5, prompt_columns_top_9, prompt_columns_top_15]

    # Experiment configurations
    train_test_combinations = [
        ([DatasetOrigin.AGGREFACCT_SOTA_XSUM_VAL, DatasetOrigin.AGGREFACCT_SOTA_CNN_DM_VAL], [DatasetOrigin.AGGREFACCT_SOTA_CNN_DM_TEST, DatasetOrigin.AGGREFACCT_SOTA_XSUM_TEST]),
        ([DatasetOrigin.AGGREFACCT_SOTA_XSUM_VAL], [DatasetOrigin.AGGREFACCT_SOTA_XSUM_TEST]),
        ([DatasetOrigin.AGGREFACCT_SOTA_CNN_DM_VAL], [DatasetOrigin.AGGREFACCT_SOTA_XSUM_TEST]),
        ([DatasetOrigin.AGGREFACCT_SOTA_XSUM_VAL, DatasetOrigin.AGGREFACCT_SOTA_CNN_DM_VAL], [DatasetOrigin.HALU_EVAL_SUMMARIZATION]),
        ([DatasetOrigin.AGGREFACCT_SOTA_XSUM_TEST, DatasetOrigin.AGGREFACCT_SOTA_CNN_DM_TEST], [DatasetOrigin.HALU_EVAL_SUMMARIZATION])
    ]
    train_test_combinations_values = [([origin.value for origin in train_set], [origin.value for origin in test_set]) for train_set, test_set in train_test_combinations]

    # Create configurations
    configs = []
    for train_set, test_set in train_test_combinations_values:
        for prompt_set in desired_prompt_sets:
            configs.append(Experiment(train_origin=train_set, test_origin=test_set, skip_rows_with_null_values=True, prompt_columns_in_use=prompt_set))

    return configs
