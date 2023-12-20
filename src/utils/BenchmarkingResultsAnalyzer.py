import pandas as pd
from src.utils.Experiments import DatasetOrigin


def get_dataset_to_method_to_prompt_size_to_acc(results_df_path):
    # Step 1: Read in the file
    df = pd.read_pickle(file_path)
    df['TestOriginTuple'] = df["TestOrigin"].apply(lambda x: tuple(x))
    df['TrainOriginTuple'] = df["TrainOrigin"].apply(lambda x: tuple(x))
    df['num_prompts'] = df["PromptColumnsInUse"].apply(lambda x: len(x))

    train_tuple = tuple([DatasetOrigin.AGGREFACCT_SOTA_XSUM_VAL.name, DatasetOrigin.AGGREFACCT_SOTA_CNN_DM_VAL.name])
    dataset_name_to_test_tuple = {"XSUM": tuple([DatasetOrigin.AGGREFACCT_SOTA_XSUM_TEST.name]), "CNN_DM": tuple([DatasetOrigin.AGGREFACCT_SOTA_CNN_DM_TEST.name]), "HaluEval": tuple([DatasetOrigin.HALU_EVAL_SUMMARIZATION.name])}

    df_with_desired_training =  df[df["TrainOriginTuple"] == train_tuple]
    test_dataset_name_to_df_for_that_test_df = {dataset_name: df_with_desired_training[df_with_desired_training["TestOriginTuple"] == test_tuple] for dataset_name, test_tuple in dataset_name_to_test_tuple.items()}
    

    xsum_eval_results_df = test_dataset_name_to_df_for_that_test_df["XSUM"]
    cnn_dm_eval_results_df = test_dataset_name_to_df_for_that_test_df["CNN_DM"]
    halu_eval_results_df = test_dataset_name_to_df_for_that_test_df["HaluEval"]

    desired_prompt_sizes = [3, 5, 9, 15]
    models_of_interest = ['DawidSkeneModel', 'WeightedMajorityVotingClassifier',
 'RandomForestSKLearnModel', 'GradientBoostingSKLearnModel',
 'AdaBoostSKLearnModel', 
 'LogisticRegressionSKLearnModel', 'SVCSKLearnModel',
 'DecisionTreeSKLearnModel', 'GaussianNBSKLearnModel',
 'MultinomialNBSKLearnModel' , 'BernoulliNBSKLearnModel', 
 'KNeighborsSKLearnModel' ,'LDASKLearnModel' ,'CatBoostSKLearnModel',
 'XGBSKLearnModel' ,'ConditionalLR' ,'SnorkelMajorityLabelVoter',
 'SnorkelLabelModel' , 'LGBMSKLearnModel3', 'LGBMSKLearnModel',
 'LGBMSKLearnModel2'] # Removed 'DummySKLearnModel', TODO: remove other models that have bad results or we don't want in chart. Select 6-10 best models and make them the ones in our chart. Have full results in the appendix
    dataset_stats = {}
    for test_dataset_name, df_for_that_test_dataset in test_dataset_name_to_df_for_that_test_df.items():
        unique_models = halu_eval_results_df["Model"].unique()
        model_stats = {}
        for model in unique_models:
            if model not in models_of_interest:
                continue
            df_for_that_test_dataset_and_model = df_for_that_test_dataset[df_for_that_test_dataset["Model"] == model]
            prompt_size_to_acc = {}
            for prompt_size in desired_prompt_sizes:
                df_for_that_test_dataset_and_model_and_prompt_size = df_for_that_test_dataset_and_model[df_for_that_test_dataset_and_model["num_prompts"] == prompt_size]
                # Don't do mean, rather pick a certain or best one
                acc = df_for_that_test_dataset_and_model_and_prompt_size["balanced_accuracy"].max()
                prompt_size_to_acc[prompt_size] = acc
            model_stats[model] = prompt_size_to_acc
        dataset_stats[test_dataset_name] = model_stats

    raise dataset_stats


def process_and_display_data(file_path, column_X, column_Y, model_column):
    # Step 1: Read in the file
    df = pd.read_pickle(file_path)
    

    # Converting lists in 'column_X' to tuples
    df[f'{column_X}_Tuple'] = df[column_X].apply(lambda x: tuple(x))

    df[f'num_prompts'] = df["PromptColumnsInUse"].apply(lambda x: len(x))
    # Dropping specified columns
    drop_columns = ['ModelParameters', 'TP_count', 'TN_count', 'SkipNulls', 
                    'FP_count', 'FN_count', 'sensitivity', 'precision', 
                    'specificity', 'f1_score', 'accuracy', 'TrainOrigin','PromptColumnsInUse','y_true','y_pred']
    df.drop(columns=drop_columns, inplace=True)
    
    # Step 2: Split the DataFrame into subsections
    unique_values = df[f'{column_X}_Tuple'].unique()
    subsections = {value: df[df[f'{column_X}_Tuple'] == value] for value in unique_values}
    
    # Step 3: Sort each subsection by 'column_Y' and get the best model
    sorted_subsections = {}
    best_models = {}
    for value, subsection in subsections.items():
        sorted_subsection = subsection.sort_values(by=column_Y, ascending=False)
        sorted_subsections[value] = sorted_subsection
        best_models[value] = sorted_subsection.iloc[0][model_column]
    
    # Displaying each sorted subsection without truncation
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_colwidth', None)
    
    for value, subsection in sorted_subsections.items():
        print(f"\nSubsection for {value}:")
        print(subsection)
        print(f"Best model for {value}: {best_models[value]}\n\n\n\n")
    
    # Step 4: Group by 'model_column', calculate statistics, and sort by mean 'column_Y'
    model_stats = df.groupby(model_column)[column_Y].agg(['max', 'min', 'mean']).sort_values(by='mean', ascending=False)
    
    # Displaying sorted model statistics
    print("Sorted Model Statistics (Highest, Lowest, Average of balanced_accuracy):")
    print(model_stats)

# Example usage of the function
file_path = 'benchmarking_stats_df_12_18.pkl'

dataset_to_method_to_prompt_size_to_acc = get_dataset_to_method_to_prompt_size_to_acc(file_path)
# file_path = 'benchmarking_for_stage_2.pkl'
process_and_display_data(file_path, 'TestOrigin', 'balanced_accuracy', 'Model')
