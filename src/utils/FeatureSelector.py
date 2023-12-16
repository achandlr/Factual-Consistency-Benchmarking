import pandas as pd
from src.utils.Evaluator import Evaluator
import pickle
from src.utils.DataLoader import BinaryDataLoader, filter_df_by_non_null_prompt, load_placeholder_data, convert_csv_files_to_df
from src.utils.ModelLoader import load_models
from src.utils.Experiments import load_config_for_feature_selection  # filter_df_by_experiment_config
# from src.models.WRENCHModels.wrench_model_runner import run_wrench_models, WrenchModelBaseClass
from src.utils.logger import setup_logger
# from src.models.SKLearnModels.SkLearnModels import instantiate_sk_learn_models
# from src.models.WRENCHModels.DawidSkeneModel import DawidSkeneModel
# from src.models.WRENCHModels.SnorkelModels import SnorkelLabelModel, SnorkelMajorityLabelVoter
# from src.models.WRENCHModels.PyannoModels import PyAnnoModelB
# from src.models.AlexEnsemblingMethods.ConditionalLR import ConditionalLR
import mrmr
from sklearn.feature_selection import RFE
from mrmr import mrmr_classif




class FeatureSelector:
    def __init__(self, models, df):
        """
        :param models: List of models (instances of ModelBaseClass or its subclasses).
        """
        self.models = models
        self.df = df
        self.results = pd.DataFrame()
        self.logger = setup_logger()

    def select_best_features_for_each_size(self):
        """
        Runs the benchmarking process: training models, making predictions, and evaluating performance.
        """
        df = self.df
        ground_truth_column_name = "Manual_Eval"

        experiment_config = load_config_for_feature_selection()


        self.logger.info(f"START of feature_selection: {experiment_config}")
        # Access the experiment configuration
        train_origin = experiment_config.train_origin
        test_origin = experiment_config.test_origin
        skip_nulls = experiment_config.skip_rows_with_null_values
        prompt_columns_in_use = experiment_config.prompt_columns_in_use

        curr_prompt_columns_in_use = experiment_config.prompt_columns_in_use

        prompt_size_to_selected_features_method_mmr = {}
        prompt_size_to_selected_features_method_rfe_svc = {}

        data_loader = BinaryDataLoader()
        # TODO: This line should not be necessary once all data is binary
        df = data_loader.convert_llm_answers_to_binary(df, columns = prompt_columns_in_use, ground_truth_column_name =  ground_truth_column_name)

        
        data_loader.report_llm_answer_errors()
        if skip_nulls:
            # TODO: Confirm that all rows where one of the columns in prompt_columns_in_use is null is removed from our dataset
            df_no_null = filter_df_by_non_null_prompt(df, needed_non_null_columns = curr_prompt_columns_in_use + [ground_truth_column_name])
            train_df = df_no_null[df_no_null['origin'].isin(train_origin)]
            test_df = df_no_null[df_no_null['origin'].isin(test_origin)]
        else:
            train_df = df[df['origin'].isin(train_origin)]
            test_df = df[df['origin'].isin(test_origin)]

        X_train= train_df[prompt_columns_in_use].to_numpy() # .transpose() 
        X_train_df = train_df[prompt_columns_in_use]
        y_train_df = train_df[ground_truth_column_name]

        Y_train= train_df[ground_truth_column_name].to_numpy() #.transpose() 
        X_test= test_df[prompt_columns_in_use].to_numpy() # .transpose() 
        Y_test= test_df[ground_truth_column_name].to_numpy() #.transpose() 

        while len(curr_prompt_columns_in_use) > 1:
            selected_features = mrmr_classif(X=X_train_df, y=y_train_df, K=len(curr_prompt_columns_in_use)-1)
            curr_prompt_columns_in_use  = selected_features
            prompt_size_to_selected_features_method_mmr[len(curr_prompt_columns_in_use)] = selected_features


        curr_prompt_columns_in_use = experiment_config.prompt_columns_in_use
        from sklearn.svm import SVC
        while len(curr_prompt_columns_in_use) > 1:
            
            # balanced_accuracies = []
            # # TODO: refactor this so that we have information to select best prompt_indices_in_use-1 prompts or num_desired_prompts - 1
            # for model in self.models:

            model = SVC(C=1, kernel='linear', gamma='scale')
            selector = RFE(model, n_features_to_select=len(curr_prompt_columns_in_use)- 1, step=1)
            selector = selector.fit(X_train, Y_train)
            selector.support_
            selector.ranking_

            curr_prompt_columns_in_use = [item for item, flag in zip(curr_prompt_columns_in_use, selector.support_) if flag]

            prompt_size_to_selected_features_method_rfe_svc[len(curr_prompt_columns_in_use)] = curr_prompt_columns_in_use

        selected_prompts_using_different_methods = {"rfe": prompt_size_to_selected_features_method_rfe_svc, "mrmr": prompt_size_to_selected_features_method_mmr}
        with open("selected_prompts_using_different_methods.pkl", "wb") as f:
            pickle.dump(selected_prompts_using_different_methods, f)


    def compile_results(self, model, stats, experiment_config):
        model_name = model.__class__.__name__
        model_parameters = model.report_trained_parameters()
        result = {
            'Model': model_name,
            'TrainOrigin': experiment_config.train_origin,
            'TestOrigin': experiment_config.test_origin,
            'SkipNulls': experiment_config.skip_rows_with_null_values,
            'PromptColumnsInUse': experiment_config.prompt_columns_in_use,
            'ModelParameters': model_parameters
        }
        result.update(stats)

        result_df = pd.DataFrame([result])  # Convert the result to a DataFrame
        # TODO: Maybe organize the data better, hierarchically
        self.results = pd.concat([self.results, result_df], ignore_index=True)



    def get_results(self):
        """
        Returns the results DataFrame.
        """
        return self.results



if __name__ == "__main__":
    LOAD_PLACE_HOLDER_DATA = False
    if LOAD_PLACE_HOLDER_DATA:
        df = load_placeholder_data()
    else:
        # TODO: Devesh, Import a new dataset here, where the train is either [all data, OR unsure data], and test is unsure_data
        df = convert_csv_files_to_df(r"data\imported\datasets\aggrefact_val_test_halu_4931_dict_1.csv", r"data\imported\datasets\aggrefact_val_test_halu_4931_dict_2.csv")
        # with open("dataframe_binary_results", "rb") as f: df = pickle.load(f)
    feature_selector = FeatureSelector(models = None, df = df)
    feature_selector.select_best_features_for_each_size()

