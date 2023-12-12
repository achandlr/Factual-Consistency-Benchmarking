
import pandas as pd
from src.utils.Evaluator import Evaluator
from src.utils.DataLoader import BinaryDataLoader, filter_df_by_non_null_prompt
from src.utils.Experiments import load_experiment_configs  # filter_df_by_experiment_config
from src.models.SKLearnModels.SkLearnModels import instantiate_sk_learn_models
# from src.models.WRENCHModels.wrench_model_runner import run_wrench_models, WrenchModelBaseClass
import random
import pickle
from src.utils.logger import setup_logger
from src.models.WRENCHModels.DawidSkeneModel import DawidSkeneModel
from src.models.WRENCHModels.SnorkelModels import SnorkelLabelModel, SnorkelMajorityLabelVoter
# from src.models.WRENCHModels.PyannoModels import PyAnnoModelB
from src.models.AlexEnsemblingMethods.ConditionalLR import ConditionalLR

class Benchmark:
    def __init__(self, models, df):
        """
        :param models: List of models (instances of ModelBaseClass or its subclasses).
        """
        self.models = models
        self.df = df
        self.results = pd.DataFrame()
        self.logger = setup_logger()

    def run_benchmark(self):
        """
        Runs the benchmarking process: training models, making predictions, and evaluating performance.
        """
        # Load the data
        # data_loader = BinaryDataLoader()

        df = self.df

        new_prompt_columns = None # our prompts
        existing_prompt_columns = None # existing prompts
        ground_truth_column_name = "Manual_Eval"

        experiment_configs = load_experiment_configs()

        for experiment_config in experiment_configs:
            # Access the experiment configuration
            train_origin = experiment_config.train_origin
            test_origin = experiment_config.test_origin
            skip_nulls = experiment_config.skip_rows_with_null_values
            prompt_columns_in_use = experiment_config.prompt_columns_in_use
            # TODO: This line should not be necessary once all data is binary
            df = BinaryDataLoader.convert_llm_answers_to_binary(df, columns = prompt_columns_in_use + [ground_truth_column_name])
            if skip_nulls:
                # TODO: Use advanced_parse_llm_output before filter by non_null
                
                df_no_null = filter_df_by_non_null_prompt(df, needed_non_null_columns = prompt_columns_in_use + [ground_truth_column_name])
                train_df = df_no_null[df_no_null['origin'].isin(train_origin)]
                test_df = df_no_null[df_no_null['origin'].isin(test_origin)]
            else:
                train_df = df[df['origin'].isin(train_origin)]
                test_df = df[df['origin'].isin(test_origin)]


            X_train= train_df[prompt_columns_in_use].to_numpy() # .transpose() 
            Y_train= train_df[ground_truth_column_name].to_numpy() #.transpose() 
            X_test= test_df[prompt_columns_in_use].to_numpy() # .transpose() 
            Y_test= test_df[ground_truth_column_name].to_numpy() #.transpose() 

            for model in self.models:
                # # is it possible so that this is not a check that needs to be done but is rather done under the hood in the WrenchModelClass
                # 4. # if we do the following below, we want it to return Y_pred and be bale to report model parameters. think step by step about what I have said and improve it
                # if isinstance(model, WrenchModelBaseClass):  # Assuming WrenchModelBaseClass is a marker class
                #     results_df = self.run_wrench_model(train_df, test_df, model.__class__.__name__)
                # Train the model
                model.train(X_train, Y_train)
                # Make predictions
                Y_pred = model.predict(X_test)
                # Evaluate the model
                stats = Evaluator.get_stats(Y_test, Y_pred)
                # Compile results
                self.compile_results(model, stats, experiment_config)

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


    wrench_models = [ SnorkelLabelModel(), SnorkelMajorityLabelVoter(), DawidSkeneModel()] # , MajorityVotingModel(), SnorkelModel(), MeTalModel(), WeaselModel(), ConsensusModel()] # PyAnnoModelB(),
    # sk_learn_models = instantiate_sk_learn_models()
    alex_ensembling_models = [ConditionalLR()]
    models = alex_ensembling_models  + wrench_models # + sk_learn_models

    '''
    TODO: Implement and add Pyanno models
    '''
    DEBUG = True
    if DEBUG:
        def generate_manual_eval():
            return random.choices([1, 0, None], weights=[45, 45, 10], k=1)[0]

        def generate_origin():
            origins = ["AGGREFACCT_SOTA_CNN_DM_DEV", "AGGREFACCT_SOTA_CNN_DM_TEST", "AGGREFACCT_SOTA_CNN_DM_VAL", "AGGREFACCT_SOTA_XSUM_TEST", "AGGREFACCT_SOTA_XSUM_DEV"]
            # Define your likelihood for each origin value
            return random.choices(origins, weights=[20, 20, 20, 20, 20], k=1)[0]

        if DEBUG:
            num_total_data_points = 2000
            data = {
                "Context": [f"Context {i}" for i in range(0, num_total_data_points)],
                "Summary": [f"Summary {i}" for i in range(0, num_total_data_points)],
                "Manual_Eval": [generate_manual_eval() for _ in range(num_total_data_points)],
            }

            for col in ["col1", "col2", "col3", "col4", "col5"]:
                data[col] = [random.choice([data["Manual_Eval"][i], None]) if random.random() < 0.1 else data["Manual_Eval"][i] for i in range(num_total_data_points)]
            
            data["origin"] = [generate_origin() for _ in range(num_total_data_points)]

        # data = {
        # "Context": ["Context 1", "Context 2", "Context 3", "Context 4", "Context 5"],
        # "Summary": ["Summary 1", "Summary 2", "Summary 3", "Summary 4", "Summary 5"],
        # "col1": ["Prediction 1", "Prediction 2", "Prediction 3", "Prediction 4", "Prediction 5"],
        # "col2": ["Prediction 1", None, "Prediction 3", "Prediction 4", "Prediction 5"],
        # "col3": ["Prediction 1", "Prediction 2", "Prediction 3", "Prediction 4", "Prediction 5"],
        # "col4": ["Prediction 1", "Prediction 2", "Prediction 3", "Prediction 4", "Prediction 5"],
        # "col5": ["Prediction 1", "Prediction 2", "Prediction 3", "Prediction 4", "Prediction 5"],
        # "Manual_Eval": ["Correct", "Incorrect", "Correct", "Incorrect", "Correct"],
        # "origin": ["AGGREFACCT_SOTA_CNN_DM_DEV", "AGGREFACCT_SOTA_CNN_DM_TEST", "AGGREFACCT_SOTA_CNN_DM_VAL", "AGGREFACCT_SOTA_XSUM_TEST", "AGGREFACCT_SOTA_XSUM_DEV"]}

        placeholder_df = pd.DataFrame(data)
        df = placeholder_df
    else:
        with open("dataframe_binary_results", "rb") as f: df = pickle.load(f)


    benchmark = Benchmark(models = models, df = df)
    benchmark.run_benchmark()

    benchmarking_stats_df = benchmark.results

    with open("benchmarking_stats_df.pkl", "wb") as f:
        pickle.dump(benchmarking_stats_df, f)