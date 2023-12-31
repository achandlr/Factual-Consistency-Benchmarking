import pandas as pd
from src.utils.Evaluator import Evaluator
import pickle
from src.utils.DataLoader import BinaryDataLoader, filter_df_by_non_null_prompt, load_placeholder_data, convert_csv_files_to_df
from src.utils.ModelLoader import load_models
from src.utils.Experiments import load_experiment_configs  # filter_df_by_experiment_config
# from src.models.WRENCHModels.wrench_model_runner import run_wrench_models, WrenchModelBaseClass
from src.utils.logger import setup_logger
# from src.models.SKLearnModels.SkLearnModels import instantiate_sk_learn_models
# from src.models.WRENCHModels.DawidSkeneModel import DawidSkeneModel
# from src.models.WRENCHModels.SnorkelModels import SnorkelLabelModel, SnorkelMajorityLabelVoter
# from src.models.WRENCHModels.PyannoModels import PyAnnoModelB
# from src.models.AlexEnsemblingMethods.ConditionalLR import ConditionalLR



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

        # TODO: either use these columns or take them out
        new_prompt_columns = None # our prompts
        existing_prompt_columns = None # existing prompts
        ground_truth_column_name = "Manual_Eval"

        experiment_configs = load_experiment_configs()

        training_information_message = f"Training a total of {len(self.models)} models on a total of {len(experiment_configs)} different configurations:"
        print(training_information_message)
        self.logger.info(training_information_message)

        for experiment_config in experiment_configs:
            self.logger.info(f"START of experiment: {experiment_config}")
            # Access the experiment configuration
            train_origin = experiment_config.train_origin
            test_origin = experiment_config.test_origin
            skip_nulls = experiment_config.skip_rows_with_null_values
            prompt_columns_in_use = experiment_config.prompt_columns_in_use

            data_loader = BinaryDataLoader()           
            DATA_TRAIN_TEST_FILTER_COLUMN = 'origin'
            # Note: Line below was previously used to determine best stage 2 consensus method 
            # DATA_TRAIN_TEST_FILTER_COLUMN = "Data_Type"


            df = data_loader.convert_llm_answers_to_binary(df, columns = prompt_columns_in_use, ground_truth_column_name =  ground_truth_column_name)
            # Note: This line below was only for determining best consensus stage 2 method

            # df = data_loader.convert_llm_answers_to_binary(df, columns = prompt_columns_in_use, ground_truth_column_name =  ground_truth_column_name, llm_parsing_method = "convert_correct_wrong_to_binary")

            
            data_loader.report_llm_answer_errors()
            if skip_nulls:
                # TODO: Confirm that all rows where one of the columns in prompt_columns_in_use is null is removed from our dataset
                df_no_null = filter_df_by_non_null_prompt(df, needed_non_null_columns = prompt_columns_in_use + [ground_truth_column_name])
                train_df = df_no_null[df_no_null[DATA_TRAIN_TEST_FILTER_COLUMN].isin(train_origin)]
                # Note: Line below was previously used to determine best stage 2 consensus method 
                # train_df =train_df[train_df['origin'] != 'HALU_EVAL_SUMMARIZATION'].sample(frac=1, random_state=42).iloc[0:1000]
                test_df = df_no_null[df_no_null[DATA_TRAIN_TEST_FILTER_COLUMN].isin(test_origin)]
            else:
                train_df = df[df[DATA_TRAIN_TEST_FILTER_COLUMN].isin(train_origin)]
                test_df = df[df[DATA_TRAIN_TEST_FILTER_COLUMN].isin(test_origin)]


            X_train= train_df[prompt_columns_in_use].to_numpy() # .transpose() 
            Y_train= train_df[ground_truth_column_name].to_numpy() #.transpose() 
            X_test= test_df[prompt_columns_in_use].to_numpy() # .transpose() 
            Y_test= test_df[ground_truth_column_name].to_numpy() #.transpose() 

            for model in self.models:
                self.logger.info(f"Running experiment: {experiment_config} on model {self.models}")
                # # is it possible so that this is not a check that needs to be done but is rather done under the hood in the WrenchModelClass
                # 4. # if we do the following below, we want it to return Y_pred and be bale to report model parameters. think step by step about what I have said and improve it
                # if isinstance(model, WrenchModelBaseClass):  # Assuming WrenchModelBaseClass is a marker class
                #     results_df = self.run_wrench_model(train_df, test_df, model.__class__.__name__)
                # Train the model
                model.train_with_timing_stats(X_train, Y_train)
                # Make predictions
                Y_pred = model.predict(X_test)
                # Evaluate the model
                stats = Evaluator.get_stats(Y_test, Y_pred)
                # Compile results
                self.compile_results(model, stats, experiment_config)

            self.logger.info(f"END of experiment: {experiment_config}")

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
    import time
    logger = setup_logger()
    print("START OF BENCHMARKING")
    logger.info("START OF BENCHMARKING")
    bench_start_time = time.time()
    # Stage 1: Load all the models
    # models = load_models(use_dawid_skene_models=False, use_sklearn_models=False, use_alex_models=False, use_snorkel_models=True, use_lgb_models = False, use_pytorch_models = False)

    models = load_models(use_dawid_skene_models=True, use_sklearn_models=True, use_alex_models=True, use_snorkel_models=True, use_lgb_models = True, use_pytorch_models = False)
    # models = load_models(use_dawid_skene_models=True, use_sklearn_models=True, use_alex_models=True, use_snorkel_models=True, use_lgb_models = True, use_pytorch_models = False)
    # models = load_models(use_dawid_skene_models=False, use_sklearn_models=True, use_alex_models=False, use_snorkel_models=False, use_lgb_models = False, use_pytorch_models = False)
    # models = load_models(use_dawid_skene_models=False, use_sklearn_models=False, use_alex_models=False, use_snorkel_models=False, use_lgb_models = True)

    # Stage 2: Load the data
    LOAD_PLACE_HOLDER_DATA = False
    if LOAD_PLACE_HOLDER_DATA:
        df = load_placeholder_data()
    else:
        # Note: This was previously used to determine the best stage 2 method
        # df = pd.read_csv(r"data\imported\datasets\aggrefact_val_test_halu_4512_all_results.csv")
        # TODO: Change this after Devesh is done fully with LLM calls so that it is over all of the prompts including existing prompts
        df = convert_csv_files_to_df(r"data\imported\datasets\aggrefact_val_test_halu_4931_dict_1.csv", r"data\imported\datasets\aggrefact_val_test_halu_4931_dict_2.csv")
        # with open("dataframe_binary_results", "rb") as f: df = pickle.load(f)
    benchmark = Benchmark(models = models, df = df)
    benchmark.run_benchmark()

    benchmarking_stats_df = benchmark.results

    with open("benchmarking_stats_df_12_18.pkl", "wb") as f:
        pickle.dump(benchmarking_stats_df, f)
    # Note: This was for storing best ensembling results for stage 2
    # with open("benchmarking_for_stage_2.pkl", "wb") as f:
    #     pickle.dump(benchmarking_stats_df, f)

    bench_end_time = time.time()
    print("END OF BENCHMARKING")
    print(f"Benchmarking took {(bench_end_time - bench_start_time)/60} minutes")
    logger.info("END OF BENCHMARKING")
    logger.info(f"Benchmarking took {(bench_end_time - bench_start_time)/60} minutes")