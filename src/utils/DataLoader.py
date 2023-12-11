import numpy as np
import pickle
import re
import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import pandas as pd

def filter_df_by_non_null_prompt(df, needed_non_null_columns):
    """
    Filters the DataFrame to include only rows where all specified columns are non-null.

    :param df: The pandas DataFrame to filter.
    :param needed_non_null_columns: List of column names that must be non-null.
    :return: Filtered DataFrame.
    """
    # Create a boolean mask where each row is True if all needed columns are non-null
    non_null_mask = df[needed_non_null_columns].notnull().all(axis=1)

    # Apply the mask to the DataFrame
    filtered_df = df[non_null_mask]

    return filtered_df


class BinaryDataLoader:
    def __init__(self, train_file, test_file):
        self.train_file = train_file
        self.test_file = test_file
        # self.X_train, self.Y_train, self.X_test, self.Y_test = None, None, None, None

    def load_data(self, data_format = "dictionary_to_df", skip_rows_with_null = True):

        if data_format == "dictionary_to_df":
            self.X_train, self.Y_train = self.convert_devesh_df_to_x_y_array(self.train_file, skip_rows_with_null)
            self.X_test, self.Y_test = self.convert_devesh_df_to_x_y_array(self.test_file, skip_rows_with_null)
        else:
            raise NotImplementedError()

    # TODO: prompt selector, data selector
    @staticmethod
    def convert_devesh_df_to_x_y_array(file_name, skip_rows_with_null):
        # with open("21_summary_1195", "rb") as f:
        #     devesh_aggrefact_data_2 = pickle.load(f) 
        with open(file_name, "rb") as f:
            devesh_df_data = pickle.load(f) 
        X = []
        contexts = devesh_df_data[list(devesh_df_data.keys())[0]]["Context"].tolist()
        summaries = devesh_df_data[list(devesh_df_data.keys())[0]]["Summary"].tolist()
        bad_value_indices_set = set()

        for key, df in devesh_df_data.items():

            unparsed_data = list(df["Unparsed"])
            predictions = [BinaryDataLoader.advanced_parse_llm_output(output) for output in unparsed_data]
            bad_value_indices = [i for i, x in enumerate(predictions) if x is None]

            # bad_value_indices = [i if # df.index[df['Unparsed'] == None].tolist()
            bad_value_indices_set.update(set(bad_value_indices))
            X.append(predictions)
        X = np.array(X)
        X = np.transpose(X)


        ground_truth = devesh_df_data['GPT_improved_nli_style_prompt']["Manual_Eval"]
        y = [1 if z == "Correct" else 0 for z in ground_truth]
    

        if skip_rows_with_null :

            X_row_count_with_none = 0
            X_without_none = []
            y_without_none = []
            summaries_without_none = []
            context_without_none = []


            for x, y_val, summary, context  in zip(X, y, summaries, contexts):
                if None in x:
                    X_row_count_with_none +=1
                    continue
                else:

                    X_without_none.append(x)
                    y_without_none.append(y_val)
                    summaries_without_none.append(summary)
                    context_without_none.append(context)

            X_without_none = np.array(X_without_none)
            y_without_none = np.array(y_without_none)

            X_without_none = X_without_none.astype(int)
            y_without_none = y_without_none.astype(int)
            return {"x_array" : X_without_none, "y_array": y_without_none, "summaries" : summaries_without_none, "contexts" : context_without_none}
        
        else:
            X = X.astype(int) 
            y = y.astype(int)
            return {"x_array" : X, "y_array": y, "summaries" : summaries, "contexts" : contexts}


    @staticmethod
    def advanced_parse_llm_output(input_string):
        """
        Parses the LLM output for a variety of responses including detailed explanations.
        Prioritizes detection of negated phrases.

        :param input_string: The string output from the LLM.
        :return: True, False, or None based on the analysis of the input string.
        """
        if input_string == None:
            return None
        elif isinstance(input_string, int) or isinstance(input_string, float):
            return int(input_string)

        # Define regex patterns for negations and affirmations
        # negated_pattern = r'\b(not supported|inconsistent|false)\b'
        # affirmative_pattern = r'\b(supported|consistent|true)\b'
        negated_pattern = r'\b(not supported|inconsistent|unsupported)\b'
        affirmative_pattern = r'\b(supported|consistent)\b'
        
        negated_match = re.findall(negated_pattern, input_string)
        affirmative_match = re.findall(affirmative_pattern, input_string)

        # Check for negated phrases first
        if negated_match:
            return 0
        elif affirmative_match:
            return 1
        else:

            return None
    
    @staticmethod
    def convert_llm_answers_to_binary(df, columns):
        for col in columns:
            if col in df.columns:
                df[col] = df[col].apply(lambda x: BinaryDataLoader.advanced_parse_llm_output(x) if isinstance(x, str) else x)
        return df

    def convert_to_torch_train_test_data_loader(self, batch_size=64, set_class_data_loaders = True):
        # Convert data to PyTorch tensors, ensuring they are of numeric type
        X_train_tensor = torch.tensor(self.X_train.astype(np.float32))
        Y_train_tensor = torch.tensor(self.Y_train.astype(np.float32))
        X_test_tensor = torch.tensor(self.X_test.astype(np.float32))
        Y_test_tensor = torch.tensor(self.Y_test.astype(np.float32))

        # Create datasets
        train_dataset = TensorDataset(X_train_tensor, Y_train_tensor)
        test_dataset = TensorDataset(X_test_tensor, Y_test_tensor)

        # Create dataloaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        if set_class_data_loaders:
            self.train_loader = train_loader
            self.test_loader = test_loader
        
        return train_loader, test_loader