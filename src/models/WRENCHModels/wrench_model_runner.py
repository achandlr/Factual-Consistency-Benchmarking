import subprocess
import pickle
import os
import pandas as pd

class WrenchModelBaseClass:
    raise NotImplementedError()
def run_wrench_model(train_df, test_df, model_name):
    # Serialize the dataframes
    train_df.to_pickle('train_df.pkl')
    test_df.to_pickle('test_df.pkl')

    # Prepare the command to run the WRENCH model in a separate environment
    command = f'python run_wrench_model.py {model_name} train_df.pkl test_df.pkl'

    # Run the subprocess
    subprocess.run(command, shell=True)

    # Read back the results
    results_df = pd.read_pickle('wrench_results.pkl')

    # Clean up
    os.remove('train_df.pkl')
    os.remove('test_df.pkl')
    os.remove('wrench_results.pkl')

    return results_df