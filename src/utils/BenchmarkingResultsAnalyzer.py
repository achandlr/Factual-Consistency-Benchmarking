import pandas as pd

def process_and_display_data(file_path, column_X, column_Y, model_column):
    # Step 1: Read in the file
    df = pd.read_pickle(file_path)
    

    # Converting lists in 'column_X' to tuples
    df[f'{column_X}_Tuple'] = df[column_X].apply(lambda x: tuple(x))

    df[f'num_prompts'] = df["PromptColumnsInUse"].apply(lambda x: len(x))
    # Dropping specified columns
    drop_columns = ['ModelParameters', 'TP_count', 'TN_count', 'SkipNulls', 
                    'FP_count', 'FN_count', 'sensitivity', 'precision', 
                    'specificity', 'f1_score', 'accuracy', 'TestOrigin','PromptColumnsInUse']
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
file_path = 'benchmarking_for_stage_2.pkl'
process_and_display_data(file_path, 'TrainOrigin', 'balanced_accuracy', 'Model')
