import pickle

# TODO: Import new data after FeatureSelector is rerun over larger dataframe including more prompts

with open("selected_prompts_using_different_methods.pkl", "rb") as f: selected_prompts_using_different_methods = pickle.load(f)

desired_prompt_sizes_rfe = [4, 7, 11, 15, 18, 20]
desired_prompt_sizes_mrmr = [3, 4, 5, 7, 9 ,11, 15, 18, 20]

best_rfe_prompt_sets = [selected_prompts_using_different_methods['rfe'][prompt_size] for prompt_size in desired_prompt_sizes_rfe]
best_mrmr_prompt_sets = [selected_prompts_using_different_methods['mrmr'][prompt_size] for prompt_size in desired_prompt_sizes_mrmr]

print("DONE")