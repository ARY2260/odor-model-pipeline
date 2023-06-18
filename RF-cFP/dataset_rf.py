#%%
from deepchem.data.data_loader import CSVLoader
from deepchem.feat import CircularFingerprint
import pandas as pd

def get_dataset():
    curated_df = pd.read_csv('./../curated_GS_LF_merged_4984.csv')
    odor_tasks = list(curated_df.drop(columns=['nonStereoSMILES', 'descriptors']).columns)
    assert len(odor_tasks) == 138
    
    featurizer = CircularFingerprint(size=2048, radius=4)
    smiles_field = 'nonStereoSMILES'
    loader = CSVLoader(tasks=odor_tasks, feature_field=smiles_field, featurizer=featurizer)
    
    input_file = ['./../curated_GS_LF_merged_4984.csv']
    data_dir='./metadata'
    dataset = loader.create_dataset(inputs=input_file, data_dir=data_dir)
    return dataset
# %%
