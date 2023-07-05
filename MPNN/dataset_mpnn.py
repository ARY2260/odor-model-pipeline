from deepchem.data.data_loader import CSVLoader
from featurizer import CustomFeaturizer
import pandas as pd

def get_class_imbalance_ratio(df):
    class_counts = df.sum().to_numpy()
    total = class_counts.sum()
    class_imbalance_ratio = class_counts/total
    return class_imbalance_ratio.tolist()
    

def get_dataset(csv_path='./../curated_GS_LF_merged_4984.csv'):
    curated_df = pd.read_csv(csv_path)
    odor_tasks = list(curated_df.drop(columns=['nonStereoSMILES', 'descriptors']).columns)
    print("number of odor tasks = ", len(odor_tasks))
    
    featurizer = CustomFeaturizer()
    smiles_field = 'nonStereoSMILES'
    loader = CSVLoader(tasks=odor_tasks, feature_field=smiles_field, featurizer=featurizer)
    
    input_file = [csv_path]
    data_dir='./metadata'
    dataset = loader.create_dataset(inputs=input_file, data_dir=data_dir)

    class_imbalance_ratio = get_class_imbalance_ratio(curated_df.drop(columns=['nonStereoSMILES', 'descriptors']))
    return dataset, class_imbalance_ratio
