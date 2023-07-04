#%%
import joblib
import torch
import torch.nn as nn
import time

device_list = [1,2,3]

def create_model(device):
    return f"assigned_device: {device}"

def wrapper(value, job_index):
    print(f"cpu count: {job_index}")
    device = device_list[job_index % len(device_list)]
    model = create_model(device)

    time.sleep(device*10)
    output = model + '#' + value
    # Process the output as needed

    return output

#%%
results = joblib.Parallel(n_jobs=3)(
    joblib.delayed(wrapper)(value, i)
    for i, value in enumerate(['apple', 'mango', 'banana', 'pineapple', 'cucumber', 'musk'])
)
# %%
