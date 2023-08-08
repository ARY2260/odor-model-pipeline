# odor-model-pipeline

## Setup
```
conda create -n odor_pom python=3.9
conda activate odor_pom
pip3 install torch torchvision torchaudio
pip install --pre deepchem
pip install  dgl -f https://data.dgl.ai/wheels/cu117/repo.html
pip install  dglgo -f https://data.dgl.ai/wheels-test/repo.html
pip install dgllife
pip install pytest
pip install ipykernel
pip install scikit-multilearn
```

## [50 trial - 5 fold - 200 epoch] Random Search CV search
```
conda activate odor_pom
cd MPNN
python random_search_cv.py -f 5 -t 50 -e 200
```

## MultiGPU Random Search CV test
```
conda activate odor_pom
cd MPNN
python multi_gpu_random_search_cv.py
```
NOTE: n_jobs parameter in multi_gpu_random_search_cv() should be equal to no of GPUs available to do the computations.