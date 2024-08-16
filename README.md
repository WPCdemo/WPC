# Environment
conda env create -f environments.yaml

# Platfrom
Our experiments are conducted on a platform with NVIDIA GeForce RTX 3090.

# Files
- **args.py**: define command line arguments
- **data_load.py**: load and process data
- **embedder.py**: define embedding layers
- **layers.py**: define model details
- **main.py**: define the main function
- **test.py**: used for testing the model
- **utils.py**：commonly used and generic functions
  
# Running
## For Cora dataset
python main.py --embedder wpc --dataset cora --im_class_num 3
## For Citeseer dataset
python main.py --embedder wpc --dataset citeseer --im_class_num 3
## For WikiCS dataset
python main.py --embedder wpc --dataset wikics --im_class_num 3
## For Pubmed dataset
python main.py --embedder wpc --dataset pubmed --im_class_num 1
