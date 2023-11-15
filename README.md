![image](https://github.com/WPCdemo/WPC/assets/126042536/ab8fa16f-93f6-4816-ac19-105ee69979ed)Thanks for your attention. The following instructions can help you reproduce the experiments.

# Environment
conda env create -f environments.yaml

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
