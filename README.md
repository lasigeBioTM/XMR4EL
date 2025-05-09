## eXtreme Multi-Label Ranking for Entity Linking - XMR4EL

Following PECOS (https://github.com/amzn/pecos/tree/mainline) pipeline, this project tries to implement any kind of model into it. 

With the use of an Hierarchical Tree, we can try different options of vectorizers, transformers, clustering and classifier models to test our data.

Even thou this project has as main vision the use of it in Entity Linking, it can be used with any data, that uses as tool eXtreme Multi-Label Ranking (XMR) to work.

Paper: (Not made yet)

### Requirements and Installation

* Python (3.12)

All the packages used are displayed in the requirements.txt.

### CUDA Version

* CUDA (11.4 - 11.8)

Right now all the gpu models are from rapids.ai, and are tested in an docker enviroment, given in the dockerfile.


python src/python/xlinker/evaluate.py -dataset bc5cdr -ent_type Disease -kb medic -model_dir test/test_data/saved_trees/TreeDisease100 -top_k 3 --abbrv --pipeline --threshold 0.15 --ppr

Result: Top-1 Accuracy: 0.7189797794117647, 0.7210477941176471

python src/python/xlinker/evaluate.py -dataset bc5cdr -ent_type Chemical -kb ctd_chemicals -model_dir test/test_data/saved_trees/TreeDisease100 -top_k 5 --abbrv --pipeline  --ppr

Result: Top-1 Accuracy: 0.9215763546798029

python src/python/xlinker/evaluate.py -dataset bc5cdr -ent_type Disease -kb medic -model_dir test/test_data/saved_trees/TreeDisease500 -top_k 3 --abbrv --pipeline --threshold 0.15 --ppr

Result: Top-1 Accuracy: 0.7233455882352942

python src/python/xlinker/evaluate.py -dataset bc5cdr -ent_type Disease -kb medic -model_dir test/test_data/saved_trees/TreeDisease1000 -top_k 3 --abbrv --pipeline --threshold 0.15 --ppr

Result: Top-1 Accuracy: 0.7244944852941176

python src/python/xlinker/evaluate.py -dataset biored -ent_type Chemical -kb ctd_chemicals -model_dir 
test/test_data/saved_trees/TreeDisease1000 -top_k 5 --abbrv --pipeline  --ppr

Result: Top-1 Accuracy: 0.9217733990147783