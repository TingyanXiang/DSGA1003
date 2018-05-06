# DSGA1003

### Group Member:Daoyang Shan, Tingyan Xiang, Kairuo(Edward) Zhou

### How to run this model
Clone this repo to your local, make sure you have all required packages installed (LightGBM is not implemented in practive, feel free comment out its dependency in /src/TDE.py), inside src folder, type the following command in shell:
```
python main.py --[option name] [option value]
```
Option name and value refer to the model settings that you're able to adjust. For example, if you want to use random forest as base learner, set the dependency length to 10, and require the max depth of each tree to be 10, you may type:
```
python main.py --classifier_name random_forest --max_depth 10 --dependency_length 10
```
Currently only decision tree, random forest and XGBoost are supported. LightGBM is in progress, although there's no guarantee to its completion :(
Check main.py for all possible model settings, including learner side and system side.
