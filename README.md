# RepEval2017 LCT-MALTA

## Requirements
~~~ bash
sudo apt-get install python python-pip
sudo pip install -r requirements.txt
~~~

## Training
To train the model:
~~~ bash
python train_mnli.py model_type model_name [...params...]
~~~

## Making prediction
To make prediction from test file, and export to a csv that satisfies RepEval 2017 shared task format:
~~~ bash
python predictions.py model_type model_name [...params...]
~~~

## XGBoost classifier
To train XGBoost classifier and make prediction from existing model:
~~~ bash
python xgboost_classfier.py model_type model_name [...params...]
~~~

To know the possible params for each command above, simply run:
~~~ bash
python [command_name].py
~~~

## Acknowledgement
This code is based on [RepEval 2017 baseline](https://github.com/NYU-MLL/multiNLI) and [PyTorch NLI example](https://github.com/pytorch/examples/tree/master/snli).
