from abc import ABCMeta, abstractmethod
import os
from datetime import datetime as dt
import json

# autogluon
import autogluon as ag
from autogluon import TabularPrediction


class TableAnalyzerFactory:
    @classmethod
    def get_instance(cls, model_type, config):
        if model_type == "autogluon":
            return AutogluonAnalyzer(config=config)
        elif model_type == "xgboost":
            return XGBoostAnalyzer(config=config)
        else:
            raise NotImplementedError("{} model is not implemented!")

class TableAnalyzer(object):
    __metaclass__ = ABCMeta
    
    def __init__(self, config=None):
        self._model      = None
        self._output_base_dir = None
        self._set_output_base_dir(config["output_base_dir"])
        self._label_column = None
        os.makedirs(self._output_base_dir, exist_ok=True)
        if "model_path" in config:
            self.load(config["model_path"])
        
    @abstractmethod
    def train(self, train_data, val_data=None, params=None):
        pass
    
    @abstractmethod
    def predict(self, data, params=None):
        pass
    
    @abstractmethod
    def evaluate(self, data_with_label, params=None):
        pass
    
    @abstractmethod
    def load(self, path):
        pass
    
    def _set_output_base_dir(self, output_base_dir):
        self._output_base_dir = output_base_dir
    
    def get_output_folders(self):
        train_folder = os.path.join(self._output_base_dir, "train")
        infer_folder = os.path.join(self._output_base_dir, "predict")
        return [train_folder, infer_folder]

    def get_label_name(self):
        return self._label_column

class AutogluonAnalyzer(TableAnalyzer):
    def __init__(self, config=None):
        super(AutogluonAnalyzer, self).__init__(config)
    
    def train(self, train_data, val_data, params):
        train_dataset = TabularPrediction.Dataset(train_data)
        val_dataset   = TabularPrediction.Dataset(val_data)
        output_dir    = os.path.join(self.get_output_folders()[0], dt.now().strftime('%Y%m%d%H%M%S'))
        hp_tune       = params["hp_tune"]
        ag_params     = params["autogluon"]
        self._label_column = params["label"]
        
        if hp_tune is True:
            hp_params       = ag_params["hyperparameters"]
            time_limits     = hp_params["time_limits"]
            num_trials      = hp_params["num_trials"]
            hyperparameters = self.__create_hp_params(hp_params)
            search_strategy = hp_params["search_strategy"]
            self._model = TabularPrediction.fit(
                train_data=train_dataset, tuning_data=val_dataset, label=self._label_column,
                output_directory=output_dir, time_limits=time_limits, 
                num_trials=num_trials, hyperparameter_tune=hp_tune, 
                hyperparameters=hyperparameters, search_strategy=search_strategy
            )
        else:
            self._model = TabularPrediction.fit(
                train_data=train_dataset, tuning_data=val_dataset, label=self._label_column,
                output_directory=output_dir
            )
        
        self.__dump_params(output_dir, params)
        
        self._model.fit_summary()
    
    def predict(self, data, params=None):
        dataset = TabularPrediction.Dataset(data)
        if self._label_column in dataset.columns:
            dataset = dataset.drop(labels=[self._label_column], axis=1)
        return self._model.predict(dataset)
    
    def evaluate(self, data_with_label, params=None):
        # TODO : imp
        pass
    
    def load(self, path):
        # load model
        self._model = TabularPrediction.load(path)
        
        # get the column name of label
        with open(os.path.join(path, "params.json"), "r") as f:
            params = json.load(f)
            self._label_column = params["label"]
    
    def _set_output_base_dir(self, output_base_dir):
        self._output_base_dir = os.path.join(output_base_dir, "autogluon")
    
    def __create_hp_params(self, hp_params):
        if hp_params is None:
            return None
        return {"NN": hp_params["NN"], "GBM": hp_params["GBM"]}

    def __dump_params(self, path, params):
        output_params = {}
        output_params["label"] = params["label"]
        with open(os.path.join(path, "params.json"), 'w') as f:
            json.dump(output_params, f, indent=2)

class XGBoostAnalyzer(TableAnalyzer):
    def __init__(self, config=None):
        # TODO : imp
        pass
    
    def train(self, train_data, val_data=None, params=None):
        # TODO : imp
        pass
    
    def predict(self, data, params=None):
        # TODO : imp
        pass
    
    def load(self, path):
        pass
    
    def _set_output_base_dir(self, output_base_dir):
        self._output_base_dir = os.path.join(output_base_dir, "xgboost")
