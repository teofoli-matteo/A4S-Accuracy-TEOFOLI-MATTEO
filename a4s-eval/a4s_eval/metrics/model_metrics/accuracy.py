from datetime import datetime
import numpy as np
import pandas as pd
import torch

from a4s_eval.data_model.evaluation import DataShape, Dataset, Model
from a4s_eval.data_model.measure import Measure
from a4s_eval.metric_registries.model_metric_registry import model_metric
from a4s_eval.service.model_functional import FunctionalModel


@model_metric(name="accuracy")
def accuracy(
    datashape: DataShape,
    model: Model,
    dataset: Dataset,
    functional_model: FunctionalModel,
) -> list[Measure]:
    target_col = datashape.target.name
    feature_cols = [f.name for f in datashape.features]

    df = dataset.data.head(10000)  

    x = df[feature_cols].to_numpy().astype(np.float32) 
    y_true = df[target_col].to_numpy()

 
    x_tensor = torch.from_numpy(x)  

    y_pred = functional_model.predict(x_tensor)  
    y_pred = np.squeeze(y_pred) 

    
    correct = np.sum(y_true == y_pred)
    total = len(y_true)
    accuracy_value = correct / total if total > 0 else 0

    return [Measure(name="accuracy", score=accuracy_value, time=datetime.now())]
