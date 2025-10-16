import uuid

import pandas as pd
from a4s_eval.metric_registries.model_metric_registry import model_metric_registry
from a4s_eval.metric_registries.model_metric_registry import ModelMetric
from a4s_eval.service.model_functional import FunctionalModel
from a4s_eval.service.model_load import load_model
import pytest

from a4s_eval.data_model.evaluation import (
    Dataset,
    DataShape,
    Model,
    ModelConfig,
    ModelFramework,
)

from tests.save_measures_utils import save_measures


@pytest.fixture
def data_shape() -> DataShape:
    metadata = pd.read_csv("tests/data/lcld_v2_metadata_api.csv").to_dict(
        orient="records"
    )

    for record in metadata:
        record["pid"] = uuid.uuid4()

    data_shape = {
        "features": [
            item
            for item in metadata
            if item.get("name") not in ["charged_off", "issue_d"]
        ],
        "target": next(rec for rec in metadata if rec.get("name") == "charged_off"),
        "date": next(rec for rec in metadata if rec.get("name") == "issue_d"),
    }

    return DataShape.model_validate(data_shape)


@pytest.fixture
def test_dataset(test_data: pd.DataFrame, data_shape: DataShape) -> Dataset:
    data = test_data
    data["issue_d"] = pd.to_datetime(data["issue_d"])
    return Dataset(pid=uuid.uuid4(), shape=data_shape, data=data)


@pytest.fixture
def ref_dataset(train_data, data_shape: DataShape) -> Dataset:
    data = train_data
    data["issue_d"] = pd.to_datetime(data["issue_d"])
    return Dataset(
        pid=uuid.uuid4(),
        shape=data_shape,
        data=data,
    )


@pytest.fixture
def ref_model(ref_dataset: Dataset) -> Model:
    return Model(
        pid=uuid.uuid4(),
        model=None,
        dataset=ref_dataset,
    )


@pytest.fixture
def functional_model() -> FunctionalModel:
    model_config = ModelConfig(
        path="./tests/data/lcld_v2_tabtransformer.pt", framework=ModelFramework.TORCH
    )
    return load_model(model_config)


def test_non_empty_registry():
    assert len(model_metric_registry._functions) > 0


@pytest.mark.parametrize("evaluator_function", model_metric_registry)
def test_data_metric_registry_contains_evaluator(
    evaluator_function: tuple[str, ModelMetric],
    data_shape: DataShape,
    ref_model: Model,
    test_dataset: Dataset,
    functional_model: FunctionalModel,
):
    measures = evaluator_function[1](
        data_shape, ref_model, test_dataset, functional_model
    )
    save_measures(evaluator_function[0], measures)
    assert len(measures) > 0

def test_accuracy_metric_in_batches(
    data_shape: DataShape,
    ref_model: Model,
    test_dataset: Dataset,
    functional_model: FunctionalModel,
):
    """
    Duplicate of test_data_metric_registry_contains_evaluator but specifically tests
    the 'accuracy' metric in batches of 10,000 rows.
    """

    evaluator_function = next(
        (f for f in model_metric_registry if f[0] == "accuracy"), None
    )
    assert evaluator_function is not None, "Accuracy metric not found in registry"

    original_data = test_dataset.data
    batch_size = 10000
    batched_measures = []

    for start in range(0, len(original_data), batch_size):
        end = start + batch_size
        test_dataset.data = original_data.iloc[start:end]

        measures = evaluator_function[1](
            data_shape, ref_model, test_dataset, functional_model
        )
        batched_measures.extend(measures)

    save_measures("accuracy", batched_measures)

    assert len(batched_measures) > 0
    for measure in batched_measures:
        assert 0.0 <= measure.score <= 1.0
        
        
