"""Prefect flow entry points."""

from .update_data_flow import update_data_flow
from .train_model_flow import train_model_flow
from .predict_flow import predict_flow

__all__ = ["update_data_flow", "train_model_flow", "predict_flow"]
