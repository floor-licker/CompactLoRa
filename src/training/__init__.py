"""
LLM Training Module for Compact Smart Contracts

This module implements compiler-in-the-loop training for generating
valid Compact smart contracts using the Compact compiler as a fitness evaluator.
"""

from .training_pipeline import CompactTrainingPipeline
from .model_wrapper import CompactCodeModel
from .training_data import CompactTrainingData
from .evaluation import CompactEvaluator

__all__ = [
    'CompactTrainingPipeline',
    'CompactCodeModel', 
    'CompactTrainingData',
    'CompactEvaluator'
] 