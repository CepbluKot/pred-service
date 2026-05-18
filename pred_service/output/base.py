"""Abstract base class for all output sinks."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pred_service.pipeline import PredictionResult


class OutputSink(ABC):
    """Write forecast results to some destination."""

    @abstractmethod
    def write(self, result: "PredictionResult") -> None:
        """
        Write a PredictionResult to this sink.

        Parameters
        ----------
        result:
            The completed prediction result to persist or display.
        """
