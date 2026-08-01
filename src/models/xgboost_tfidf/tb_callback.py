"""Optional TensorBoard logging of XGBoost train/val curves during boosting.

Design contract (mirrors ``src/analysis/tensorboard_logger.py``): **logging must
never break training.** Creating the writer, every per-round write, and close
are wrapped so that a missing tensorboard install, a full disk, or any other
failure prints one warning and permanently disables logging instead of
propagating into the boosting loop.

What gets logged, once per boosting round (the XGBoost "iteration"):

* ``xgb/<split>_logloss`` — the binary log-loss XGBoost already computes;
* ``xgb/<split>_accuracy`` — ``1 - error``, derived from the ``error`` eval
  metric (0.5-threshold classification error), so the live accuracy curve the
  KAN model shows has an XGBoost equivalent.

``<split>`` is ``train`` / ``validation``, resolved from the position of each
dataset in ``eval_set`` (``validation_0`` -> first, ``validation_1`` -> second).

View with::

    tensorboard --logdir outputs/tensorboard/xgboost_tfidf
"""

from __future__ import annotations

from pathlib import Path

import xgboost as xgb


class TensorBoardCallback(xgb.callback.TrainingCallback):
    """Failure-proof TensorBoard writer for XGBoost train/val scalars.

    Args:
        log_dir: TensorBoard event-file directory.
        eval_names: maps XGBoost's positional dataset keys (``validation_0``,
            ``validation_1``, ...) to friendly split names (``train``,
            ``validation``). Any key not present falls back to itself.
    """

    def __init__(self, log_dir: str | Path, eval_names: dict[str, str]) -> None:
        self.eval_names = dict(eval_names)
        self._writer = None
        try:
            from torch.utils.tensorboard import SummaryWriter

            self._writer = SummaryWriter(log_dir=str(log_dir))
            print(f"[INFO] TensorBoard XGBoost logging enabled: {Path(log_dir).resolve()}")
        except Exception as exc:  # noqa: BLE001 - by contract, never raise
            self._disable(f"could not create SummaryWriter ({exc})")

    def _disable(self, reason: str) -> None:
        print(f"[WARN] TensorBoard XGBoost logging disabled: {reason}. Training continues.")
        self._writer = None

    def after_iteration(self, model, epoch: int, evals_log) -> bool:  # noqa: ANN001
        """Log the latest scalar for every (dataset, metric); never stop early.

        Returning ``False`` tells XGBoost to keep training — early stopping is
        handled independently by the built-in ``early_stopping_rounds`` logic,
        which this callback deliberately does not touch.
        """
        if self._writer is None:
            return False
        try:
            for data_name, metric_dict in evals_log.items():
                split = self.eval_names.get(data_name, data_name)
                for metric_name, history in metric_dict.items():
                    value = float(history[-1])  # current round is the last entry
                    self._writer.add_scalar(f"xgb/{split}_{metric_name}", value, epoch)
                    if metric_name == "error":
                        self._writer.add_scalar(
                            f"xgb/{split}_accuracy", 1.0 - value, epoch
                        )
            self._writer.flush()  # flush each round so curves update live
        except Exception as exc:  # noqa: BLE001
            self._disable(f"scalar logging failed ({exc})")
        return False

    def close(self) -> None:
        """Flush and release the writer (safe to call multiple times)."""
        if self._writer is None:
            return
        try:
            self._writer.close()
        except Exception:  # noqa: BLE001
            pass
        self._writer = None
