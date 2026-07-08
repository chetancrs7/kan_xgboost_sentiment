"""Optional TensorBoard logging of KAN spline diagnostics during training.

Design contract: **logging must never break training.** Every public method
is a no-op when the logger is disabled, and any internal exception (missing
tensorboard install, a figure that fails to render, a full disk) prints one
warning and permanently disables the logger instead of propagating.

What gets logged:

* per-epoch scalars — train/val loss and accuracy (``kan/…``);
* per-epoch histograms — spline coefficient and scale distributions per
  layer (watch coefficients drift away from their near-zero init as the
  splines learn shape);
* per-epoch scalars — min/max of each layer's adapted knot grid (jumps in
  these curves mark the periodic ``update_grid_from_samples`` refits);
* every ``figure_every`` epochs — the strongest learned edge-function
  curves per layer as images, so spline shapes can be watched evolving
  live in TensorBoard's Images tab.

View with::

    tensorboard --logdir outputs/tensorboard/kan
"""

from __future__ import annotations

from pathlib import Path


class SplineTensorBoardLogger:
    """Failure-proof TensorBoard writer for KAN spline diagnostics.

    Args:
        log_dir: TensorBoard event-file directory.
        enabled: master switch; when ``False`` every method is a cheap no-op.
        figure_every: epoch interval for the (relatively expensive) spline
            curve figures; histograms/scalars are logged every epoch.
        top_k: number of strongest edges per layer in the curve figures.
    """

    def __init__(
        self,
        log_dir: str | Path,
        enabled: bool = True,
        figure_every: int = 10,
        top_k: int = 4,
    ) -> None:
        self.enabled = bool(enabled)
        self.figure_every = max(1, int(figure_every))
        self.top_k = int(top_k)
        self._writer = None
        if not self.enabled:
            return
        try:
            from torch.utils.tensorboard import SummaryWriter

            self._writer = SummaryWriter(log_dir=str(log_dir))
            print(f"[INFO] TensorBoard spline logging enabled: {Path(log_dir).resolve()}")
        except Exception as exc:  # noqa: BLE001 - by contract, never raise
            self._disable(f"could not create SummaryWriter ({exc})")

    def _disable(self, reason: str) -> None:
        print(f"[WARN] TensorBoard spline logging disabled: {reason}. Training continues.")
        self.enabled = False
        self._writer = None

    # ------------------------------------------------------------------
    # Public logging API (all failure-proof)
    # ------------------------------------------------------------------

    def log_epoch_metrics(self, epoch: int, metrics: dict[str, float]) -> None:
        """Log scalar training metrics (loss/accuracy) under ``kan/<name>``."""
        if not self.enabled:
            return
        try:
            for name, value in metrics.items():
                self._writer.add_scalar(f"kan/{name}", float(value), epoch)
        except Exception as exc:  # noqa: BLE001
            self._disable(f"scalar logging failed ({exc})")

    def log_spline_state(self, model, epoch: int) -> None:
        """Log spline internals: coefficient/scale histograms, grid ranges, curves."""
        if not self.enabled:
            return
        try:
            # Imported lazily so a broken analysis module can only ever
            # disable logging, never crash the training process at import.
            from src.analysis.spline_extractor import iter_kan_layers

            for name, _wrapper, kan_layer in iter_kan_layers(model):
                if kan_layer is None:  # non-B-spline layer in a mixed-spline model
                    continue
                self._writer.add_histogram(
                    f"spline_coefficients/{name}", kan_layer.coef.detach().cpu(), epoch
                )
                self._writer.add_histogram(
                    f"spline_scale_sp/{name}", kan_layer.scale_sp.detach().cpu(), epoch
                )
                self._writer.add_histogram(
                    f"spline_scale_base/{name}", kan_layer.scale_base.detach().cpu(), epoch
                )
                grid = kan_layer.grid.detach()
                self._writer.add_scalar(f"knot_grid_min/{name}", grid.min().item(), epoch)
                self._writer.add_scalar(f"knot_grid_max/{name}", grid.max().item(), epoch)

            if epoch == 1 or epoch % self.figure_every == 0:
                self._log_curve_figures(model, epoch)
        except Exception as exc:  # noqa: BLE001
            self._disable(f"spline-state logging failed ({exc})")

    def _log_curve_figures(self, model, epoch: int) -> None:
        """Render the top edge-function curves of each layer as TB images."""
        import matplotlib.pyplot as plt

        from src.analysis.spline_extractor import LAYER_TITLES, iter_kan_layers, top_edges
        from src.analysis.spline_visualizer import PUBLICATION_RC, edge_curves_figure

        for name, _wrapper, kan_layer in iter_kan_layers(model):
            if kan_layer is None:  # non-B-spline layer in a mixed-spline model
                continue
            edges = top_edges(kan_layer, k=self.top_k)
            with plt.rc_context(PUBLICATION_RC):
                fig = edge_curves_figure(kan_layer, LAYER_TITLES[name], edges)
            self._writer.add_figure(f"spline_curves/{name}", fig, epoch)
            plt.close(fig)

    def close(self) -> None:
        """Flush and release the writer (safe to call multiple times)."""
        if self._writer is None:
            return
        try:
            self._writer.close()
        except Exception:  # noqa: BLE001
            pass
        self._writer = None
        self.enabled = False
