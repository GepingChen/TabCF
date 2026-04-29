"""
Local context helpers for Stage-2 backend adaptation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional

import numpy as np
from scipy.spatial import cKDTree


GLOBAL_LOCAL_STRATEGY = "global"
LOCAL_KNN_STRATEGY = "local_knn"
SUPPORTED_LOCAL_STRATEGIES = (GLOBAL_LOCAL_STRATEGY, LOCAL_KNN_STRATEGY)
SUPPORTED_LOCAL_METRICS = ("euclidean",)


@dataclass(frozen=True)
class LocalContextConfig:
    strategy: str = GLOBAL_LOCAL_STRATEGY
    k_neighbors: int | None = None
    metric: str = "euclidean"
    scale_features: bool = True

    def __post_init__(self) -> None:
        strategy = str(self.strategy or GLOBAL_LOCAL_STRATEGY).strip().lower()
        metric = str(self.metric or "euclidean").strip().lower()
        if strategy not in SUPPORTED_LOCAL_STRATEGIES:
            raise ValueError(
                f"Unsupported local-context strategy '{self.strategy}'. Expected one of {SUPPORTED_LOCAL_STRATEGIES}."
            )
        if metric not in SUPPORTED_LOCAL_METRICS:
            raise ValueError(
                f"Unsupported local-context metric '{self.metric}'. Expected one of {SUPPORTED_LOCAL_METRICS}."
            )
        if self.k_neighbors is not None and int(self.k_neighbors) <= 0:
            raise ValueError("k_neighbors must be positive when provided.")

        object.__setattr__(self, "strategy", strategy)
        object.__setattr__(self, "metric", metric)
        if self.k_neighbors is not None:
            object.__setattr__(self, "k_neighbors", int(self.k_neighbors))
        object.__setattr__(self, "scale_features", bool(self.scale_features))


def ensure_local_context_config(local_context: object | None) -> LocalContextConfig:
    if local_context is None:
        return LocalContextConfig()
    if isinstance(local_context, LocalContextConfig):
        return local_context
    if isinstance(local_context, Mapping):
        return LocalContextConfig(**dict(local_context))
    if all(hasattr(local_context, attr) for attr in ("strategy", "k_neighbors", "metric", "scale_features")):
        return LocalContextConfig(
            strategy=str(getattr(local_context, "strategy")),
            k_neighbors=getattr(local_context, "k_neighbors"),
            metric=str(getattr(local_context, "metric")),
            scale_features=bool(getattr(local_context, "scale_features")),
        )
    raise TypeError(
        f"local_context must be None, LocalContextConfig, or a mapping; got {type(local_context).__name__}."
    )


def resolve_local_k_neighbors(local_context: object | None, n_train_samples: int) -> int | None:
    cfg = ensure_local_context_config(local_context)
    if cfg.strategy == GLOBAL_LOCAL_STRATEGY:
        return None

    n_train = int(n_train_samples)
    if n_train <= 0:
        raise ValueError("n_train_samples must be positive to resolve local_knn neighbors.")

    if cfg.k_neighbors is None:
        resolved = min(int(10.0 * np.sqrt(n_train)), 1000, n_train)
    else:
        resolved = min(int(cfg.k_neighbors), n_train)
    return max(1, int(resolved))


def local_context_suffix(
    local_context: object | None,
    *,
    n_train_samples: int | None = None,
    resolved_k: int | None = None,
) -> str:
    cfg = ensure_local_context_config(local_context)
    if cfg.strategy == GLOBAL_LOCAL_STRATEGY:
        return ""

    resolved = resolved_k
    if resolved is None:
        if n_train_samples is None:
            raise ValueError("n_train_samples is required to build a local_knn filename suffix.")
        resolved = resolve_local_k_neighbors(cfg, int(n_train_samples))
    return f"_lknnx{int(resolved)}"


def local_context_metadata(
    local_context: object | None,
    *,
    n_train_samples: int,
    resolved_k: int | None = None,
) -> dict[str, object]:
    cfg = ensure_local_context_config(local_context)
    return {
        "local_strategy": cfg.strategy,
        "local_k_neighbors_requested": cfg.k_neighbors,
        "local_k_neighbors_resolved": (
            resolved_k
            if resolved_k is not None
            else resolve_local_k_neighbors(cfg, n_train_samples)
        ),
        "local_retrieval_features": "x_only" if cfg.strategy == LOCAL_KNN_STRATEGY else "none",
        "local_metric": cfg.metric,
        "local_scale_features": cfg.scale_features,
    }


def is_local_knn_config(local_context: object | None) -> bool:
    return ensure_local_context_config(local_context).strategy == LOCAL_KNN_STRATEGY


class GroupedLocalDistributionAdapter:
    """Combine per-neighborhood predictive distributions into query order."""

    def __init__(self, *, n_samples: int, group_indices: list[np.ndarray], group_adapters: list[object]) -> None:
        self.n_samples = int(n_samples)
        self.group_indices = [np.asarray(indices, dtype=int) for indices in group_indices]
        self.group_adapters = list(group_adapters)
        if len(self.group_indices) != len(self.group_adapters):
            raise ValueError("group_indices and group_adapters must have the same length.")

    def _subset_values(self, values: np.ndarray, indices: np.ndarray) -> np.ndarray:
        arr = np.asarray(values, dtype=float)
        if arr.ndim == 0:
            return arr
        if arr.ndim == 1:
            if arr.size == self.n_samples:
                return arr[indices]
            return arr
        if arr.ndim == 2 and arr.shape[0] == self.n_samples:
            return arr[indices]
        raise ValueError(
            f"Incompatible evaluation shape {arr.shape}; expected scalar, (n,), or (n, m) with n={self.n_samples}."
        )

    def _merge_group_results(self, group_outputs: list[tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
        first_indices, first_output = group_outputs[0]
        first_arr = np.asarray(first_output, dtype=float)
        if first_arr.ndim == 0:
            merged = np.empty((self.n_samples,), dtype=float)
            merged[first_indices] = float(first_arr)
            for indices, output in group_outputs[1:]:
                merged[indices] = float(np.asarray(output, dtype=float))
            return merged

        merged_shape = (self.n_samples,) + tuple(first_arr.shape[1:])
        merged = np.empty(merged_shape, dtype=float)
        merged[first_indices] = first_arr
        for indices, output in group_outputs[1:]:
            merged[indices] = np.asarray(output, dtype=float)
        return merged

    def _dispatch_with_values(self, method_name: str, values: np.ndarray) -> np.ndarray:
        outputs: list[tuple[np.ndarray, np.ndarray]] = []
        for indices, adapter in zip(self.group_indices, self.group_adapters):
            subset_values = self._subset_values(values, indices)
            outputs.append((indices, np.asarray(getattr(adapter, method_name)(subset_values), dtype=float)))
        return self._merge_group_results(outputs)

    def _dispatch_no_values(self, method_name: str) -> np.ndarray:
        outputs = [
            (indices, np.asarray(getattr(adapter, method_name)(), dtype=float))
            for indices, adapter in zip(self.group_indices, self.group_adapters)
        ]
        return self._merge_group_results(outputs)

    def cdf(self, values: np.ndarray) -> np.ndarray:
        return self._dispatch_with_values("cdf", values)

    def pdf(self, values: np.ndarray) -> np.ndarray:
        return self._dispatch_with_values("pdf", values)

    def icdf(self, alphas: np.ndarray) -> np.ndarray:
        return self._dispatch_with_values("icdf", alphas)

    def mean(self) -> np.ndarray:
        return self._dispatch_no_values("mean")

    def variance(self) -> np.ndarray:
        return self._dispatch_no_values("variance")


class LocalKNNRegressorWrapper:
    """Fit backend regressors on query-local neighborhoods."""

    is_local_context_wrapper_ = True

    def __init__(
        self,
        *,
        backend_name: str,
        local_context: object | None,
        base_estimator_factory: Callable[[], Any],
    ) -> None:
        self.backend_name = str(backend_name)
        self.local_context = ensure_local_context_config(local_context)
        self.base_estimator_factory = base_estimator_factory
        self.X_train_: np.ndarray | None = None
        self.y_train_: np.ndarray | None = None
        self.resolved_k_: int | None = None
        self.retrieval_feature_mean_: np.ndarray | None = None
        self.retrieval_feature_scale_: np.ndarray | None = None
        self.tree_: cKDTree | None = None
        self._estimator_cache: dict[tuple[int, ...], Any] = {}

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LocalKNNRegressorWrapper":
        X_train = np.asarray(X, dtype=float)
        if X_train.ndim != 2:
            raise ValueError(f"Expected 2D training features, got shape {X_train.shape}.")

        y_train = np.asarray(y, dtype=float).reshape(-1)
        if len(X_train) != len(y_train):
            raise ValueError(
                f"Training feature/target length mismatch: {len(X_train)} features vs {len(y_train)} targets."
            )
        if len(X_train) == 0:
            raise ValueError("Cannot fit LocalKNNRegressorWrapper on an empty training set.")

        self.X_train_ = X_train
        self.y_train_ = y_train
        self.resolved_k_ = resolve_local_k_neighbors(self.local_context, len(X_train))
        retrieval_X = self._retrieval_view(X_train)
        if self.local_context.scale_features:
            self.retrieval_feature_mean_ = retrieval_X.mean(axis=0)
            self.retrieval_feature_scale_ = retrieval_X.std(axis=0)
            self.retrieval_feature_scale_[self.retrieval_feature_scale_ == 0.0] = 1.0
        else:
            self.retrieval_feature_mean_ = np.zeros(retrieval_X.shape[1], dtype=float)
            self.retrieval_feature_scale_ = np.ones(retrieval_X.shape[1], dtype=float)

        retrieval_X = self._transform_features(X_train)
        if self.resolved_k_ is not None and self.resolved_k_ < len(X_train):
            self.tree_ = cKDTree(retrieval_X)
        else:
            self.tree_ = None
        self._estimator_cache = {}
        return self

    def _retrieval_view(self, X: np.ndarray) -> np.ndarray:
        features = np.asarray(X, dtype=float)
        if features.ndim == 1:
            features = features.reshape(1, -1)
        if features.shape[1] == 0:
            raise ValueError("At least one feature column is required for local kNN retrieval.")
        # local_knn now selects neighborhoods using x only while local model fitting
        # continues to use the full feature slice [x, v, ...].
        return features[:, :1]

    def _transform_features(self, X: np.ndarray) -> np.ndarray:
        features = self._retrieval_view(X)
        if self.retrieval_feature_mean_ is None or self.retrieval_feature_scale_ is None:
            raise RuntimeError("LocalKNNRegressorWrapper must be fitted before transforming features.")
        return (features - self.retrieval_feature_mean_) / self.retrieval_feature_scale_

    def _require_fitted_data(self) -> tuple[np.ndarray, np.ndarray]:
        if self.X_train_ is None or self.y_train_ is None or self.resolved_k_ is None:
            raise RuntimeError("LocalKNNRegressorWrapper must be fitted before prediction.")
        return self.X_train_, self.y_train_

    def _neighbor_matrix(self, X_query: np.ndarray) -> np.ndarray:
        X_train, _ = self._require_fitted_data()
        n_queries = len(X_query)
        if self.resolved_k_ is None:
            raise RuntimeError("resolved_k_ is unavailable before fitting.")
        if self.resolved_k_ >= len(X_train):
            return np.tile(np.arange(len(X_train), dtype=int), (n_queries, 1))

        if self.tree_ is None:
            raise RuntimeError("KD-tree is unavailable despite a strict local neighborhood size.")
        _, indices = self.tree_.query(self._transform_features(X_query), k=self.resolved_k_)
        indices = np.asarray(indices, dtype=int)
        if indices.ndim == 1:
            indices = indices.reshape(-1, 1)
        return np.sort(indices, axis=1)

    def _query_groups(self, X_query: np.ndarray) -> list[tuple[tuple[int, ...], np.ndarray]]:
        neighbor_matrix = self._neighbor_matrix(X_query)
        grouped: dict[tuple[int, ...], list[int]] = {}
        for row_idx, neighbors in enumerate(neighbor_matrix):
            grouped.setdefault(tuple(int(idx) for idx in neighbors.tolist()), []).append(row_idx)
        return [(key, np.asarray(row_indices, dtype=int)) for key, row_indices in grouped.items()]

    def _fit_estimator_for_group(self, neighbor_key: tuple[int, ...]) -> Any:
        X_train, y_train = self._require_fitted_data()
        neighbor_idx = np.asarray(neighbor_key, dtype=int)
        estimator = self.base_estimator_factory()
        estimator.fit(X_train[neighbor_idx], y_train[neighbor_idx])
        return estimator

    def predict(
        self,
        X: np.ndarray,
        *,
        output_type: str = "mean",
        quantiles: Optional[list[float]] = None,
        alphas: Optional[list[float]] = None,
    ) -> Any:
        X_query = np.asarray(X, dtype=float)
        if X_query.ndim == 1:
            X_query = X_query.reshape(1, -1)
        if X_query.ndim != 2:
            raise ValueError(f"Expected 2D query features, got shape {X_query.shape}.")

        # Keep local-kNN memory bounded by fitting each neighborhood model on demand
        # for the current prediction only. Retaining fitted foundation models across
        # predict calls leads to unbounded cache growth over large evaluation grids.
        self._estimator_cache = {}
        groups = self._query_groups(X_query)
        n_queries = len(X_query)

        if output_type == "mean":
            merged = np.empty((n_queries,), dtype=float)
            for neighbor_key, query_indices in groups:
                estimator = self._fit_estimator_for_group(neighbor_key)
                group_pred = np.asarray(estimator.predict(X_query[query_indices], output_type="mean"), dtype=float).reshape(-1)
                merged[query_indices] = group_pred
            return merged

        if output_type == "quantiles":
            levels = list(quantiles if quantiles is not None else alphas if alphas is not None else [])
            if not levels:
                raise ValueError("Quantile prediction requires quantiles/alphas.")
            merged: np.ndarray | None = None
            for neighbor_key, query_indices in groups:
                estimator = self._fit_estimator_for_group(neighbor_key)
                if self.backend_name in {"tabpfn", "tabpfn_real"}:
                    raw = estimator.predict(X_query[query_indices], output_type="quantiles", quantiles=levels)
                    if isinstance(raw, list):
                        group_pred = np.column_stack([np.asarray(arr, dtype=float).reshape(-1) for arr in raw])
                    else:
                        group_pred = np.asarray(raw, dtype=float)
                else:
                    group_pred = np.asarray(
                        estimator.predict(X_query[query_indices], output_type="quantiles", alphas=levels),
                        dtype=float,
                    )
                if merged is None:
                    merged = np.empty((n_queries, group_pred.shape[1]), dtype=float)
                merged[query_indices] = group_pred
            if merged is None:
                return np.empty((0, len(levels)), dtype=float)
            return merged

        if output_type == "raw_quantiles":
            merged: np.ndarray | None = None
            for neighbor_key, query_indices in groups:
                estimator = self._fit_estimator_for_group(neighbor_key)
                group_pred = np.asarray(estimator.predict(X_query[query_indices], output_type="raw_quantiles"), dtype=float)
                if merged is None:
                    merged = np.empty((n_queries, group_pred.shape[1]), dtype=float)
                merged[query_indices] = group_pred
            if merged is None:
                return np.empty((0, 0), dtype=float)
            return merged

        if output_type == "full":
            import foundation_backends as fb

            group_indices: list[np.ndarray] = []
            group_adapters: list[object] = []
            for neighbor_key, query_indices in groups:
                estimator = self._fit_estimator_for_group(neighbor_key)
                if self.backend_name in {"tabpfn", "tabpfn_real"}:
                    group_output = estimator.predict(
                        X_query[query_indices],
                        output_type="full",
                        quantiles=list(quantiles or []),
                    )
                else:
                    raw_quantiles = np.asarray(
                        estimator.predict(X_query[query_indices], output_type="raw_quantiles"),
                        dtype=float,
                    )
                    group_output = {"distribution": fb.TabICLDistributionAdapter(raw_quantiles=raw_quantiles)}
                group_indices.append(query_indices)
                group_adapters.append(fb.distribution_adapter_from_output(group_output))

            return {
                "backend_name": self.backend_name,
                "distribution": GroupedLocalDistributionAdapter(
                    n_samples=n_queries,
                    group_indices=group_indices,
                    group_adapters=group_adapters,
                ),
                "local_strategy": self.local_context.strategy,
                "local_k_neighbors_resolved": self.resolved_k_,
            }

        raise ValueError(f"Unsupported output_type '{output_type}' for LocalKNNRegressorWrapper.")
