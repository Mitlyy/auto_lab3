from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from .models import (
    ClassifierNet,
    FParamSpec,
    HyperNet,
    flatten_classifier_params,
    functional_mlp_forward,
    load_classifier_params_from_flat,
)


@dataclass
class TaskTensors:
    dataset_id: int
    dataset_name: str
    X_train: torch.Tensor
    y_train: torch.Tensor
    X_test: torch.Tensor
    y_test: torch.Tensor
    meta: torch.Tensor


@dataclass
class ClassifierTrainingResult:
    dataset_id: int
    dataset_name: str
    curve: list[float]
    final_balanced_accuracy: float
    theta: torch.Tensor



def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)



def get_device(preferred: str) -> torch.device:
    if preferred == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(preferred)



def balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> float:
    recalls: list[float] = []
    for class_id in range(n_classes):
        class_mask = y_true == class_id
        if not np.any(class_mask):
            recalls.append(0.0)
            continue
        recalls.append(float(np.mean(y_pred[class_mask] == class_id)))
    return float(np.mean(recalls))



def _evaluate_model(model: ClassifierNet, X_test: torch.Tensor, y_test: torch.Tensor, n_classes: int) -> float:
    model.eval()
    with torch.no_grad():
        pred = model(X_test).argmax(dim=1)
    return balanced_accuracy(
        y_true=y_test.detach().cpu().numpy(),
        y_pred=pred.detach().cpu().numpy(),
        n_classes=n_classes,
    )



def _evaluate_theta(
    theta: torch.Tensor,
    X_test: torch.Tensor,
    y_test: torch.Tensor,
    spec: FParamSpec,
) -> float:
    with torch.no_grad():
        pred = functional_mlp_forward(X_test, theta, spec).argmax(dim=1)
    return balanced_accuracy(
        y_true=y_test.detach().cpu().numpy(),
        y_pred=pred.detach().cpu().numpy(),
        n_classes=spec.output_dim,
    )



def train_classifier_on_task(
    task: TaskTensors,
    spec: FParamSpec,
    epochs: int,
    lr: float,
    weight_decay: float,
    seed: int,
    init_theta: torch.Tensor | None = None,
) -> ClassifierTrainingResult:
    torch.manual_seed(seed)

    model = ClassifierNet(spec).to(task.X_train.device)
    if init_theta is not None:
        load_classifier_params_from_flat(model, init_theta.to(task.X_train.device))

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss()

    curve: list[float] = [_evaluate_model(model, task.X_test, task.y_test, spec.output_dim)]

    for _ in range(epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(task.X_train)
        loss = loss_fn(logits, task.y_train)
        loss.backward()
        optimizer.step()

        curve.append(_evaluate_model(model, task.X_test, task.y_test, spec.output_dim))

    theta = flatten_classifier_params(model).detach().cpu()
    return ClassifierTrainingResult(
        dataset_id=task.dataset_id,
        dataset_name=task.dataset_name,
        curve=curve,
        final_balanced_accuracy=float(curve[-1]),
        theta=theta,
    )



def train_hypernet_supervised(
    metas: torch.Tensor,
    target_thetas: torch.Tensor,
    meta_dim: int,
    param_dim: int,
    hidden_dim: int,
    epochs: int,
    lr: float,
    device: torch.device,
) -> tuple[HyperNet, dict[str, list[float]]]:
    hyper = HyperNet(meta_dim=meta_dim, output_dim=param_dim, hidden_dim=hidden_dim).to(device)
    optimizer = torch.optim.Adam(hyper.parameters(), lr=lr)
    mse = nn.MSELoss()

    history = {"loss": [], "cosine": []}

    for _ in range(epochs):
        hyper.train()
        optimizer.zero_grad(set_to_none=True)
        pred = hyper(metas)
        loss = mse(pred, target_thetas)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            cosine = torch.nn.functional.cosine_similarity(pred, target_thetas, dim=1).mean()
        history["loss"].append(float(loss.detach().cpu()))
        history["cosine"].append(float(cosine.detach().cpu()))

    return hyper, history



def evaluate_hypernet_zero_shot(
    hyper: HyperNet,
    tasks: list[TaskTensors],
    spec: FParamSpec,
) -> list[float]:
    scores: list[float] = []
    hyper.eval()
    with torch.no_grad():
        for task in tasks:
            theta = hyper(task.meta.unsqueeze(0)).squeeze(0)
            score = _evaluate_theta(theta=theta, X_test=task.X_test, y_test=task.y_test, spec=spec)
            scores.append(score)
    return scores



def train_hypernet_dynamic(
    hyper_start: HyperNet,
    tasks: list[TaskTensors],
    spec: FParamSpec,
    steps: int,
    lr: float,
    seed: int,
) -> tuple[HyperNet, dict[str, list[float]]]:
    hyper = copy.deepcopy(hyper_start)
    hyper.train()

    optimizer = torch.optim.Adam(hyper.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    rng = np.random.default_rng(seed)

    history = {"train_loss": [], "balanced_accuracy": [], "dataset_id": []}

    for _ in range(steps):
        idx = int(rng.integers(0, len(tasks)))
        task = tasks[idx]

        optimizer.zero_grad(set_to_none=True)
        theta = hyper(task.meta.unsqueeze(0)).squeeze(0)
        logits_train = functional_mlp_forward(task.X_train, theta, spec)
        loss = loss_fn(logits_train, task.y_train)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            theta_eval = hyper(task.meta.unsqueeze(0)).squeeze(0)
            score = _evaluate_theta(theta=theta_eval, X_test=task.X_test, y_test=task.y_test, spec=spec)

        history["train_loss"].append(float(loss.detach().cpu()))
        history["balanced_accuracy"].append(float(score))
        history["dataset_id"].append(int(task.dataset_id))

    return hyper, history
