from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.mixture import GaussianMixture

from .data import make_loader


@dataclass
class ProteusConfig:
    batch_size: int = 128
    num_workers: int = 0
    epochs: int = 100
    lr: float = 1e-3
    gmm_threshold: float = 0.6
    pseudo_refresh_epochs: int = 5
    mmd_weight: float = 1.0
    entropy_weight: float = 1.0
    pseudo_weight: float = 1.0
    source_ce_weight: float = 1.0


def compute_gaussian_kernel(source, target):
    sample_count = int(source.size(0)) + int(target.size(0))
    combined = torch.cat([source, target], dim=0)
    l2_distance = ((combined.unsqueeze(0) - combined.unsqueeze(1)) ** 2).sum(2)
    denom = max(sample_count ** 2 - sample_count, 1)
    bandwidth = torch.sum(l2_distance) / denom
    bandwidth = torch.clamp(bandwidth, min=1e-5)
    return torch.exp(-l2_distance / (bandwidth + 1e-5))


def calculate_mmd_loss(source_features, target_features):
    batch_size = min(source_features.size(0), target_features.size(0))
    source_features = source_features[:batch_size]
    target_features = target_features[:batch_size]
    kernels = compute_gaussian_kernel(source_features, target_features)
    xx = kernels[:batch_size, :batch_size]
    yy = kernels[batch_size:, batch_size:]
    xy = kernels[:batch_size, batch_size:]
    yx = kernels[batch_size:, :batch_size]
    return torch.mean(xx + yy - xy - yx)


def compute_softmax_entropy(logits):
    return -(logits.softmax(1) * logits.log_softmax(1)).sum(1)


def _forward_logits_features(model, inputs):
    outputs = model(inputs)
    if not isinstance(outputs, (tuple, list)) or len(outputs) < 2:
        raise ValueError("Proteus requires model(inputs) to return (logits, features).")
    return outputs[0], outputs[1]


def evaluate(model, data_loader, device):
    model.eval()
    preds = []
    labels = []
    with torch.no_grad():
        for inputs, batch_labels in data_loader:
            inputs = inputs.to(device)
            logits, _features = _forward_logits_features(model, inputs)
            preds.append(torch.argmax(logits, dim=1).cpu().numpy())
            labels.append(batch_labels.numpy())
    preds = np.concatenate(preds)
    labels = np.concatenate(labels)
    accuracy = float(np.mean(preds == labels))
    return {"accuracy": accuracy}


def compute_gmm_pseudo_labels(model, target_X, batch_size, num_workers, device, threshold):
    target_y_dummy = torch.zeros((target_X.shape[0],), dtype=torch.int64)
    loader = make_loader(target_X, target_y_dummy, batch_size, False, num_workers, False)
    entropies = []
    predictions = []
    model.eval()
    with torch.no_grad():
        for inputs, _labels in loader:
            inputs = inputs.to(device)
            logits, _features = _forward_logits_features(model, inputs)
            entropies.append(compute_softmax_entropy(logits).cpu().numpy())
            predictions.append(torch.argmax(logits, dim=1).cpu().numpy())
    entropies = np.concatenate(entropies).astype(np.float32)
    predictions = np.concatenate(predictions).astype(np.int64)
    denom = float(entropies.max() - entropies.min())
    if denom <= 1e-12:
        normalized = np.zeros_like(entropies, dtype=np.float32)
    else:
        normalized = (entropies - entropies.min()) / denom
    if len(normalized) < 2:
        clean_probs = np.ones_like(normalized, dtype=np.float32)
    else:
        gmm = GaussianMixture(n_components=2, tol=1e-6)
        gmm.fit(normalized.reshape(-1, 1))
        low_uncertainty_index = int(np.argmin(gmm.means_.reshape(-1)))
        clean_probs = gmm.predict_proba(normalized.reshape(-1, 1))[:, low_uncertainty_index]
    keep = clean_probs >= float(threshold)
    if not np.any(keep):
        keep[int(np.argmax(clean_probs))] = True
    return keep, torch.tensor(predictions, dtype=torch.int64), entropies, clean_probs


def adapt_model(model, source_X, source_y, target_X, target_y=None, device="cpu", config=None):
    """Run Proteus unsupervised target adaptation.

    target_y is optional and is used only for reporting target accuracy.
    """
    config = config or ProteusConfig()
    device = torch.device(device)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    ce_loss = torch.nn.CrossEntropyLoss()

    source_loader = make_loader(
        source_X, source_y, config.batch_size, True, config.num_workers, True
    )
    target_dummy_y = torch.zeros((target_X.shape[0],), dtype=torch.int64)
    target_loader = make_loader(
        target_X, target_dummy_y, config.batch_size, True, config.num_workers, True
    )
    eval_loader = None
    if target_y is not None:
        eval_loader = make_loader(target_X, target_y, config.batch_size, False, config.num_workers, False)

    source_iter = iter(source_loader)
    pseudo_loader = None
    pseudo_iter = None
    history = []

    for epoch in range(config.epochs):
        if epoch % max(1, config.pseudo_refresh_epochs) == 0 or pseudo_loader is None:
            keep, pseudo_labels, entropies, clean_probs = compute_gmm_pseudo_labels(
                model,
                target_X,
                config.batch_size,
                config.num_workers,
                device,
                config.gmm_threshold,
            )
            pseudo_loader = make_loader(
                target_X[keep],
                pseudo_labels[keep],
                config.batch_size,
                True,
                config.num_workers,
                True,
            )
            pseudo_iter = iter(pseudo_loader)
            pseudo_summary = {
                "pseudo_count": int(np.sum(keep)),
                "pseudo_ratio": float(np.mean(keep)),
                "target_entropy_mean": float(np.mean(entropies)),
                "clean_prob_mean": float(np.mean(clean_probs)),
            }

        model.train()
        loss_sums = {
            "source_ce": 0.0,
            "pseudo_ce": 0.0,
            "entropy": 0.0,
            "mmd": 0.0,
            "total": 0.0,
        }
        step_count = 0

        for target_inputs, _target_dummy in target_loader:
            try:
                source_inputs, source_labels = next(source_iter)
            except StopIteration:
                source_iter = iter(source_loader)
                source_inputs, source_labels = next(source_iter)
            try:
                pseudo_inputs, pseudo_labels_batch = next(pseudo_iter)
            except StopIteration:
                pseudo_iter = iter(pseudo_loader)
                pseudo_inputs, pseudo_labels_batch = next(pseudo_iter)

            source_inputs = source_inputs.to(device)
            source_labels = source_labels.to(device)
            target_inputs = target_inputs.to(device)
            pseudo_inputs = pseudo_inputs.to(device)
            pseudo_labels_batch = pseudo_labels_batch.to(device)

            optimizer.zero_grad()
            source_logits, source_features = _forward_logits_features(model, source_inputs)
            target_logits, target_features = _forward_logits_features(model, target_inputs)
            pseudo_logits, _pseudo_features = _forward_logits_features(model, pseudo_inputs)

            source_loss = ce_loss(source_logits, source_labels)
            pseudo_loss = ce_loss(pseudo_logits, pseudo_labels_batch)
            target_prob = F.softmax(target_logits, dim=-1)
            mean_prob = target_prob.mean(dim=0)
            entropy_loss = compute_softmax_entropy(target_logits).mean()
            entropy_loss = entropy_loss + torch.sum(mean_prob * torch.log(mean_prob + 1e-5))
            mmd_loss = calculate_mmd_loss(source_features, target_features)

            total_loss = (
                config.source_ce_weight * source_loss
                + config.pseudo_weight * pseudo_loss
                + config.entropy_weight * entropy_loss
                + config.mmd_weight * mmd_loss
            )
            total_loss.backward()
            optimizer.step()

            loss_sums["source_ce"] += float(source_loss.detach().cpu())
            loss_sums["pseudo_ce"] += float(pseudo_loss.detach().cpu())
            loss_sums["entropy"] += float(entropy_loss.detach().cpu())
            loss_sums["mmd"] += float(mmd_loss.detach().cpu())
            loss_sums["total"] += float(total_loss.detach().cpu())
            step_count += 1

        epoch_result = {k: v / max(step_count, 1) for k, v in loss_sums.items()}
        epoch_result["epoch"] = epoch
        epoch_result.update(pseudo_summary)
        if eval_loader is not None:
            epoch_result.update({"target_" + k: v for k, v in evaluate(model, eval_loader, device).items()})
        print(epoch_result)
        history.append(epoch_result)

    return model, history
