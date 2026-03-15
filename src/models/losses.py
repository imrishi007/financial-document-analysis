"""Loss functions for the multimodal fusion model.

Includes ListNet ranking loss as the primary direction loss function,
replacing binary cross-entropy.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def listnet_loss(
    scores: torch.Tensor,
    labels: torch.Tensor,
    dates: torch.Tensor,
    temperature: float = 5.0,
) -> torch.Tensor:
    """ListNet ranking loss.

    For each date, compute KL divergence between predicted probability
    distribution and true return ranking across stocks.

    Parameters
    ----------
    scores : [B, 2] logits from direction head
    labels : [B] binary direction labels (0=DOWN, 1=UP)
    dates : [B] date indices for grouping stocks by trading day
    temperature : Temperature for true label distribution (higher = sharper)

    Returns
    -------
    Scalar ranking loss
    """
    pred_probs = F.softmax(scores, dim=1)[:, 1]  # P(UP) for each sample

    loss = torch.tensor(0.0, device=scores.device, dtype=scores.dtype)
    n_dates = 0

    unique_dates = dates.unique()

    for date in unique_dates:
        mask = dates == date
        n_stocks = mask.sum()

        if n_stocks < 2:  # Need at least 2 stocks to rank
            continue

        pred = pred_probs[mask]
        true = labels[mask].float()

        # Normalize to probability distributions using softmax
        pred_dist = F.softmax(pred, dim=0)
        true_dist = F.softmax(true * temperature, dim=0)

        # Cross-entropy: -sum(true * log(pred))
        # More numerically stable than KL divergence
        ce = -(true_dist * torch.log(pred_dist + 1e-8)).sum()
        loss = loss + ce
        n_dates += 1

    return loss / max(n_dates, 1)


class ListNetRankingLoss(nn.Module):
    """ListNet ranking loss as a PyTorch module."""

    def __init__(self, temperature: float = 5.0):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        scores: torch.Tensor,
        labels: torch.Tensor,
        dates: torch.Tensor,
    ) -> torch.Tensor:
        return listnet_loss(scores, labels, dates, self.temperature)


class CombinedLoss(nn.Module):
    """Combined loss: ListNet ranking + volatility MSE.

    L = lambda_dir * L_ListNet + lambda_vol * L_vol_MSE

    Parameters
    ----------
    lambda_dir : Weight for direction (ListNet) loss. Default 0.7.
    lambda_vol : Weight for volatility (MSE) loss. Default 0.3.
    temperature : ListNet temperature. Default 5.0.
    """

    def __init__(
        self,
        lambda_dir: float = 0.7,
        lambda_vol: float = 0.3,
        temperature: float = 5.0,
    ):
        super().__init__()
        self.lambda_dir = lambda_dir
        self.lambda_vol = lambda_vol
        self.listnet = ListNetRankingLoss(temperature)
        self.mse = nn.MSELoss(reduction="none")

    def forward(
        self,
        direction_logits: torch.Tensor,
        direction_labels: torch.Tensor,
        volatility_pred: torch.Tensor,
        volatility_target: torch.Tensor,
        dates: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Compute combined loss.

        Parameters
        ----------
        direction_logits : [B, 2]
        direction_labels : [B]
        volatility_pred : [B]
        volatility_target : [B]
        dates : [B] date indices for ListNet grouping

        Returns
        -------
        dict with 'total', 'direction', 'volatility' loss tensors
        """
        # Direction loss (ListNet)
        dir_valid = direction_labels >= 0
        if dir_valid.any():
            dir_loss = self.listnet(
                direction_logits[dir_valid],
                direction_labels[dir_valid],
                dates[dir_valid],
            )
        else:
            dir_loss = torch.tensor(0.0, device=direction_logits.device)

        # Volatility loss (MSE)
        vol_valid = ~torch.isnan(volatility_target)
        if vol_valid.any():
            vol_loss = self.mse(
                volatility_pred[vol_valid],
                volatility_target[vol_valid],
            ).mean()
        else:
            vol_loss = torch.tensor(0.0, device=direction_logits.device)

        total = self.lambda_dir * dir_loss + self.lambda_vol * vol_loss

        return {
            "total": total,
            "direction": dir_loss,
            "volatility": vol_loss,
        }


# ────────────────────────────────────────────────────────────────────
# Phase 12: QLIKE loss and volatility-primary combined loss
# ────────────────────────────────────────────────────────────────────


class QLIKELoss(nn.Module):
    """QLIKE (Quasi-Likelihood) loss for volatility forecasting.

    L = mean( sigma_true^2 / sigma_pred^2 + log(sigma_pred^2) )

    This is the standard loss for evaluating volatility forecasts in
    financial econometrics (Patton, 2011). It is scale-sensitive and
    penalizes under-prediction more than over-prediction.

    Inputs are *annualized volatility* (not variance). Internally
    squared to get variance.
    """

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        pred : [B] predicted volatility (must be > 0, e.g. from Softplus)
        target : [B] realized volatility
        """
        valid = (target > self.eps) & (pred > self.eps) & ~torch.isnan(target)
        if not valid.any():
            return torch.tensor(0.0, device=pred.device, requires_grad=True)

        pred_var = pred[valid].pow(2).clamp(min=self.eps)
        true_var = target[valid].pow(2).clamp(min=self.eps)

        # QLIKE: true_var / pred_var + log(pred_var)
        qlike = true_var / pred_var + torch.log(pred_var)
        return qlike.mean()


class CombinedVolatilityLoss(nn.Module):
    """Phase 12 combined loss: QLIKE-primary + ListNet-auxiliary.

    L = lambda_vol * L_QLIKE + lambda_dir * L_ListNet

    Default: 0.85 × QLIKE + 0.15 × ListNet (volatility-primary design)
    """

    def __init__(
        self,
        lambda_vol: float = 0.85,
        lambda_dir: float = 0.15,
        temperature: float = 5.0,
        qlike_eps: float = 1e-6,
    ):
        super().__init__()
        self.lambda_vol = lambda_vol
        self.lambda_dir = lambda_dir
        self.qlike = QLIKELoss(eps=qlike_eps)
        self.listnet = ListNetRankingLoss(temperature)

    def forward(
        self,
        direction_logits: torch.Tensor,
        direction_labels: torch.Tensor,
        volatility_pred: torch.Tensor,
        volatility_target: torch.Tensor,
        dates: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Returns dict with 'total', 'volatility', 'direction' loss tensors.
        """
        # Primary: QLIKE volatility loss
        vol_valid = ~torch.isnan(volatility_target) & (volatility_target > 1e-6)
        if vol_valid.any():
            vol_loss = self.qlike(
                volatility_pred[vol_valid], volatility_target[vol_valid]
            )
        else:
            vol_loss = torch.tensor(0.0, device=volatility_pred.device)

        # Auxiliary: ListNet direction loss
        dir_valid = direction_labels >= 0
        if dir_valid.any():
            dir_loss = self.listnet(
                direction_logits[dir_valid],
                direction_labels[dir_valid],
                dates[dir_valid],
            )
        else:
            dir_loss = torch.tensor(0.0, device=direction_logits.device)

        total = self.lambda_vol * vol_loss + self.lambda_dir * dir_loss

        return {
            "total": total,
            "volatility": vol_loss,
            "direction": dir_loss,
        }
