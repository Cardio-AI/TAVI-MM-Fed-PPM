import os
from pathlib import Path
import logging
from typing import List, Dict, Tuple, Optional

from omegaconf import OmegaConf
import pandas as pd
import numpy as np
import SimpleITK as sitk
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, WeightedRandomSampler
from transformers import DistilBertTokenizer, DistilBertModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, balanced_accuracy_score, roc_curve
from torchmetrics.functional import accuracy
from monai.networks.nets.swin_unetr import SwinTransformer

from fedbiomed.common.training_plans import TorchTrainingPlan
from fedbiomed.common.datamanager import DataManager
from fedbiomed.common.dataloader import PytorchDataLoader
from ecg_transform.inp import ECGInput, ECGInputSchema
from ecg_transform.sample import ECGMetadata, ECGSample
from ecg_transform.t.common import (
	HandleConstantLeads,
	LinearResample,
	ReorderLeads,
)
from ecg_transform.t.scale import Standardize
from ecg_transform.t.cut import SegmentNonoverlapping

from fairseq_signals.models.wav2vec2.wav2vec2_cmsc import Wav2Vec2CMSCModel
from fairseq_signals.models.classification.ecg_transformer_classifier import ECGTransformerClassificationModel


##############################
# Training plan definition
##############################


class TP(TorchTrainingPlan):

	""" Custom Fed-BioMed Training Plan."""

	logger = logging.getLogger(__name__)
	logger.setLevel(logging.ERROR)

	class TrainingMode:
		RECONSTRUCTION = "reconstruction"
		SIMPLE_MISSING_TOKEN = "simple"
		CROSS_ATTENTION = "cross_attention"

	def __init__(self, *args, **kwargs):

		super().__init__(*args, **kwargs)

		self.trainset = None
		self.custom_testset = None
		self.bce_pos_weight = None

		self.train_results = []
		self.test_results = []

		# Per-sample loss tracking for training and testing
		self.train_losses = {
			'classification': [],
			'reconstruction_total': [],
			'ct_reconstruction': [],
			'ecg_reconstruction': [],
			'total': []
		}
		self.test_losses = {
			'classification': [],
			'reconstruction_total': [],
			'ct_reconstruction': [],
			'ecg_reconstruction': [],
			'total': []
		}

	def init_dependencies(self):
		deps = [
			"import os",
			"from pathlib import Path",
			"import logging",
			"from typing import List, Dict, Tuple, Optional",

			"from omegaconf import OmegaConf",
			"import pandas as pd",
			"import numpy as np",
			"import SimpleITK as sitk",
			"import torch",
			"import torch.nn as nn",
			"import torch.nn.functional as F",
			"from torch.utils.data import Dataset, WeightedRandomSampler",
			"from transformers import DistilBertTokenizer, DistilBertModel",
			"from sklearn.model_selection import train_test_split",
			"from sklearn.metrics import roc_auc_score, average_precision_score, balanced_accuracy_score, roc_curve",
			"from torchmetrics.functional import accuracy",
			"from monai.networks.nets.swin_unetr import SwinTransformer",

			"from fedbiomed.common.datamanager import DataManager",
			"from fedbiomed.common.dataloader import PytorchDataLoader",
			"from ecg_transform.inp import ECGInput, ECGInputSchema",
			"from ecg_transform.sample import ECGMetadata, ECGSample",
			"from ecg_transform.t.common import HandleConstantLeads, LinearResample, ReorderLeads",
			"from ecg_transform.t.scale import Standardize",
			"from ecg_transform.t.cut import SegmentNonoverlapping",
			"from fairseq_signals.models.wav2vec2.wav2vec2_cmsc import Wav2Vec2CMSCModel",
			"from fairseq_signals.models.classification.ecg_transformer_classifier import ECGTransformerClassificationModel"
		]
		return deps

	##############################
	# Model definition - Feature Extractors
	##############################

	@staticmethod
	def standardize(x: torch.Tensor) -> torch.Tensor:
		valid_mask = ~torch.isnan(x).all(dim=tuple(range(1, len(x.shape))))
		if valid_mask.any():
			valid_values = x[valid_mask]
			mean_val = valid_values.mean()
			std_val = valid_values.std()
			x[valid_mask] = (valid_values - mean_val) / (std_val + 1e-8)
		return x

	class ECGFeatureExtractor(nn.Module):
		def __init__(self,
					 ecg_fm_cfg: str,
					 wav2vec2_cfg: str,
					 ckpt_path: Optional[str] = None,
					 strip_output_layers: bool = True,
					 ):
			super().__init__()
			cfg = OmegaConf.create(ecg_fm_cfg)
			wav2vec2_cfg = OmegaConf.create(wav2vec2_cfg)
			encoder = Wav2Vec2CMSCModel(wav2vec2_cfg)
			self.ecg_fm = ECGTransformerClassificationModel(cfg, encoder)
			if strip_output_layers:
				self.ecg_fm.encoder.quantizer = None
				self.ecg_fm.encoder.project_q = None
				self.ecg_fm.encoder.final_proj = None

			if ckpt_path is not None:
				if not Path(ckpt_path).is_file():
					raise FileNotFoundError("configured ECG checkpoint was not found")
				sd_ecg_transformer = torch.load(
					ckpt_path, map_location='cpu', weights_only=True
				)
				self.ecg_fm.load_state_dict(sd_ecg_transformer)
				print('>>> LOADING ECG CKPT SUCCESSFUL <<<')

		def forward(self, ecg: torch.Tensor) -> torch.Tensor:
			res = self.ecg_fm.encoder.extract_features(source=ecg, padding_mask=None)
			ecg_feats = res["x"]
			ecg_feats = self.ecg_fm.final_dropout(ecg_feats)
			missing_ecg_mask = torch.isnan(ecg).all(dim=(1, 2))
			ecg_feats[missing_ecg_mask] = float('nan')
			return ecg_feats

	class CTFeatureExtractor(nn.Module):
		def __init__(self, ckpt_path: Optional[str] = None):
			super().__init__()
			self.swin = SwinTransformer(
				in_chans=1,
				embed_dim=48,
				window_size=(7, 7, 7),
				patch_size=(2, 2, 2),
				depths=(2, 2, 2, 2),
				num_heads=(3, 6, 12, 24),
				mlp_ratio=4.0,
				qkv_bias=True,
				drop_rate=0.0,
				attn_drop_rate=0.0,
				drop_path_rate=0.0,
				norm_layer=nn.LayerNorm,
				spatial_dims=3,
				use_checkpoint=False,
				downsample='merging',
			)
			if ckpt_path is not None:
				if not Path(ckpt_path).is_file():
					raise FileNotFoundError("configured CT checkpoint was not found")
				sd_swin = torch.load(
					ckpt_path, map_location='cpu', weights_only=True
				)
				self.swin.load_state_dict(sd_swin)
				print('>>> LOADING CT CKPT SUCCESSFUL <<<')

		def forward(self, ct: torch.Tensor) -> torch.Tensor:
			ct_feats = self.swin(ct)[-1]
			B, C, D, H, W = ct_feats.shape
			ct_feats = ct_feats.view(B, C, -1)  # [B, C, D*H*W]
			ct_feats = ct_feats.permute(0, 2, 1)  # [B, D*H*W, C]

			missing_ct_mask = torch.isnan(ct).all(dim=(1, 2, 3, 4))
			ct_feats[missing_ct_mask] = float('nan')

			return ct_feats

	##############################
	# Model definition - Multimodal Fusion
	##############################

	class SimpleMissingTokenFusion(nn.Module):
		"""Simple fusion with learnable missing tokens."""

		def __init__(
				self,
				ct_dim: int = 768,
				ecg_dim: int = 768,
				demo_dim: int = 64,
				prosthesis_dim: int = 768,
				hidden_dim: int = 256,
				dropout: float = 0.1
		):
			super().__init__()

			self.hidden_dim = hidden_dim

			# Project all modalities to same dimension
			self.ct_proj = nn.Linear(ct_dim, hidden_dim)
			self.ecg_proj = nn.Linear(ecg_dim, hidden_dim)
			self.demo_proj = nn.Linear(demo_dim, hidden_dim)
			self.prosthesis_proj = nn.Linear(prosthesis_dim, hidden_dim)

			# Learnable missing tokens
			self.ct_missing_token = nn.Parameter(torch.randn(1, hidden_dim))
			self.ecg_missing_token = nn.Parameter(torch.randn(1, hidden_dim))
			self.demo_missing_token = nn.Parameter(torch.randn(1, hidden_dim))
			self.prosthesis_missing_token = nn.Parameter(torch.randn(1, hidden_dim))

			# Final fusion
			self.fusion = nn.Sequential(
				nn.Linear(hidden_dim * 4, hidden_dim * 2),
				nn.ReLU(),
				nn.Dropout(dropout),
				nn.Linear(hidden_dim * 2, hidden_dim)
			)

		def forward(
				self,
				ct_features: Optional[torch.Tensor],
				ecg_features: Optional[torch.Tensor],
				demographics: Optional[torch.Tensor],
				prosthesis_and_location: Optional[torch.Tensor] = None
		) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:

			batch_size = ct_features.size(0)

			# Standardize and detect missing
			ct_features = TP.standardize(ct_features)
			ecg_features = TP.standardize(ecg_features)

			ct_missing = torch.isnan(ct_features).all(dim=(1, 2))
			ecg_missing = torch.isnan(ecg_features).all(dim=(1, 2))
			demo_missing = torch.isnan(demographics).all(dim=1)
			prosthesis_missing = prosthesis_and_location is None or torch.isnan(prosthesis_and_location).all(dim=1)

			# Clean and project
			ct_clean = torch.nan_to_num(ct_features, nan=0.0)
			ecg_clean = torch.nan_to_num(ecg_features, nan=0.0)
			demo_clean = torch.nan_to_num(demographics, nan=0.0)

			if prosthesis_and_location is not None:
				prosthesis_clean = torch.nan_to_num(prosthesis_and_location, nan=0.0)
			else:
				prosthesis_clean = torch.zeros(batch_size, 768, device=ct_features.device)
				prosthesis_missing = torch.ones(batch_size, dtype=torch.bool, device=ct_features.device)

			# Project and pool
			ct_pooled = self.ct_proj(ct_clean).mean(dim=1)  # [B, hidden_dim]
			ecg_pooled = self.ecg_proj(ecg_clean).mean(dim=1)  # [B, hidden_dim]
			demo_projected = self.demo_proj(demo_clean)  # [B, hidden_dim]
			prosthesis_projected = self.prosthesis_proj(prosthesis_clean)  # [B, hidden_dim]

			# Replace missing with learnable tokens
			ct_pooled[ct_missing] = self.ct_missing_token.expand(ct_missing.sum(), -1)
			ecg_pooled[ecg_missing] = self.ecg_missing_token.expand(ecg_missing.sum(), -1)
			demo_projected[demo_missing] = self.demo_missing_token.expand(demo_missing.sum(), -1)
			prosthesis_projected[prosthesis_missing] = self.prosthesis_missing_token.expand(prosthesis_missing.sum(),
																							-1)

			# Concatenate and fuse
			concatenated = torch.cat([ct_pooled, ecg_pooled, demo_projected, prosthesis_projected], dim=-1)
			fused_features = self.fusion(concatenated)

			return fused_features, {'dummy': torch.tensor(0.0).to(concatenated.device)}

	class MultiModalCrossAttention(nn.Module):
		"""Cross-attention based multimodal fusion."""

		def __init__(
				self,
				ct_dim: int = 768,
				ecg_dim: int = 768,
				demo_dim: int = 64,
				prosthesis_dim: int = 768,
				hidden_dim: int = 512,
				num_heads: int = 8,
				dropout: float = 0.1,
				use_learnable_tokens: bool = True,
				balance_method: Optional[str] = None
		):
			super().__init__()

			self.hidden_dim = hidden_dim
			self.num_heads = num_heads
			self.use_learnable_tokens = use_learnable_tokens

			# Project all modalities to same dimension
			self.ct_proj = nn.Linear(ct_dim, hidden_dim)
			self.ecg_proj = nn.Linear(ecg_dim, hidden_dim)
			self.demo_proj = nn.Linear(demo_dim, hidden_dim)
			self.prosthesis_proj = nn.Linear(prosthesis_dim, hidden_dim)

			# Learnable tokens for missing modalities
			if use_learnable_tokens:
				self.ct_missing_token = nn.Parameter(torch.randn(1, 1, hidden_dim))
				self.ecg_missing_token = nn.Parameter(torch.randn(1, 1, hidden_dim))
				self.demo_missing_token = nn.Parameter(torch.randn(1, demo_dim))
				self.prosthesis_missing_token = nn.Parameter(torch.randn(1, prosthesis_dim))

			# Modality embeddings
			self.ct_embed = nn.Parameter(torch.randn(1, 1, hidden_dim))
			self.ecg_embed = nn.Parameter(torch.randn(1, 1, hidden_dim))
			self.demo_embed = nn.Parameter(torch.randn(1, 1, hidden_dim))
			self.prosthesis_embed = nn.Parameter(torch.randn(1, 1, hidden_dim))

			self.target_seq_len = 1
			self.ct_pool = nn.AdaptiveAvgPool1d(self.target_seq_len)
			self.ecg_pool = nn.AdaptiveAvgPool1d(self.target_seq_len)
			self.demo_expand = nn.Linear(hidden_dim, hidden_dim * self.target_seq_len)

			# Cross-attention layers
			self.ct_to_others = nn.MultiheadAttention(
				hidden_dim, num_heads, dropout=dropout, batch_first=True
			)
			self.ecg_to_others = nn.MultiheadAttention(
				hidden_dim, num_heads, dropout=dropout, batch_first=True
			)
			self.demo_to_others = nn.MultiheadAttention(
				hidden_dim, num_heads, dropout=dropout, batch_first=True
			)
			self.prosthesis_to_others = nn.MultiheadAttention(
				hidden_dim, num_heads, dropout=dropout, batch_first=True
			)

			# Feed-forward networks
			self.ct_ffn = self._make_ffn(hidden_dim, dropout)
			self.ecg_ffn = self._make_ffn(hidden_dim, dropout)
			self.demo_ffn = self._make_ffn(hidden_dim, dropout)
			self.prosthesis_ffn = self._make_ffn(hidden_dim, dropout)

			# Layer normalization
			self.ct_norm1 = nn.LayerNorm(hidden_dim)
			self.ct_norm2 = nn.LayerNorm(hidden_dim)
			self.ecg_norm1 = nn.LayerNorm(hidden_dim)
			self.ecg_norm2 = nn.LayerNorm(hidden_dim)
			self.demo_norm1 = nn.LayerNorm(hidden_dim)
			self.demo_norm2 = nn.LayerNorm(hidden_dim)
			self.prosthesis_norm1 = nn.LayerNorm(hidden_dim)
			self.prosthesis_norm2 = nn.LayerNorm(hidden_dim)

			self.balance_method = balance_method

			# Final fusion layer
			self.fusion = nn.Sequential(
				nn.Linear(hidden_dim * 4, hidden_dim * 2),
				nn.ReLU(),
				nn.Dropout(dropout),
				nn.Linear(hidden_dim * 2, hidden_dim)
			)

		@staticmethod
		def _make_ffn(hidden_dim: int, dropout: float) -> nn.Module:
			return nn.Sequential(
				nn.Linear(hidden_dim, hidden_dim * 4),
				nn.ReLU(),
				nn.Dropout(dropout),
				nn.Linear(hidden_dim * 4, hidden_dim)
			)

		def _handle_missing_modality(
				self,
				features: torch.Tensor,
				proj_fn: nn.Module,
				missing_token: nn.Parameter,
				modality_embed: nn.Parameter
		) -> torch.Tensor:
			"""Handle missing modalities with learnable tokens."""
			missing = torch.isnan(features).all(dim=(1, 2))

			if self.use_learnable_tokens:
				features_clean = torch.nan_to_num(features, nan=0.0)
				proj_features = proj_fn(features_clean)
				enhanced_features = proj_features + modality_embed

				if missing.any():
					batch_size, seq_len, hidden_dim = proj_features.shape
					missing_replacement = (missing_token + modality_embed).expand(
						missing.sum(), seq_len, hidden_dim
					)
					enhanced_features[missing] = missing_replacement
			else:
				features_clean = torch.nan_to_num(features, nan=0.0)
				proj_features = proj_fn(features_clean)
				enhanced_features = proj_features + modality_embed

				if missing.any():
					enhanced_features[missing] = 0.0

			return enhanced_features

		def _handle_demographics_missing(self, demographics: torch.Tensor) -> torch.Tensor:
			"""Handle missing demographics."""
			missing_mask = torch.isnan(demographics)
			demo_filled = demographics.clone()

			if self.use_learnable_tokens:
				demo_filled[missing_mask] = self.demo_missing_token.expand_as(demographics)[missing_mask]
			else:
				demo_filled = torch.nan_to_num(demographics, nan=0.0)

			projected = self.demo_proj(demo_filled)
			return projected

		def _handle_prosthesis_missing(self, prosthesis_and_location: torch.Tensor) -> torch.Tensor:
			"""Handle missing prosthesis and location."""
			if prosthesis_and_location is None:
				# Create dummy data
				batch_size = 1  # This will be corrected in forward
				prosthesis_filled = torch.zeros(batch_size, 768, device='cpu')
				if self.use_learnable_tokens:
					prosthesis_filled = self.prosthesis_missing_token.expand_as(prosthesis_filled)
			else:
				missing_mask = torch.isnan(prosthesis_and_location)
				prosthesis_filled = prosthesis_and_location.clone()

				if self.use_learnable_tokens:
					prosthesis_filled[missing_mask] = self.prosthesis_missing_token.expand_as(prosthesis_and_location)[
						missing_mask]
				else:
					prosthesis_filled = torch.nan_to_num(prosthesis_and_location, nan=0.0)

			projected = self.prosthesis_proj(prosthesis_filled)
			return projected

		def forward(
				self,
				ct_features: Optional[torch.Tensor],
				ecg_features: Optional[torch.Tensor],
				demographics: Optional[torch.Tensor],
				prosthesis_and_location: Optional[torch.Tensor] = None
		) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
			"""Forward pass with cross-attention."""
			demographics = demographics[:, None]

			ct_features = self.ct_pool(ct_features.transpose(1, 2)).transpose(1, 2)
			ecg_features = self.ecg_pool(ecg_features.transpose(1, 2)).transpose(1, 2)

			ct_features = TP.standardize(ct_features)
			ecg_features = TP.standardize(ecg_features)

			# Handle missing modalities
			ct_tokens = self._handle_missing_modality(
				ct_features, self.ct_proj, self.ct_missing_token, self.ct_embed
			)
			ecg_tokens = self._handle_missing_modality(
				ecg_features, self.ecg_proj, self.ecg_missing_token, self.ecg_embed
			)
			demo_tokens = self._handle_demographics_missing(demographics)
			demo_tokens = self.demo_expand(demo_tokens.squeeze(1))
			demo_tokens = demo_tokens.view(demo_tokens.size(0), self.target_seq_len, -1)

			# Handle prosthesis and location
			if prosthesis_and_location is not None:
				prosthesis_tokens = self._handle_prosthesis_missing(prosthesis_and_location)
				prosthesis_tokens = prosthesis_tokens.view(prosthesis_tokens.size(0), self.target_seq_len, -1)
			else:
				# Create dummy prosthesis tokens
				batch_size = ct_features.size(0)
				device = ct_features.device
				prosthesis_tokens = torch.zeros(batch_size, self.target_seq_len, self.hidden_dim, device=device)
				if self.use_learnable_tokens:
					prosthesis_tokens = self.prosthesis_missing_token.expand(batch_size, -1).view(batch_size,
																								  self.target_seq_len,
																								  -1)

			# Cross-attention computations
			attention_weights = {}

			# CT attends to ECG + Demographics + Prosthesis
			kv_for_ct = torch.cat([ecg_tokens, demo_tokens, prosthesis_tokens], dim=1)
			ct_attended, ct_attn_weights = self.ct_to_others(
				ct_tokens, kv_for_ct, kv_for_ct
			)
			ct_enhanced = self.ct_norm1(ct_tokens + ct_attended)
			ct_output = self.ct_norm2(ct_enhanced + self.ct_ffn(ct_enhanced))
			attention_weights['ct_to_others'] = ct_attn_weights

			# ECG attends to CT + Demographics + Prosthesis
			kv_for_ecg = torch.cat([ct_tokens, demo_tokens, prosthesis_tokens], dim=1)
			ecg_attended, ecg_attn_weights = self.ecg_to_others(
				ecg_tokens, kv_for_ecg, kv_for_ecg
			)
			ecg_enhanced = self.ecg_norm1(ecg_tokens + ecg_attended)
			ecg_output = self.ecg_norm2(ecg_enhanced + self.ecg_ffn(ecg_enhanced))
			attention_weights['ecg_to_others'] = ecg_attn_weights

			# Demographics attends to CT + ECG + Prosthesis
			kv_for_demo = torch.cat([ct_tokens, ecg_tokens, prosthesis_tokens], dim=1)
			demo_attended, demo_attn_weights = self.demo_to_others(
				demo_tokens, kv_for_demo, kv_for_demo
			)
			demo_enhanced = self.demo_norm1(demo_tokens + demo_attended)
			demo_output = self.demo_norm2(demo_enhanced + self.demo_ffn(demo_enhanced))
			attention_weights['demo_to_others'] = demo_attn_weights

			# Prosthesis attends to CT + ECG + Demographics
			kv_for_prosthesis = torch.cat([ct_tokens, ecg_tokens, demo_tokens], dim=1)
			prosthesis_attended, prosthesis_attn_weights = self.prosthesis_to_others(
				prosthesis_tokens, kv_for_prosthesis, kv_for_prosthesis
			)
			prosthesis_enhanced = self.prosthesis_norm1(prosthesis_tokens + prosthesis_attended)
			prosthesis_output = self.prosthesis_norm2(prosthesis_enhanced + self.prosthesis_ffn(prosthesis_enhanced))
			attention_weights['prosthesis_to_others'] = prosthesis_attn_weights

			# Pool each modality to single vector
			ct_pooled = ct_output.mean(dim=1)
			ecg_pooled = ecg_output.mean(dim=1)
			demo_pooled = demo_output.mean(dim=1)
			prosthesis_pooled = prosthesis_output.mean(dim=1)

			# Simple concatenation (other balance methods can be added later)
			concatenated = torch.cat([ct_pooled, ecg_pooled, demo_pooled, prosthesis_pooled], dim=-1)
			fused_features = self.fusion(concatenated)

			return fused_features, attention_weights

	class ModalityImputer(nn.Module):
		def __init__(self, hidden_dim: int = 256):
			super().__init__()
			# Much simpler - just MLPs
			self.ct_imputer = nn.Sequential(
				nn.Linear(hidden_dim * 2, hidden_dim),  # ECG_pooled + demo
				nn.ReLU(),
				nn.Dropout(0.3),  # Higher dropout
				nn.Linear(hidden_dim, hidden_dim)
			)

			self.ecg_imputer = nn.Sequential(
				nn.Linear(hidden_dim * 2, hidden_dim),  # CT_pooled + demo
				nn.ReLU(),
				nn.Dropout(0.3),
				nn.Linear(hidden_dim, hidden_dim)
			)

		def forward(self,
					ct_projected: torch.Tensor,
					ecg_projected: torch.Tensor,
					demo_features: torch.Tensor
					) -> Dict[str, torch.Tensor]:
			# Just use pooled features - much simpler
			ct_pooled = ct_projected.mean(dim=1)
			ecg_pooled = ecg_projected.mean(dim=1)

			# Simple concatenation and imputation
			ct_imputed = self.ct_imputer(torch.cat([ecg_pooled, demo_features], dim=-1))
			ecg_imputed = self.ecg_imputer(torch.cat([ct_pooled, demo_features], dim=-1))

			return {
				'ct_original_pooled': ct_pooled,
				'ecg_original_pooled': ecg_pooled,
				'ct_imputed_pooled': ct_imputed,
				'ecg_imputed_pooled': ecg_imputed
			}

	class ReconstructionLoss(nn.Module):
		"""Compute reconstruction losses for modality imputation."""

		def __init__(self,
					 loss_type: str = 'cosine',
					 weight: float = 1.0
					 ):
			super().__init__()
			self.loss_type = loss_type
			self.weight = weight

		def forward(self,
					imputation_results: Dict,
					ct_missing: torch.Tensor,
					ecg_missing: torch.Tensor
					) -> Dict[str, torch.Tensor]:
			"""
			Compute reconstruction losses for available modalities.

			Args:
				imputation_results: dict from ModalityImputer.forward()
				ct_missing: [B] boolean mask for missing CT
				ecg_missing: [B] boolean mask for missing ECG

			Returns:
				dict with individual and total reconstruction losses
			"""
			losses = {}

			# Only compute reconstruction loss for samples where both modalities are available
			both_available = ~ct_missing & ~ecg_missing

			if both_available.any():
				if self.loss_type == 'cosine':
					# Cosine similarity loss (1 - cosine_similarity) on pooled representations
					ct_loss = 1 - F.cosine_similarity(
						imputation_results['ct_original_pooled'][both_available],  # .clone().detach(),
						imputation_results['ct_imputed_pooled'][both_available],
						dim=-1
					).mean()

					ecg_loss = 1 - F.cosine_similarity(
						imputation_results['ecg_original_pooled'][both_available],  # .clone().detach(),
						imputation_results['ecg_imputed_pooled'][both_available],
						dim=-1
					).mean()

				elif self.loss_type == 'mse':
					# Mean squared error on pooled representations
					ct_loss = F.mse_loss(
						imputation_results['ct_imputed_pooled'][both_available],
						imputation_results['ct_original_pooled'][both_available]
					)

					ecg_loss = F.mse_loss(
						imputation_results['ecg_imputed_pooled'][both_available],
						imputation_results['ecg_original_pooled'][both_available]
					)

				elif self.loss_type == 'l1':
					# L1 loss on pooled representations
					ct_loss = F.l1_loss(
						imputation_results['ct_imputed_pooled'][both_available],
						imputation_results['ct_original_pooled'][both_available]
					)

					ecg_loss = F.l1_loss(
						imputation_results['ecg_imputed_pooled'][both_available],
						imputation_results['ecg_original_pooled'][both_available]
					)

				else:
					raise ValueError(f"Unknown loss type: {self.loss_type}")

				losses['ct_reconstruction'] = ct_loss
				losses['ecg_reconstruction'] = ecg_loss
				losses['total_reconstruction'] = (ct_loss + ecg_loss) * self.weight
			else:
				# No samples with both modalities available
				device = imputation_results['ct_original_pooled'].device
				losses['ct_reconstruction'] = torch.tensor(0.0, device=device)
				losses['ecg_reconstruction'] = torch.tensor(0.0, device=device)
				losses['total_reconstruction'] = torch.tensor(0.0, device=device)

			return losses

	class MultiModalFusion(nn.Module):
		"""Simple fusion module using imputation for missing modalities."""

		def __init__(
				self,
				ct_dim: int = 768,
				ecg_dim: int = 768,
				demo_dim: int = 64,
				prosthesis_dim: int = 768,
				hidden_dim: int = 512,
				dropout: float = 0.1,
				reconstruction_loss_weight: float = 0.1,
				reconstruction_loss_type: str = 'cosine'
		):
			super().__init__()

			self.hidden_dim = hidden_dim

			# Project all modalities to same dimension BEFORE pooling
			self.ct_proj = nn.Linear(ct_dim, hidden_dim)
			self.ecg_proj = nn.Linear(ecg_dim, hidden_dim)
			self.demo_proj = nn.Linear(demo_dim, hidden_dim)
			self.prosthesis_proj = nn.Linear(prosthesis_dim, hidden_dim)

			# Modality embeddings (optional, can help distinguish modalities)
			self.ct_embed = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
			self.ecg_embed = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
			self.demo_embed = nn.Parameter(torch.randn(1, hidden_dim) * 0.02)
			self.prosthesis_embed = nn.Parameter(torch.randn(1, hidden_dim) * 0.02)

			# Pooling layers - applied AFTER projection
			self.ct_pool = nn.AdaptiveAvgPool1d(1)  # Pool from ~8 tokens to 1
			self.ecg_pool = nn.AdaptiveAvgPool1d(1)  # Pool from ~156 tokens to 1

			# Imputation and reconstruction loss
			self.modality_imputer = TP.ModalityImputer(hidden_dim)
			self.reconstruction_loss = TP.ReconstructionLoss(
				loss_type=reconstruction_loss_type,
				weight=reconstruction_loss_weight
			)

			# Final fusion layer
			self.fusion = nn.Sequential(
				nn.Linear(hidden_dim * 4, hidden_dim * 2),
				nn.ReLU(),
				nn.Dropout(dropout),
				nn.Linear(hidden_dim * 2, hidden_dim)
			)

		def _handle_demographics_missing(self, demographics: torch.Tensor) -> torch.Tensor:
			"""Handle missing demographics by filling with zeros."""
			# Replace NaNs with zeros
			demo_filled = torch.nan_to_num(demographics, nan=0.0)
			projected = self.demo_proj(demo_filled) + self.demo_embed
			return projected

		def forward(
				self,
				ct_features: Optional[torch.Tensor],  # [B, ~8, 768]
				ecg_features: Optional[torch.Tensor],  # [B, ~156, 768]
				demographics: Optional[torch.Tensor],  # [B, demo_dim]
				prosthesis_and_location: Optional[torch.Tensor] = None,  # [B, 1, 768]
		) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
			"""
			Forward pass with imputation-based missing modality handling.
			"""
			# Standardize features first
			ct_features = TP.standardize(ct_features)
			ecg_features = TP.standardize(ecg_features)

			# Detect missing modalities
			ct_missing = torch.isnan(ct_features).all(dim=(1, 2))
			ecg_missing = torch.isnan(ecg_features).all(dim=(1, 2))

			# Clean NaNs and project to hidden space BEFORE pooling
			ct_clean = torch.nan_to_num(ct_features, nan=0.0)
			ecg_clean = torch.nan_to_num(ecg_features, nan=0.0)

			# Project each token to hidden dimension
			ct_projected = self.ct_proj(ct_clean) + self.ct_embed  # [B, ~8, hidden_dim]
			ecg_projected = self.ecg_proj(ecg_clean) + self.ecg_embed  # [B, ~156, hidden_dim]

			# Handle demographics
			demo_projected = self._handle_demographics_missing(demographics)  # [B, hidden_dim]

			# Handle prosthesis and location
			if prosthesis_and_location is not None:
				prosthesis_projected = self.prosthesis_proj(prosthesis_and_location) + self.prosthesis_embed
			else:
				# Create dummy prosthesis features
				batch_size = ct_features.size(0)
				device = ct_features.device
				prosthesis_projected = torch.zeros(batch_size, self.hidden_dim, device=device) + self.prosthesis_embed

			# Perform imputation for all samples at token level (for reconstruction loss)
			imputation_results = self.modality_imputer(
				ct_projected, ecg_projected, demo_projected
			)

			# Compute reconstruction losses
			reconstruction_losses = self.reconstruction_loss(
				imputation_results, ct_missing, ecg_missing
			)

			ct_pooled = imputation_results['ct_original_pooled']
			ct_pooled[ct_missing] = imputation_results['ct_imputed_pooled'][ct_missing]
			ecg_pooled = imputation_results['ecg_original_pooled']
			ecg_pooled[ecg_missing] = imputation_results['ecg_imputed_pooled'][ecg_missing]

			# Concatenate for final fusion
			concatenated = torch.cat([ct_pooled, ecg_pooled, demo_projected, prosthesis_projected], dim=-1)
			fused_features = self.fusion(concatenated)

			# Return both fused features and reconstruction losses
			return fused_features, {
				'reconstruction_losses': reconstruction_losses,
				'imputation_results': imputation_results
			}

	##############################
	# Model definition - Classifier heads
	##############################

	class MultiModalClassifier(nn.Module):
		"""Classifier using imputation-based fusion."""

		def __init__(
				self,
				num_classes: int,
				demo_dim: int,
				ecg_fm_cfg: Dict,
				wav2vec2_cfg: Dict,
				dropout: float = 0.1,
				train_img_encoder: bool = False,
				train_ecg_encoder: bool = False,
				modalities: List[str] = ['ct', 'ecg', 'demographics'],
				reconstruction_loss_weight: float = 0.1,
				reconstruction_loss_type: str = 'cosine',
				ct_feature_extractor_ckpt: Optional[str] = None,
				ecg_feature_extractor_ckpt: Optional[str] = None
		):
			super().__init__()

			self.ct_feature_extractor = TP.CTFeatureExtractor(ckpt_path=ct_feature_extractor_ckpt)
			self.ecg_feature_extractor = TP.ECGFeatureExtractor(ecg_fm_cfg, wav2vec2_cfg,
															 ckpt_path=ecg_feature_extractor_ckpt)

			for p in self.ct_feature_extractor.parameters():
				p.requires_grad = train_img_encoder
			for p in self.ecg_feature_extractor.parameters():
				p.requires_grad = train_ecg_encoder

			self.fusion = TP.MultiModalFusion(
				ct_dim=768,
				ecg_dim=768,
				demo_dim=demo_dim,
				prosthesis_dim=768,
				hidden_dim=256,
				reconstruction_loss_weight=reconstruction_loss_weight,
				reconstruction_loss_type=reconstruction_loss_type
			)

			self.classifier = nn.Sequential(
				nn.Dropout(dropout),
				nn.Linear(self.fusion.hidden_dim, num_classes)
			)

			self.modalities = modalities

		def forward(
				self,
				ct: Optional[torch.Tensor] = None,
				ecg: Optional[torch.Tensor] = None,
				demographics: Optional[torch.Tensor] = None,
				prosthesis_and_location: Optional[torch.Tensor] = None
		) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:

			ct_features = self.ct_feature_extractor(ct)
			ecg_features = self.ecg_feature_extractor(ecg)

			# Zero out modalities not in the specified list
			if 'ct' not in self.modalities:
				ct_features = torch.full_like(ct_features, float('nan'))
			if 'ecg' not in self.modalities:
				ecg_features = torch.full_like(ecg_features, float('nan'))
			if 'demographics' not in self.modalities:
				demographics = torch.zeros_like(demographics)

			fused_features, auxiliary_outputs = self.fusion(
				ct_features, ecg_features, demographics, prosthesis_and_location
			)

			logits = self.classifier(fused_features)

			return logits, auxiliary_outputs

		def compute_total_loss(self,
							   logits: torch.Tensor,
							   targets: torch.Tensor,
							   auxiliary_outputs: torch.Tensor,
							   classification_loss_fn: nn.Module,
							   ) -> Dict[str, torch.Tensor]:
			"""
			Compute total loss including classification and reconstruction losses.

			Args:
				logits: Model predictions
				targets: Ground truth labels
				auxiliary_outputs: Dict containing reconstruction losses
				classification_loss_fn: Loss function for classification (e.g., CrossEntropyLoss)

			Returns:
				dict with individual losses and total loss
			"""
			# Classification loss
			classification_loss = classification_loss_fn(logits, targets)

			# Reconstruction loss
			reconstruction_loss = auxiliary_outputs['reconstruction_losses']['total_reconstruction']

			# Total loss
			total_loss = classification_loss + reconstruction_loss.sum()

			return {
				'classification_loss': classification_loss,
				'reconstruction_loss': reconstruction_loss,
				'ct_reconstruction_loss': auxiliary_outputs['reconstruction_losses']['ct_reconstruction'],
				'ecg_reconstruction_loss': auxiliary_outputs['reconstruction_losses']['ecg_reconstruction'],
				'total_loss': total_loss
			}

	class SimpleMultiModalClassifier(nn.Module):
		"""Simple classifier with missing tokens."""

		def __init__(
				self,
				num_classes: int,
				demo_dim: int,
				ecg_fm_cfg: Dict,
				wav2vec2_cfg: Dict,
				dropout: float = 0.1,
				train_img_encoder: bool = False,
				train_ecg_encoder: bool = False,
				modalities: List[str] = ['ct', 'ecg', 'demographics'],
				ct_feature_extractor_ckpt: Optional[str] = None,
				ecg_feature_extractor_ckpt: Optional[str] = None
		):
			super().__init__()

			self.ct_feature_extractor = TP.CTFeatureExtractor(ckpt_path=ct_feature_extractor_ckpt)
			self.ecg_feature_extractor = TP.ECGFeatureExtractor(ecg_fm_cfg, wav2vec2_cfg,
																ckpt_path=ecg_feature_extractor_ckpt)

			self.train_img_encoder = train_img_encoder
			self.train_ecg_encoder = train_ecg_encoder

			for p in self.ct_feature_extractor.parameters():
				p.requires_grad = train_img_encoder
			for p in self.ecg_feature_extractor.parameters():
				p.requires_grad = train_ecg_encoder

			self.fusion = TP.SimpleMissingTokenFusion(
				ct_dim=768,
				ecg_dim=768,
				demo_dim=demo_dim,
				prosthesis_dim=768,
				hidden_dim=256,  # TODO: Add to configs
				dropout=dropout
			)

			self.classifier = nn.Sequential(
				nn.Dropout(dropout),
				nn.Linear(self.fusion.hidden_dim, num_classes)
			)

			self.modalities = modalities

		def forward(
				self,
				ct: Optional[torch.Tensor] = None,
				ecg: Optional[torch.Tensor] = None,
				demographics: Optional[torch.Tensor] = None,
				prosthesis_and_location: Optional[torch.Tensor] = None
		) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:

			ct_features = self.ct_feature_extractor(ct)
			ecg_features = self.ecg_feature_extractor(ecg)

			# Disable modalities not in the list
			if 'ct' not in self.modalities:
				ct_features = torch.full_like(ct_features, float('nan'))
			if 'ecg' not in self.modalities:
				ecg_features = torch.full_like(ecg_features, float('nan'))
			if 'demographics' not in self.modalities:
				demographics = torch.full_like(demographics, float('nan'))

			# print(f"{self.modalities=}")
			# print(f"{ct_features.min()=}\t{ct_features.max()=}")
			# print(f"{ecg_features.min()=}\t{ecg_features.max()=}")
			# print(f"{demographics.min()=}\t{demographics.max()=}")
			# print(f"{prosthesis_and_location.min()=}\t{prosthesis_and_location.max()=}")

			fused_features, auxiliary_outputs = self.fusion(
				ct_features, ecg_features, demographics, prosthesis_and_location
			)

			logits = self.classifier(fused_features)

			return logits, auxiliary_outputs

	class CrossAttentionMultiModalClassifier(nn.Module):
		"""Cross-attention based classifier."""

		def __init__(
				self,
				num_classes: int,
				demo_dim: int,
				ecg_fm_cfg: Dict,
				wav2vec2_cfg: Dict,
				dropout: float = 0.1,
				train_img_encoder: bool = False,
				train_ecg_encoder: bool = False,
				modalities: List[str] = ['ct', 'ecg', 'demographics'],
				use_learnable_token: bool = True,
				balance_method: Optional[str] = None,
				ct_feature_extractor_ckpt: Optional[str] = None,
				ecg_feature_extractor_ckpt: Optional[str] = None
		):
			super().__init__()

			self.ct_feature_extractor = TP.CTFeatureExtractor(ckpt_path=ct_feature_extractor_ckpt)
			self.ecg_feature_extractor = TP.ECGFeatureExtractor(ecg_fm_cfg, wav2vec2_cfg,
																ckpt_path=ecg_feature_extractor_ckpt)

			for p in self.ct_feature_extractor.parameters():
				p.requires_grad = train_img_encoder
			for p in self.ecg_feature_extractor.parameters():
				p.requires_grad = train_ecg_encoder

			self.fusion = TP.MultiModalCrossAttention(
				ct_dim=768,
				ecg_dim=768,
				demo_dim=demo_dim,
				prosthesis_dim=768,
				hidden_dim=256,
				num_heads=8,
				dropout=dropout,
				use_learnable_tokens=use_learnable_token,
				balance_method=balance_method
			)

			self.classifier = nn.Sequential(
				nn.Dropout(dropout),
				nn.Linear(self.fusion.hidden_dim, num_classes)
			)

			self.modalities = modalities

		def forward(
				self,
				ct: Optional[torch.Tensor] = None,
				ecg: Optional[torch.Tensor] = None,
				demographics: Optional[torch.Tensor] = None,
				prosthesis_and_location: Optional[torch.Tensor] = None
		) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:

			ct_features = self.ct_feature_extractor(ct)
			ecg_features = self.ecg_feature_extractor(ecg)

			# Disable modalities not in the list
			if 'ct' not in self.modalities:
				ct_features = torch.full_like(ct_features, float('nan'))
			if 'ecg' not in self.modalities:
				ecg_features = torch.full_like(ecg_features, float('nan'))
			if 'demographics' not in self.modalities:
				demographics = torch.full_like(demographics, float('nan'))

			fused_features, attention_weights = self.fusion(
				ct_features, ecg_features, demographics, prosthesis_and_location
			)

			logits = self.classifier(fused_features)

			return logits, attention_weights

	def init_model(self, model_args: dict = {}):
		"""Factory method to create models based on training mode."""
		random_seed = int(model_args.get('random_seed', 42))
		np.random.seed(random_seed)
		torch.manual_seed(random_seed)
		if torch.cuda.is_available():
			torch.cuda.manual_seed_all(random_seed)
		mode = model_args.get('training_mode', TP.TrainingMode.RECONSTRUCTION)

		common_args = {
			'num_classes': 1,
			'demo_dim': len(model_args['features']),
			'ecg_fm_cfg': model_args['ecg_fm_cfg'],
			'wav2vec2_cfg': model_args['wav2vec2_cfg'],
			'train_ecg_encoder': model_args.get('train_ecg_encoder', False),
			'train_img_encoder': model_args.get('train_img_encoder', False),
			'modalities': model_args['modalities'],
			'dropout': model_args.get('dropout', 0.1),
			'ct_feature_extractor_ckpt': model_args.get('ct_feature_extractor_ckpt'),
			'ecg_feature_extractor_ckpt': model_args.get('ecg_feature_extractor_ckpt')
		}

		if mode == TP.TrainingMode.RECONSTRUCTION:
			model = TP.MultiModalClassifier(
				**common_args,
				reconstruction_loss_weight=model_args.get('reconstruction_loss_weight', 1.0),
				reconstruction_loss_type=model_args.get('reconstruction_loss_type', 'cosine')
			)

		elif mode == TP.TrainingMode.SIMPLE_MISSING_TOKEN:
			model = TP.SimpleMultiModalClassifier(**common_args)

		elif mode == TP.TrainingMode.CROSS_ATTENTION:
			model = TP.CrossAttentionMultiModalClassifier(
				**common_args,
				use_learnable_token=model_args.get('attention_use_learnable_token', True),
				balance_method=model_args.get('attention_balance_method', None)
			)

		else:
			raise ValueError(f"Unknown training mode: {mode}")

		ckpt_path = model_args.get('ckpt', None)
		if ckpt_path is not None:
			if not Path(ckpt_path).is_file():
				raise FileNotFoundError("configured model checkpoint was not found")
			sd = torch.load(ckpt_path, map_location='cpu', weights_only=True)
			model.load_state_dict(sd)
			print('LOADING SD SUCCESSFUL')

		return model

	def tag_parameters(self, name: str):
		""" Tagging frozen parameters as 'local' so they are neither sent nor aggregated. """
		if name.startswith('ct_feature_extractor') and not self.model_args()["train_img_encoder"]:
			return {"local"}
		if name.startswith('ecg_feature_extractor') and not self.model_args()["train_ecg_encoder"]:
			return {"local"}
		return set()

	def init_optimizer(self, optimizer_args: dict = {}):
		"""Initializes and returns optimizer instance."""
		return torch.optim.AdamW(self.model().parameters(), **optimizer_args)

	##############################
	# Dataset Definition
	##############################

	class TAVIPMDataset(Dataset):

		def __init__(
				self,
				dataset_path: str,
				demographics: pd.DataFrame,
				prosthesis_names: list[str],
				features: list[str],
				preprocess_values,
				ct_type: str = 'aortic_root_roi',
				ct_target_size: tuple[int, int, int] | None = None,
				site_name: str = 'unknown',
				site_env_var: Optional[str] = 'FLOTO_SITE_ALIAS',
				site_dropout: float = 0.5,
				text_encoder_name_or_path: str = 'distilbert-base-uncased',
				text_encoder_cache_dir: Optional[str] = None,
				allow_model_download: bool = False,
				both_dropout: float = 0.0,
				unknown_locations: list = [],
				ct_norm_mode: str | None = None,
				ct_global_stats: tuple | None = None,
				dropna_subset: list = [],
				device: str = 'cuda',
				**kwargs
		):
			self.device = device

			# Read only a neutral alias on each node; never pass infrastructure IDs
			# through model arguments or embed institution names in model inputs.
			self.site_name = os.environ.get(site_env_var, site_name) if site_env_var else site_name
			if self.site_name != 'unknown':
				prefix, _, numeric_id = self.site_name.partition('_')
				if prefix != 'site' or not numeric_id.isdigit():
					raise ValueError("site aliases must use the form 'site_<number>'")
			if not 0.0 <= site_dropout <= 1.0:
				raise ValueError("site_dropout must be between 0 and 1")
			self.site_dropout = site_dropout
			self.text_encoder_name_or_path = text_encoder_name_or_path
			self.text_encoder_cache_dir = text_encoder_cache_dir
			self.dataset_path = Path(dataset_path)
			self.prosthesis_names = prosthesis_names
			self.features = features
			self.preprocess_values = preprocess_values
			self.both_dropout = both_dropout
			self.unknown_locations = unknown_locations
			self.ct_type = ct_type
			self.ct_target_size = ct_target_size
			self.ct_norm_mode = ct_norm_mode
			self.ct_global_stats = ct_global_stats

			self.patient_ids, self.meta, self.prosthesis, self.y = \
				self.preprocess_demographics(
					dataset_path, demographics, prosthesis_names,
					features, preprocess_values, ct_type, dropna_subset
				)

			loader_kwargs = {
				'cache_dir': self.text_encoder_cache_dir,
				'local_files_only': not allow_model_download,
			}
			self.tokenizer = DistilBertTokenizer.from_pretrained(
				self.text_encoder_name_or_path,
				**loader_kwargs,
			)
			self.bert = DistilBertModel.from_pretrained(
				self.text_encoder_name_or_path,
				**loader_kwargs,
			)

			self.bert.eval()

		@staticmethod
		def filter_demographics(
				dataset_path: str,
				df: pd.DataFrame,
				prosthesis_names: list = [],
				ct_type: str = 'aortic_root_roi',
				dropna_subset: list = [],
		) -> pd.DataFrame:

			"""
			Filters and cleans the demographic data, ensuring physical file availability for each patient.

			This method performs several data cleaning operations:
			1. Fills missing prosthesis names and sizes with default values.
			2. Standardizes legacy prosthesis names (e.g., 'Sapien 3' to 'Edwards Sapien 3').
			3. Filters the cohort to include only specified prosthesis types (if provided).
			4. Replaces specific artifact/categorical values in clinical features (8.0 -> 5.0 for lung disease/infarct).
			5. Drops records with missing values in specified critical columns (dropna_subset).
			6. Excludes patients who already had a pacemaker prior to admission (admission_pre_pm == 0).
			7. Verifies physical file existence, keeping only patients who have at least an ECG or a CT scan available.

			Args:
				dataset_path (str): The root directory path where patient folders are stored.
				df (pd.DataFrame): The raw demographic dataframe containing a 'PatientID' column.
				prosthesis_names (list, optional): A list of valid prosthesis names to keep.
					If empty, no filtering by prosthesis name is applied. Defaults to [].
				ct_type (str, optional): The name of the subdirectory containing the CT scans
					within the patient's folder. Defaults to 'aortic_root_roi'.
				dropna_subset (list, optional): A list of column names. Rows with NaN values
					in any of these columns will be dropped. Defaults to [].

			Returns:
				pd.DataFrame: The filtered and cleaned demographic dataframe.
			"""

			if 'prosthesis_name' not in df.columns.tolist():
				df['prosthesis_name'] = 'unknown'
			if 'prosthesis_size' not in df.columns.tolist():
				df['prosthesis_size'] = 0.0
			if 'PatientID' not in df.columns:
				raise ValueError("demographics.csv must contain a PatientID column")

			df = df.copy()
			df['PatientID'] = df['PatientID'].astype(str)
			invalid_ids = df['PatientID'].map(
				lambda value: value in {'', '.', '..'} or Path(value).name != value
			)
			if invalid_ids.any():
				raise ValueError("PatientID values must be opaque single-directory identifiers")

			def rename_prosthesis(x):
				if x == 'Sapien 3':
					return 'Edwards Sapien 3'
				elif x == 'Edwards Sapien S4':
					return 'Edwards Sapien 3 Ultra'
				return x

			df['prosthesis_name'] = df.prosthesis_name.apply(rename_prosthesis).copy()
			# Filter for specific prosthesis names
			if len(prosthesis_names) >= 1:
				df = df[df['prosthesis_name'].isin(prosthesis_names)].copy()

			if 'admission_lung_disease' in df.columns:
				df['admission_lung_disease'] = df['admission_lung_disease'].replace({8.0: 5.0})
			if 'admission_infarct' in df.columns:
				df['admission_infarct'] = df['admission_infarct'].replace({8.0: 5.0})

			df = df.dropna(subset=dropna_subset)
			if 'admission_pre_pm' in df.columns:
				df = df[(df.admission_pre_pm == 0)].copy()

			# File availability checks
			root = Path(dataset_path)
			pids_only_ecg = [str(p) for p in df.PatientID if
							 (root / str(p)  / 'pre_ecg').exists() and not (root / str(p)  / ct_type).exists()]
			pids_only_ct = [str(p)  for p in df.PatientID if
							not (root / str(p)  / 'pre_ecg').exists() and (root / str(p)  / ct_type).exists()]
			pids_both = [str(p)  for p in df.PatientID if
						 (root / str(p)  / 'pre_ecg').exists() and (root / str(p)  / ct_type).exists()]

			pids = np.unique(pids_only_ecg + pids_only_ct + pids_both).tolist()

			df = df[df.PatientID.isin(pids)]

			return df

		def preprocess_demographics(
				self,
				dataset_path: str,
				df: pd.DataFrame,
				prosthesis_names: list,
				features: list[str],
				preprocess_values: list,
				ct_type: str = 'aortic_root_roi',
				dropna_subset: list = []
		) -> tuple[pd.Series, pd.DataFrame, pd.Series, pd.Series]:

			""" Demographics values preprocessing.
				Calls filter_demographics(), then normalizes the features.

				Returns four dictionaries:
					- Patient ID per sample
					- (Filtered) features per sample
					- Prosthesis metadata per sample
					- Pacemaker info (ground truth target) per sample

			"""

			df = self.filter_demographics(dataset_path, df, prosthesis_names, ct_type, dropna_subset)

			for c in features:  # +cfg.confounders:

				# Skip features we don't want
				if c not in preprocess_values:
					continue

				c_stats = preprocess_values[c]

				# Fill features we want but don't exist with NaN
				# ecg_atrial_fibrillation not yet in MUC
				if c not in df.columns:
					df[c] = np.nan
					continue

				# No operation for binary features
				if c_stats['type'] == 'binary':
					continue
				# Standardisation for continous features
				elif c_stats['type'] == 'continuous':
					df.loc[:, c] = (df[c] - c_stats['mean']) / c_stats['std']
				# Ordinal encoding for categorical features
				elif c_stats['type'] == 'category':
					df.loc[:, c] = pd.Categorical(df[c], categories=c_stats['values']).codes


			# Fill missing values with 0s for specified columns
			df = df.fillna(value={
				'admission_infarct': 0.0,
				'admission_cerebro': 0.0,
			})

			return df['PatientID'], df[features], df[['prosthesis_name', 'prosthesis_size']], df['discharge_pm_ops']

		@torch.no_grad()
		def prosthesis_name_and_location_transform(self, prosthesis_name: str, prosthesis_size: str):
			site = self.site_name

			###############################
			# node to unknown for inference
			if site in self.unknown_locations:
				site = 'unknown'
			elif np.random.random() < self.site_dropout:
				site = 'unknown'
			###############################

			text = f"A {prosthesis_name} prosthesis of size {prosthesis_size} mm was used at {site}"
			inputs = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True)
			# inputs = {k: v.to(self.device) for k, v in inputs.items()}
			inputs = {k: v for k, v in inputs.items()}
			outputs = self.bert(**inputs)
			embeddings = outputs.last_hidden_state.mean(dim=1)[0]
			return embeddings

		@staticmethod
		def img_transform(
				img: torch.Tensor,
				mode: str | None = None,
				global_stats: tuple[float, float] | None = None,
				target_size: tuple[int, int, int] | None = None,
		) -> torch.Tensor:

			"""CT data transformation."""

			clip_range = (-200, 800)
			img = torch.clamp(img, *clip_range)

			match mode:
				case 'global':
					assert global_stats is not None, \
						"mode='global' requires global_stats to be set"
					# Standardize using global ds statistics
					img = (img - global_stats[0]) / (global_stats[1] + 1e-9)
				case 'clip_range':
					# Normalize to [0, 1] using the clip range
					img = (img - clip_range[0]) / (clip_range[1] - clip_range[0])
				case 'local':
					img = (img - img.min()) / (img.max() - img.min() + 1e-9)
					img = (img - img.mean()) / (img.std() + 1e-9)
				case None:
					pass
				case _:
					raise RuntimeError

			# TODO: resizing?
			img = img.unsqueeze(0).unsqueeze(0)
			if target_size is not None:
				img = F.interpolate(
					img,
					size=target_size,
					mode='trilinear',
					align_corners=False,
				)
			return img.squeeze(0).float()

		@staticmethod
		def ecg_transform(ecg: np.ndarray) -> torch.Tensor:

			"""ECG data transformation."""

			org_sample_rate = ecg.shape[1] / 10

			ECG_FM_LEAD_ORDER = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
			SAMPLE_RATE = 500
			N_SAMPLES = SAMPLE_RATE * 5

			ECG_FM_SCHEMA = ECGInputSchema(
				sample_rate=SAMPLE_RATE,
				expected_lead_order=ECG_FM_LEAD_ORDER,
				required_num_samples=N_SAMPLES,
			)

			ECG_FM_TRANSFORMS = [
				ReorderLeads(
					expected_order=ECG_FM_LEAD_ORDER,
					missing_lead_strategy='raise',
				),
				LinearResample(desired_sample_rate=SAMPLE_RATE),
				HandleConstantLeads(strategy='zero'),
				Standardize(),
				SegmentNonoverlapping(segment_length=N_SAMPLES),
			]

			metadata = ECGMetadata(
				sample_rate=org_sample_rate,
				num_samples=ecg.shape[1],
				lead_names=['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6'],
				unit=None,
				input_start=0,
				input_end=ecg.shape[1],
			)
			# metadata.file = self.file_paths[idx]
			inp = ECGInput(ecg, metadata)
			sample = ECGSample(
				inp,
				ECG_FM_SCHEMA,
				ECG_FM_TRANSFORMS,
			)
			# I could return all segments, for now only one random one  # TODO: ?
			source = torch.from_numpy(sample.out).float()
			idx = np.random.choice(source.shape[0], 1, replace=False)[0]
			return source[idx]

		def load_ct(self, patient_id: str) -> torch.Tensor:

			""" Reads and processes CT data. """

			try:
				img = sitk.ReadImage(next(self.dataset_path.glob(f'{patient_id}/{self.ct_type}/*.nii.gz')))
				img = sitk.GetArrayFromImage(img)
				img = torch.from_numpy(img.copy()).float()
				img = self.img_transform(
					img,
					mode=self.ct_norm_mode,
					global_stats=self.ct_global_stats,
					target_size=self.ct_target_size,
				)
				return img
			except (StopIteration, OSError, RuntimeError):
				# Do not emit sample identifiers or private filesystem paths to logs.
				TP.logger.warning('Failed to load a CT sample; substituting a missing-modality tensor.')
				return torch.full((1, *self.ct_target_size), float('nan'))

		def load_ecg(self, patient_id: str) -> torch.Tensor:

			"""	Reads in and processes ECG data. """

			try:
				ecg = sitk.ReadImage(next(self.dataset_path.glob(f'{patient_id}/pre_ecg/*.nii.gz')))
			except (StopIteration, OSError, RuntimeError):
				# Do not emit sample identifiers or private filesystem paths to logs.
				TP.logger.warning('Failed to load an ECG sample; substituting a missing-modality tensor.')
				return torch.full((12, 2500), float('nan'))  # TODO: Add to config / read target sample rate

			ecg = sitk.GetArrayFromImage(ecg).T
			ecg = self.ecg_transform(ecg)

			return ecg

		def __getitem__(self, index):

			try:

				patient_id = self.patient_ids.iloc[index]
				meta = self.meta.iloc[index]
				meta = torch.tensor(meta.tolist()).float()
				y = torch.tensor(self.y.iloc[index].tolist()).float()
				ecg = self.load_ecg(patient_id)
				ct = self.load_ct(patient_id)
				# prosthesis_type, prosthesis_size = self.prosthesis.iloc[index].tolist()
				prosthesis_and_location = self.prosthesis_name_and_location_transform(*self.prosthesis.iloc[index].tolist())

				# Apply both_dropout only when both modalities are present
				if self.both_dropout > 0.0:
					# Check if both modalities are present (not all NaN)
					ecg_present = not torch.isnan(ecg).all()
					ct_present = not torch.isnan(ct).all()

					if ecg_present and ct_present:
						# Randomly decide whether to apply dropout
						if np.random.random() < self.both_dropout:
							# Randomly choose which modality to set to NaN
							if np.random.random() < 0.5:
								# Set ECG to NaN
								ecg = torch.full_like(ecg, float('nan'))
							else:
								# Set CT to NaN
								ct = torch.full_like(ct, float('nan'))

				# Flatten the modalities into the single tensor expected by Fed-BioMed.
				ecg_flat = ecg.reshape(-1)
				ct_flat = ct.reshape(-1)
				meta_flat = meta.reshape(-1)
				pros_flat = prosthesis_and_location.reshape(-1)

				# training_step/testing_step unpack this transport representation.
				fbm_compliant_data = torch.cat([ecg_flat, ct_flat, meta_flat, pros_flat])

				return fbm_compliant_data, y


			except Exception:
				# Avoid persisting exceptions that can contain sample IDs or private paths.
				raise RuntimeError(f"Failed to construct dataset item at index {index}") from None

		def __len__(self):
			return len(self.patient_ids)

	@staticmethod
	def create_modality_stratified_split(
		demographics: pd.DataFrame,
		test_size: float = 0.2,
		random_state: int = 42,
		target_col: str = 'discharge_pm_ops'
	) -> Tuple[pd.DataFrame, pd.DataFrame]:

		"""
		Create a split stratified by outcome and modality availability.

		Args:
			demographics: DataFrame with patient data
			test_size: Proportion of test set
			random_state: Random seed
			target_col: Target variable column name

		Returns:
			demo_train, demo_test: Split dataframes
		"""

		# Create modality indicators
		df = demographics.copy()

		# Create modality pattern
		def get_modality_pattern(row):
			if row['has_ecg'] and row['has_ct']:
				return 'both'
			elif row['has_ecg'] and not row['has_ct']:
				return 'ecg_only'
			elif not row['has_ecg'] and row['has_ct']:
				return 'ct_only'
			else:
				return 'none'

		df['modality_pattern'] = df.apply(get_modality_pattern, axis=1)

		# Create combined stratification variable (outcome + modality).
		df['stratify_var'] = (df[target_col].astype(str) + '_' + df['modality_pattern'])

		demo_train, demo_test = train_test_split(
			df.drop(['has_ecg', 'has_ct', 'modality_pattern', 'stratify_var'], axis=1),
			test_size=test_size,
			stratify=df['stratify_var'],
			random_state=random_state
		)

		return demo_train, demo_test


	@staticmethod
	def make_balanced_sampler(dataset: Dataset, seed: int = 42):
		labels_raw = np.asarray(dataset.y)

		if labels_raw.ndim != 1 or len(labels_raw) != len(dataset):
			raise ValueError("dataset.y must contain one label per sample")

		if not np.isin(labels_raw, [0, 1]).all():
			invalid = np.unique(labels_raw[~np.isin(labels_raw, [0, 1])])
			raise ValueError(f"Expected binary labels 0/1, found: {invalid}")

		labels = torch.as_tensor(labels_raw.astype(np.int64), dtype=torch.long)
		counts = torch.bincount(labels, minlength=2)

		if (counts == 0).any():
			raise ValueError(f"Cannot balance a one-class training split: counts={counts.tolist()}")

		# Each class receives the same total sampling probability.
		class_weights = 1.0 / counts.to(torch.double)
		sample_weights = class_weights[labels]

		generator = torch.Generator()
		generator.manual_seed(int(seed))

		return WeightedRandomSampler(
			weights=sample_weights,
			num_samples=len(dataset),  # Preserve original epoch/node size
			replacement=True,
			generator=generator,
		)

	def training_data(self):

		"""Initializes and returns training data as DataManager instance."""

		try:

			args = self.model_args()
			random_seed = int(args.get('random_seed', 42))
			np.random.seed(random_seed)
			torch.manual_seed(random_seed)
			if torch.cuda.is_available():
				torch.cuda.manual_seed_all(random_seed)
			data_dir = self.dataset_path
			dataset_args = dict(args)
			dataset_args.pop("sampler", None)
			dataset_args["dataset_path"] = data_dir

			df = pd.read_csv(Path(data_dir) / 'demographics.csv')

			df = TP.TAVIPMDataset.filter_demographics(data_dir, df, args['prosthesis_names'],
											  ct_type=args['ct_type'],
											  dropna_subset=args.get('dropna_subset', []))

			# TODO: Redundant with filter_demographics()?
			df['has_ecg'] = df.PatientID.apply(lambda x: (Path(data_dir) / x / 'pre_ecg').exists())
			df['has_ct'] = df.PatientID.apply(lambda x: (Path(data_dir) / x / args['ct_type']).exists())

			modality_mode = args.get('modality_mode', 'all')
			if modality_mode == 'only_ecg':
				df = df[df.has_ecg]
			elif modality_mode == 'only_ct':
				df = df[df.has_ct]
			elif modality_mode == 'both':
				df = df[df.has_ecg & df.has_ct]
			elif modality_mode != 'all':
				raise ValueError("modality_mode must be all, only_ecg, only_ct, or both")

			if 'discharge_pm_ops' not in df.columns:
				raise ValueError("demographics.csv is missing the binary target column")
			if not df['discharge_pm_ops'].isin([0, 1]).all():
				raise ValueError("the target column must contain only 0 and 1")
			if df['discharge_pm_ops'].nunique() < 2:
				raise ValueError("the filtered cohort must contain both target classes")

			demo_train, demo_test = self.create_modality_stratified_split(
				df,
				test_size=args['test_size'],
				random_state=args.get('random_seed', 42),
				target_col='discharge_pm_ops'
			)

			trainset = TP.TAVIPMDataset(
				demographics=demo_train,
				**dataset_args
			)
			print(f">>> {len(trainset)=} <<<")
			self.trainset = trainset

			dataset_args.pop("both_dropout", None)
			testset = TP.TAVIPMDataset(
				demographics=demo_test,
				both_dropout=0.0,
				**dataset_args
			)
			print(f">>> {len(testset)=} <<<")
			self.custom_testset = testset

			return DataManager(trainset, shuffle=True)

		except Exception:
			# Exceptions can contain private paths; do not persist or print them here.
			raise RuntimeError("Failed to initialize the configured dataset") from None

	def set_data_loaders(
		self,
		train_data_loader=None,
		test_data_loader=None,
		*args, **kwargs
	) -> None:

		"""
			Bypass FBM's internal DataManager split logic completely.
			We manually construct both PyTorch DataLoaders using our stratified sets.
		"""

		try:

			# DO NOT call super().set_data_loaders()!
			# It triggers FBM's automatic test_ratio subsetting.

			assert self.trainset is not None, "trainset was not initialized"
			assert self.custom_testset is not None, "custom_testset was not initialized"

			train_loader_args = dict(self._loader_args)
			train_loader_args.pop("shuffle", None)
			train_loader_args.pop("sampler", None)

			model_args = self.model_args()
			sampler_mode = model_args.get("sampler", None)

			# TODO: Load from model args
			if sampler_mode == "over":
				sampler_seed = model_args.get("sampler_seed")
				if sampler_seed is None:
					raise ValueError("sampler='over' requires a sampler_seed")
				train_loader_args["sampler"] = TP.make_balanced_sampler(
					self.trainset,
					seed=int(sampler_seed),
				)
				self.bce_pos_weight = 1.0
				print(
					f"{'#'*10} Using oversampling with sampler_seed={sampler_seed}, "
					f"bce_pos_weight={self.bce_pos_weight}"
				)
			elif sampler_mode in (None, "none"):
				train_loader_args["shuffle"] = True
				self.bce_pos_weight = float(model_args.get("bce_pos_weight", 1.0))
				print(f"{'#' * 10} Using unweighted sampling with {self.bce_pos_weight=}")
			else:
				raise ValueError(f"Unknown sampler mode: {sampler_mode}")

			# Explicitly build the Training Loader
			self.training_data_loader = PytorchDataLoader(
				self.trainset,
				**train_loader_args
			)

			# Explicitly build the testing loader
			test_loader_args = dict(self._loader_args)
			test_loader_args.pop("shuffle", None)
			test_loader_args.pop("sampler", None)

			self.testing_data_loader = PytorchDataLoader(
				self.custom_testset,
				shuffle=False,
				**test_loader_args,
			)

		except Exception:
			raise RuntimeError("Failed to initialize the configured data loaders") from None

	##############################
	# Training & Testing
	##############################

	def store_losses(self, losses, mode):

		"""Store per-sample losses for tracking."""

		loss_tracker = self.train_losses if mode == 'train' else self.test_losses
		ds = self.training_data_loader.dataset if mode == 'train' else self.testing_data_loader.dataset

		# Reset loss tracker at beginning of new epoch
		if len(loss_tracker['total']) == len(ds):
			for key in loss_tracker:
				loss_tracker[key] = []

		# Store individual loss components - convert to per-sample if tensor has batch dimension
		classification_loss = losses['classification_loss']
		if hasattr(classification_loss, 'shape') and len(classification_loss.shape) > 0:
			# If batch dimension exists, store per sample
			for i in range(classification_loss.shape[0]):
				loss_tracker['classification'].append(classification_loss[i].item())
				# loss_tracker['reconstruction_total'].append(losses['reconstruction_loss'][i].item() if hasattr(losses['reconstruction_loss'], 'shape') and len(losses['reconstruction_loss'].shape) > 0 else losses['reconstruction_loss'].item())
				# loss_tracker['ct_reconstruction'].append(losses['ct_reconstruction_loss'][i].item() if hasattr(losses['ct_reconstruction_loss'], 'shape') and len(losses['ct_reconstruction_loss'].shape) > 0 else losses['ct_reconstruction_loss'].item())
				# loss_tracker['ecg_reconstruction'].append(losses['ecg_reconstruction_loss'][i].item() if hasattr(losses['ecg_reconstruction_loss'], 'shape') and len(losses['ecg_reconstruction_loss'].shape) > 0 else losses['ecg_reconstruction_loss'].item())
				loss_tracker['total'].append(
					losses['total_loss'][i].item() if hasattr(losses['total_loss'], 'shape') and len(
						losses['total_loss'].shape) > 0 else losses['total_loss'].item())
		else:
			# Scalar losses - store as single value (this happens when batch losses are already averaged)
			loss_tracker['classification'].append(losses['classification_loss'].item())
			loss_tracker['reconstruction_total'].append(losses['reconstruction_loss'].item())
			loss_tracker['ct_reconstruction'].append(losses['ct_reconstruction_loss'].item())
			loss_tracker['ecg_reconstruction'].append(losses['ecg_reconstruction_loss'].item())
			loss_tracker['total'].append(losses['total_loss'].item())

	def store_results(self, logits, y, ct, ecg, mode):

		"""Store per-sample predictions and modality availability."""

		ds = self.training_data_loader.dataset if mode == 'train' else self.testing_data_loader.dataset
		if mode == 'train':
			if len(self.train_results) == len(ds):
				self.train_results = []
		elif mode == 'test':
			if len(self.test_results) == len(ds):
				self.test_results = []

		logits = logits.detach().cpu().flatten().tolist()
		y = y.cpu().flatten().tolist()

		ecg_mask = (~torch.isnan(ecg).all(dim=(1, 2))).int().cpu().flatten().tolist()
		ct_mask = (~torch.isnan(ct).all(dim=(1, 2, 3, 4))).int().cpu().flatten().tolist()

		for l, gt, c, e in zip(logits, y, ct_mask, ecg_mask):
			res = {'logits': l, 'y_true': gt, 'ct': c, 'ecg': e}
			if mode == 'train':
				self.train_results.append(res)
			elif mode == 'test':
				self.test_results.append(res)

	@staticmethod
	def bce_metrics(y_true, y_prob):
		if type(y_prob) == torch.Tensor:
			pred = (y_prob > 0.5).long()
		elif type(y_prob) == pd.Series:
			pred = (y_prob.to_numpy() > 0.5).astype(int)
		else:
			raise ValueError
		if np.unique(np.asarray(y_true)).size < 2:
			roc_auc = float('nan')
			auprc = float('nan')
		else:
			roc_auc = roc_auc_score(y_true, y_prob)
			auprc = average_precision_score(y_true, y_prob)
		balanced_accuracy = balanced_accuracy_score(y_true, pred)
		return roc_auc, auprc, balanced_accuracy

	@staticmethod
	def roc(y_true, y_prob):
		fpr_common = np.linspace(0, 1, num=100)
		if np.unique(np.asarray(y_true)).size < 2:
			return fpr_common, np.full_like(fpr_common, np.nan)
		fpr, tpr, _ = roc_curve(y_true, y_prob)
		tpr_interpolated = np.interp(fpr_common, fpr, tpr)
		return fpr_common, tpr_interpolated

	def evaluate_by_modality(self, mode: str, verbose: bool = False) -> dict:

		"""Evaluate metrics by modality availability."""

		metrics = {}

		df = pd.DataFrame(getattr(self, f'{mode}_results'))

		modality_subsets = {
			'none': (df.ct == 0) & (df.ecg == 0),
			'ct': (df.ct == 1) & (df.ecg == 0),
			'ecg': (df.ct == 0) & (df.ecg == 1),
			'both': (df.ct == 1) & (df.ecg == 1),
		}

		roc_auc, auprc, balanced_accuracy = self.bce_metrics(df.y_true, df.logits)
		metrics[f'roc_auc_{mode}'] = roc_auc
		metrics[f'auprc_{mode}'] = auprc
		metrics[f'balanced_accuracy_{mode}'] = balanced_accuracy

		fpr, tpr = self.roc(df.y_true, df.logits)
		for i in range(len(tpr)):
			metrics[f'tpr_{mode}_{i}'] = tpr[i]

		for name, subset in modality_subsets.items():
			subset_df = df[subset]
			num_samples = len(subset_df)
			metrics[f'num_samples_{mode}_{name}'] = num_samples

			if verbose:
				print(f"  {name}: {num_samples} samples")
			if num_samples > 0:
				num_positive = int(subset_df.y_true.sum())
				metrics[f'num_positive_{mode}_{name}'] = num_positive

				roc_auc, auprc, balanced_accuracy = self.bce_metrics(subset_df.y_true, subset_df.logits)
				metrics[f'roc_auc_{mode}_{name}'] = roc_auc
				metrics[f'auprc_{mode}_{name}'] = auprc
				metrics[f'balanced_accuracy_{mode}_{name}'] = balanced_accuracy

				# Compute ROC curve for this subset
				fpr_subset, tpr_subset = self.roc(subset_df.y_true, subset_df.logits)
				for i in range(len(tpr_subset)):
					metrics[f'tpr_{mode}_{name}_{i}'] = tpr_subset[i]
			else:
				metrics[f'num_positive_{mode}_{name}'] = 0
				metrics[f'roc_auc_{mode}_{name}'] = 0.0
				metrics[f'auprc_{mode}_{name}'] = 0.0
				metrics[f'balanced_accuracy_{mode}_{name}'] = 0.0

		return metrics

	def training_step(self, data, target):
		device = self._device
		model = self.model().to(device)
		model.train()

		try:
			ecg_size = 30000
			ct_size = self.trainset.ct_target_size
			demo_size = len(self.model_args()['features'])
			ecg = data[:, 0:ecg_size].reshape(-1, 12, 2500).to(device)
			ct = data[:, ecg_size:ecg_size+np.prod(ct_size)].reshape(-1, 1, *ct_size).to(device)
			demographics = data[:, ecg_size+np.prod(ct_size):ecg_size+np.prod(ct_size)+demo_size].to(device)
			prosthesis_and_location = data[:, ecg_size+np.prod(ct_size)+demo_size:].squeeze(1).to(device)

			target = target.to(device).float()[:, None]

			pos_weight = target.new_tensor([self.bce_pos_weight])

			# Forward pass
			logits, auxiliary_outputs = model(ct=ct, ecg=ecg, demographics=demographics,
											  prosthesis_and_location=prosthesis_and_location)

			# Compute loss based on model type
			loss_fn = lambda l, t: F.binary_cross_entropy_with_logits(l, t, pos_weight=pos_weight)

			if hasattr(model, 'compute_total_loss'):
				# Reconstruction model with detailed loss components
				losses = model.compute_total_loss(logits, target, auxiliary_outputs, loss_fn)
				total_loss = losses['total_loss']
			else:
				# Simple/CrossAttention models with only classification loss
				classification_loss = loss_fn(logits, target)
				losses = {
					'classification_loss': classification_loss,
					'reconstruction_loss': torch.tensor(0.0, device=classification_loss.device),
					'ct_reconstruction_loss': torch.tensor(0.0, device=classification_loss.device),
					'ecg_reconstruction_loss': torch.tensor(0.0, device=classification_loss.device),
					'total_loss': classification_loss
				}
				total_loss = classification_loss

			# Store losses and results
			self.store_losses(losses, 'train')
			self.store_results(torch.sigmoid(logits), target, ct, ecg, 'train')

			return total_loss

		except Exception:
			# Failing loudly avoids silently training on a fabricated zero loss.
			raise RuntimeError("Training step failed") from None


	def testing_step(self, data, target):
		device = self._device
		model = self.model().to(device)
		model.eval()

		try:
			ecg_size = 30000
			ct_size = self.trainset.ct_target_size
			demo_size = len(self.model_args()['features'])
			ecg = data[:, 0:ecg_size].reshape(-1, 12, 2500).to(device)
			ct = data[:, ecg_size:ecg_size + np.prod(ct_size)].reshape(-1, 1, *ct_size).to(device)
			demographics = data[:, ecg_size + np.prod(ct_size):ecg_size + np.prod(ct_size) + demo_size].to(device)
			prosthesis_and_location = data[:, ecg_size + np.prod(ct_size) + demo_size:].squeeze(1).to(device)

			target = target.to(device).float()[:, None]

			logits, auxiliary_outputs = model(
				ct=ct,
				ecg=ecg,
				demographics=demographics,
				prosthesis_and_location=prosthesis_and_location
			)

			# Compute loss based on model type
			loss_fn = lambda l, t: F.binary_cross_entropy_with_logits(l, t, pos_weight=None)

			if hasattr(model, 'compute_total_loss'):
				# Reconstruction model with detailed loss components
				losses = model.compute_total_loss(logits, target, auxiliary_outputs, loss_fn)
				total_loss = losses['total_loss']
			else:
				# Simple/CrossAttention models with only classification loss
				classification_loss = loss_fn(logits, target)
				losses = {
					'classification_loss': classification_loss,
					'reconstruction_loss': torch.tensor(0.0, device=classification_loss.device),
					'ct_reconstruction_loss': torch.tensor(0.0, device=classification_loss.device),
					'ecg_reconstruction_loss': torch.tensor(0.0, device=classification_loss.device),
					'total_loss': classification_loss
				}
				total_loss = classification_loss

			# Store losses and results for global calculation
			self.store_losses(losses, 'test')
			self.store_results(torch.sigmoid(logits), target, ct, ecg, 'test')

			# Batch-level metrics
			target = target.to(torch.uint8)
			acc = accuracy(logits, target, task='binary')

			return {
				'loss': total_loss.item(),
				'acc': acc.item(),
			}

		except Exception:
			raise RuntimeError("Testing step failed") from None

	@staticmethod
	def _loss_to_sum(loss, batch_size: int) -> float:

		""" Converts scalar mean losses or per-sample losses into a sum over samples. """

		loss = loss.detach()

		if loss.ndim == 0:
			return loss.cpu().item() * batch_size

		if loss.shape[0] != batch_size:
			raise ValueError("Loss shape doesn't match batch size.")

		per_sample = loss.reshape(batch_size, -1).mean(dim=1)
		return per_sample.sum().cpu().item()

	def _validation_batch(self, model, data, target):

		""" Validation forward pass + loss computation. """

		device = self._device

		ecg_size = 30000
		ct_size = self.trainset.ct_target_size
		ct_numel = int(np.prod(ct_size))
		demo_size = len(self.model_args()['features'])

		ecg = data[:, :ecg_size].reshape(-1, 12, 2500).to(device)
		ct = data[:, ecg_size:ecg_size + ct_numel].reshape(-1, 1, *ct_size).to(device)
		demographics = data[:, ecg_size + ct_numel:ecg_size + ct_numel + demo_size].to(device)
		prosthesis_and_location = data[:, ecg_size + ct_numel + demo_size:].squeeze(1).to(device)

		target = target.to(device).float().reshape(-1, 1)

		logits, auxiliary_outputs = model(
			ct=ct,
			ecg=ecg,
			demographics=demographics,
			prosthesis_and_location=prosthesis_and_location
		)

		# Compute loss based on model type
		loss_fn = lambda l, t: F.binary_cross_entropy_with_logits(l, t, pos_weight=None)

		if hasattr(model, 'compute_total_loss'):

			# Reconstruction model with detailed loss components
			losses = model.compute_total_loss(logits, target, auxiliary_outputs, loss_fn)

		else:

			# Simple/CrossAttention models with only classification loss
			classification_loss = loss_fn(logits, target)

			losses = {
				'classification_loss': classification_loss,
				'reconstruction_loss': torch.tensor(0.0, device=classification_loss.device),
				'ct_reconstruction_loss': torch.tensor(0.0, device=classification_loss.device),
				'ecg_reconstruction_loss': torch.tensor(0.0, device=classification_loss.device),
				'total_loss': classification_loss
			}

		return logits, target, losses


	def testing_routine(
            self,
            metric,
            metric_args,
            history_monitor,
            before_train
    ) -> None:

		""" Overriding testing routine for whole-validation-split evaluation. """

		try:

			del metric, metric_args

			if self.testing_data_loader is None:
				raise RuntimeError("No testing data loader configured.")

			dl = self.testing_data_loader
			n_batches = len(dl)

			if n_batches == 0:
				raise RuntimeError("Testing loader is empty.")

			model = self.model().to(self._device)
			was_training = model.training
			model.eval()

			loss_key_maps = {
				'classification_loss': 'classification_loss',
				'reconstruction_loss': 'reconstruction_loss',
				'ct_reconstruction_loss': 'ct_reconstruction_loss',
				'ecg_reconstruction_loss': 'ecg_reconstruction_loss',
				'loss': 'total_loss'
			}
			loss_sums = {name: 0.0 for name in loss_key_maps}

			y_true_parts, y_prob_parts = [], []

			n_samples = 0

			try:
				with torch.no_grad():
					for data, target in dl:
						data, target = data.to(self._device), target.to(self._device)
						logits, target, losses = self._validation_batch(model, data, target)

						batch_size = target.shape[0]
						n_samples += batch_size

						for metric_name, loss_name in loss_key_maps.items():
							loss_sums[metric_name] += self._loss_to_sum(losses[loss_name], batch_size)

						y_true_parts.append(target.detach().reshape(-1).to(torch.int64).cpu().numpy())
						y_prob_parts.append(torch.sigmoid(logits).detach().reshape(-1).cpu().numpy())

						del losses, logits, target

			finally:
				model.train(was_training)

			if n_samples == 0:
				raise RuntimeError("No validation samples were evaluated.")

			y_true = np.concatenate(y_true_parts)
			y_prob = np.concatenate(y_prob_parts)
			y_pred = (y_prob > 0.5).astype(np.int64)

			tp = int(((y_true == 1) & (y_pred == 1)).sum())
			tn = int(((y_true == 0) & (y_pred == 0)).sum())
			fp = int(((y_true == 0) & (y_pred == 1)).sum())
			fn = int(((y_true == 1) & (y_pred == 0)).sum())

			loss_metrics_dict = {
				name: float(loss_sum / n_samples) for name, loss_sum in loss_sums.items()
			}

			def safe_div(num, den):
				return float(num / den) if den else 0.0

			loss_metrics_dict.update({
				"accuracy": safe_div(tp + tn, n_samples),
				#"precision": safe_div(tp, tp + fp),
				#"recall": safe_div(tp, tp + fn),
				"f1": safe_div(2 * tp, 2 * tp + fp + fn),
				#"n_samples": int(n_samples),
				#"n_positive": int(y_true.sum()),
				#"n_negative": int((y_true == 0).sum()),
				#"tp": float(tp),
				#"tn": float(tn),
				#"fp": float(fp),
				#"fn": float(fn),
			})

			# Catching AUROC one-class computation
			if np.unique(y_true).size >= 2:
				loss_metrics_dict.update({
					"auroc": float(roc_auc_score(y_true, y_prob)),
					"auprc": float(average_precision_score(y_true, y_prob)),
				})
			else:
				loss_metrics_dict["ranking_metrics_defined"] = 0.0

			# Send validation/testing results to TensorBoard / researcher monitor
			if history_monitor is not None:
				history_monitor.add_scalar(
					metric=loss_metrics_dict,
					iteration=n_batches,
					epoch=None,
					test=True,
					test_on_global_updates=before_train,
					test_on_local_updates=not before_train,
					total_samples=n_samples,
					batch_samples=n_samples,
					num_batches=n_batches,
				)

		except Exception:
			raise RuntimeError("Validation routine failed") from None
