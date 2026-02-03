"""
neuro_prior_dgp.py - Neuro-Prior Data Generating Process Sampling

Implements continuous DGP sampling π(θ) for CausalPFN training.

KEY FUNCTIONS (maintaining spec names):
- sample_DGP_from_neuro_prior(): Sample θ ~ π(θ)
- generate_treatment_propensity(X_anatomy, θ): Generate W ~ Bernoulli(π)
- generate_outcomes(X_anatomy, θ): Generate Y_0, Y_1, τ_0, τ_1, CATE

DEPARTURE FROM INTERSYNTH:
- Continuous distribution π(θ) instead of 22,528 discrete scenarios
- γ ~ Beta(2,2) instead of 11 discrete levels
- β ~ Mixture instead of 4 discrete levels
- α ~ Mixture instead of 4 discrete levels

Usage:
    python neuro_prior_dgp.py \
        --lesionpath /path/to/lesions \
        --discopath /path/to/disconnectomes \
        --vae_lesion_path /path/to/vae_lesion.pth \
        --vae_disco_path /path/to/vae_disco.pth \
        --atlaspath /path/to/atlases \
        --n_dgps 160000 \
        --n_patients_per_dgp 1024 \
        --stage 1
"""

import argparse
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any

import nibabel as nib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm, trange

# Import VAE from representation module
from representation_neuro_prior import VAE3D


# =============================================================================
# CONSTANTS
# =============================================================================

# 16 NeuroQuery Functional Networks
NEUROQUERY_16_NETWORKS = [
    "hearing",
    "language",
    "introspection",
    "cognition",
    "mood",
    "memory",
    "aversion",
    "coordination",
    "interoception",
    "sleep",
    "reward",
    "visual_recognition",
    "visual_perception",
    "spatial_reasoning",
    "motor",
    "somatosensory",
]

# Severity weights for D_baseline computation
SEVERITY_WEIGHTS = {
    "hearing": 0.5,
    "language": 0.9,
    "introspection": 0.5,
    "cognition": 0.8,
    "mood": 0.5,
    "memory": 0.7,
    "aversion": 0.5,
    "coordination": 0.8,
    "interoception": 0.5,
    "sleep": 0.6,
    "reward": 0.7,
    "visual_recognition": 0.6,
    "visual_perception": 0.6,
    "spatial_reasoning": 0.7,
    "motor": 0.8,
    "somatosensory": 0.6,
}


# =============================================================================
# DGP PARAMETER DATACLASS
# =============================================================================

@dataclass
class DGPParameters:
    """
    Parameters for one Data Generating Process θ ~ π(θ).
    
    Each DGP represents one "world" from which we generate patients.
    Called ~162,050 times during CausalPFN training.
    """
    # Anatomical template
    patient_idx: int = 0
    z_lesion: np.ndarray = field(default_factory=lambda: np.zeros(50))
    z_disco: np.ndarray = field(default_factory=lambda: np.zeros(50))
    X_template: np.ndarray = field(default_factory=lambda: np.zeros(100))
    
    # Functional disruption (16 networks)
    D_networks: Dict[str, int] = field(default_factory=dict)
    D_baseline: float = 0.0
    
    # Treatment susceptibility
    S_networks: Dict[str, int] = field(default_factory=dict)
    S_treatment: float = 0.5
    
    # Confounding parameters
    gamma_confound: float = 0.5
    confound_type: str = "severity"
    
    # Treatment effect size
    beta_treatment: float = 0.25
    
    # Spontaneous recovery rate
    alpha_recovery: float = 0.20
    
    # Noise level
    sigma_noise: float = 0.10
    
    # Spatial features (for location-driven confounding)
    cortical_score: float = 0.5
    centroid: Tuple[float, float, float] = (45.5, 54.5, 45.5)


# =============================================================================
# ATLAS AND DATA LOADING
# =============================================================================

def load_functional_parcellation(parcellation_path: str) -> Dict[str, np.ndarray]:
    """
    Load 16-network functional parcellation.
    
    Parameters
    ----------
    parcellation_path : str
        Path to functional_parcellation_2mm.nii
    
    Returns
    -------
    networks : Dict[str, np.ndarray]
        Dictionary mapping network name to binary mask
    """
    networks = {}
    
    if not os.path.exists(parcellation_path):
        print(f"Warning: Parcellation not found: {parcellation_path}")
        return networks
    
    parc = nib.load(parcellation_path).get_fdata()
    
    for i, name in enumerate(NEUROQUERY_16_NETWORKS):
        networks[name] = (parc == i + 1).astype(float)
    
    return networks


def load_roi_pairs(roipath: str) -> Dict[int, np.ndarray]:
    """
    Load ROI pairs for treatment susceptibility.
    
    Each ROI pair divides a functional network into two subregions
    based on gene expression (Allen Atlas) or receptor density (Hansen).
    
    Parameters
    ----------
    roipath : str
        Path to ROI pair directory (e.g., atlases/genetics/)
    
    Returns
    -------
    roi_pairs : Dict[int, np.ndarray]
        Dictionary mapping ROI index (1-16) to 3D array
        where value 1 = subregion A, value 2 = subregion B
    """
    roi_pairs = {}
    
    if not os.path.exists(roipath):
        return roi_pairs
    
    files = os.listdir(roipath)
    
    for idx in range(1, 17):
        prefix = f"{idx}_"
        for f in files:
            if f.startswith(prefix) and ".nii" in f:
                roi_pairs[idx] = nib.load(os.path.join(roipath, f)).get_fdata()
                break
    
    return roi_pairs


def get_nifti_paths(directory: str) -> List[str]:
    """Get all NIfTI file paths from a directory."""
    if not os.path.exists(directory):
        return []
    
    paths = []
    for f in sorted(os.listdir(directory)):
        if f.endswith(".nii") or f.endswith(".nii.gz"):
            paths.append(os.path.join(directory, f))
    return paths


# =============================================================================
# ANATOMICAL FEATURE COMPUTATION
# =============================================================================

def compute_overlap(
    lesion_mask: np.ndarray,
    roi_mask: np.ndarray,
    threshold: float = 0.05,
    disco_thresh: float = 0.5,
) -> Tuple[int, float]:
    """
    Compute overlap between lesion/disconnectome and ROI mask.
    
    Core logic from Giles' deficit_modelling.py.
    
    Parameters
    ----------
    lesion_mask : np.ndarray
        Binary lesion mask or probabilistic disconnectome
    roi_mask : np.ndarray
        Binary ROI mask
    threshold : float
        Fraction of ROI required for deficit (default: 0.05 = 5%)
    disco_thresh : float
        Threshold for binarizing disconnectomes
    
    Returns
    -------
    deficit : int
        1 if overlap >= threshold, 0 otherwise
    overlap_fraction : float
        Actual overlap fraction
    """
    roi_volume = roi_mask.sum()
    
    if roi_volume == 0:
        return 0, 0.0
    
    # Binarize if probabilistic
    if lesion_mask.max() > 1:
        lesion_binary = (lesion_mask > disco_thresh).astype(float)
    else:
        lesion_binary = lesion_mask
    
    overlap_voxels = (lesion_binary * roi_mask).sum()
    overlap_fraction = overlap_voxels / roi_volume
    
    deficit = 1 if overlap_fraction >= threshold else 0
    
    return deficit, overlap_fraction


def compute_cortical_score(lesion_mask: np.ndarray) -> float:
    """
    Compute fraction of lesion in cortical regions.
    
    Uses z-coordinate as proxy (cortical = superior).
    Used for location-driven confounding.
    """
    nonzero = np.where(lesion_mask > 0.5)
    
    if len(nonzero[0]) == 0:
        return 0.5
    
    mean_z = nonzero[2].mean()
    max_z = lesion_mask.shape[2]
    
    return np.clip(mean_z / max_z, 0, 1)


def compute_centroid(lesion_mask: np.ndarray) -> Tuple[float, float, float]:
    """Compute lesion centroid in voxel coordinates."""
    nonzero = np.where(lesion_mask > 0.5)
    
    if len(nonzero[0]) == 0:
        shape = lesion_mask.shape
        return shape[0] / 2, shape[1] / 2, shape[2] / 2
    
    return (nonzero[0].mean(), nonzero[1].mean(), nonzero[2].mean())


# =============================================================================
# CORE DGP FUNCTIONS (maintaining spec names)
# =============================================================================

class NeuroPriorSampler:
    """
    Neuro-Prior DGP Sampler for CausalPFN Training.
    
    Implements continuous π(θ) sampling as specified in project docs.
    """
    
    def __init__(
        self,
        lesion_paths: List[str],
        disco_paths: List[str],
        functional_networks: Dict[str, np.ndarray],
        roi_pairs: Dict[int, np.ndarray],
        vae_lesion: Optional[VAE3D] = None,
        vae_disco: Optional[VAE3D] = None,
        latent_dim: int = 50,
        device: str = "cpu",
    ):
        """
        Initialize sampler.
        
        Parameters
        ----------
        lesion_paths : List[str]
            Paths to lesion NIfTI files (N=4119 in full dataset)
        disco_paths : List[str]
            Paths to disconnectome NIfTI files
        functional_networks : Dict[str, np.ndarray]
            16 network masks from functional parcellation
        roi_pairs : Dict[int, np.ndarray]
            ROI pairs for susceptibility computation
        vae_lesion : VAE3D, optional
            Pre-trained VAE for lesion encoding
        vae_disco : VAE3D, optional
            Pre-trained VAE for disconnectome encoding
        latent_dim : int
            Latent dimension (default: 50)
        device : str
            Device for VAE inference
        """
        self.lesion_paths = lesion_paths
        self.disco_paths = disco_paths
        self.functional_networks = functional_networks
        self.roi_pairs = roi_pairs
        self.vae_lesion = vae_lesion
        self.vae_disco = vae_disco
        self.latent_dim = latent_dim
        self.device = device
        
        self.n_patients = len(lesion_paths)
        
        # Precompute severity weights array
        self.severity_weights_array = np.array([
            SEVERITY_WEIGHTS.get(net, 0.5) for net in NEUROQUERY_16_NETWORKS
        ])
        self.severity_weights_array /= self.severity_weights_array.sum()
    
    def _load_nifti(self, path: str) -> np.ndarray:
        """Load NIfTI file."""
        return nib.load(path).get_fdata()
    
    def _encode_with_vae(self, vae: Optional[VAE3D], img: np.ndarray) -> np.ndarray:
        """
        Encode image using VAE, returning μ.
        
        If VAE is None, uses random projection as fallback.
        """
        if vae is None:
            # Fallback: random projection
            np.random.seed(42)
            flat = img.flatten()
            proj = np.random.randn(len(flat), self.latent_dim) / np.sqrt(len(flat))
            return (flat @ proj).astype(np.float32)
        
        vae.eval()
        with torch.no_grad():
            x = torch.from_numpy(img).float().unsqueeze(0).unsqueeze(0).to(self.device)
            mu = vae.get_latent(x)
            return mu.cpu().numpy().flatten()
    
    def sample_DGP_from_neuro_prior(self, stage: int = 1) -> DGPParameters:
        """
        Sample one data-generating process θ ~ π(θ).
        
        Called ~162,050 times during CausalPFN training.
        
        Parameters
        ----------
        stage : int
            Training stage (1, 2, or 3) - affects parameter distributions
        
        Returns
        -------
        θ : DGPParameters
            Complete specification of one DGP
        """
        θ = DGPParameters()
        
        # ═══════════════════════════════════════════════════════════
        # 1. ANATOMICAL TEMPLATE (Bootstrap from real data)
        # ═══════════════════════════════════════════════════════════
        
        θ.patient_idx = np.random.randint(0, self.n_patients)
        
        lesion_voxels = self._load_nifti(self.lesion_paths[θ.patient_idx])
        disco_voxels = self._load_nifti(self.disco_paths[θ.patient_idx])
        
        # Encode to latent space using VAE μ
        θ.z_lesion = self._encode_with_vae(self.vae_lesion, lesion_voxels)
        θ.z_disco = self._encode_with_vae(self.vae_disco, disco_voxels)
        θ.X_template = np.concatenate([θ.z_lesion, θ.z_disco])
        
        # Spatial features for confounding
        θ.cortical_score = compute_cortical_score(lesion_voxels)
        θ.centroid = compute_centroid(lesion_voxels)
        
        # ═══════════════════════════════════════════════════════════
        # 2. FUNCTIONAL DISRUPTION (16 networks from NeuroQuery)
        # ═══════════════════════════════════════════════════════════
        
        θ.D_networks = {}
        disruption_values = []
        
        for network in NEUROQUERY_16_NETWORKS:
            if network in self.functional_networks:
                deficit, _ = compute_overlap(disco_voxels, self.functional_networks[network])
                θ.D_networks[network] = deficit
                disruption_values.append(deficit)
            else:
                θ.D_networks[network] = 0
                disruption_values.append(0)
        
        θ.D_baseline = np.average(disruption_values, weights=self.severity_weights_array)
        
        # ═══════════════════════════════════════════════════════════
        # 3. TREATMENT SUSCEPTIBILITY (Gene expression from Allen Atlas)
        # ═══════════════════════════════════════════════════════════
        
        θ.S_networks = {}
        susceptible_count = 0
        damaged_count = 0
        
        for i, (network, deficit) in enumerate(θ.D_networks.items()):
            if deficit == 1:
                damaged_count += 1
                roi_idx = i + 1
                
                if roi_idx in self.roi_pairs:
                    roi_pair = self.roi_pairs[roi_idx]
                    subnetwork_A = (roi_pair == 1).astype(float)
                    subnetwork_B = (roi_pair == 2).astype(float)
                    
                    _, overlap_A = compute_overlap(disco_voxels, subnetwork_A)
                    _, overlap_B = compute_overlap(disco_voxels, subnetwork_B)
                    
                    # Susceptibility: 1 if more in high-responder region
                    θ.S_networks[network] = 1 if overlap_A >= overlap_B else 0
                else:
                    # Random fallback
                    θ.S_networks[network] = np.random.randint(0, 2)
                
                susceptible_count += θ.S_networks[network]
        
        θ.S_treatment = susceptible_count / damaged_count if damaged_count > 0 else 0.5
        
        # ═══════════════════════════════════════════════════════════
        # 4. CONFOUNDING STRENGTH (Continuous γ ∈ [0, 1])
        # ═══════════════════════════════════════════════════════════
        
        if stage == 1:
            # Stage 1: Mild confounding only
            θ.gamma_confound = np.random.uniform(0, 0.3)
        elif stage == 3:
            # Stage 3: Strong confounding only
            θ.gamma_confound = np.random.uniform(0.5, 1.0)
        else:
            # Stage 2: Full distribution
            θ.gamma_confound = np.random.beta(a=2, b=2)
        
        θ.confound_type = np.random.choice(["severity", "location", "network", "mixed"])
        
        # ═══════════════════════════════════════════════════════════
        # 5. TREATMENT EFFECT SIZE (Continuous β)
        # ═══════════════════════════════════════════════════════════
        
        if stage == 1:
            # Stage 1: Moderate effects
            θ.beta_treatment = np.random.uniform(0.2, 0.4)
        elif stage == 3:
            # Stage 3: Small effects (challenging)
            θ.beta_treatment = np.random.uniform(0.05, 0.15)
        else:
            # Stage 2: Full mixture distribution
            effect_regime = np.random.choice(["small", "medium", "large"], p=[0.3, 0.5, 0.2])
            if effect_regime == "small":
                θ.beta_treatment = np.random.uniform(0.05, 0.15)
            elif effect_regime == "medium":
                θ.beta_treatment = np.random.uniform(0.15, 0.35)
            else:
                θ.beta_treatment = np.random.uniform(0.35, 0.55)
        
        # ═══════════════════════════════════════════════════════════
        # 6. SPONTANEOUS RECOVERY RATE (Continuous α)
        # ═══════════════════════════════════════════════════════════
        
        if stage == 1:
            # Stage 1: Occasional recovery
            θ.alpha_recovery = np.random.uniform(0.15, 0.30)
        elif stage == 3:
            # Stage 3: High recovery (challenging)
            θ.alpha_recovery = np.random.uniform(0.30, 0.45)
        else:
            # Stage 2: Full mixture distribution
            recovery_regime = np.random.choice(["rare", "occasional", "frequent"], p=[0.2, 0.5, 0.3])
            if recovery_regime == "rare":
                θ.alpha_recovery = np.random.uniform(0.05, 0.15)
            elif recovery_regime == "occasional":
                θ.alpha_recovery = np.random.uniform(0.15, 0.30)
            else:
                θ.alpha_recovery = np.random.uniform(0.30, 0.45)
        
        # ═══════════════════════════════════════════════════════════
        # 7. NOISE LEVEL (Aleatoric uncertainty)
        # ═══════════════════════════════════════════════════════════
        
        θ.sigma_noise = np.random.uniform(0.05, 0.15)
        
        return θ
    
    def generate_treatment_propensity(
        self,
        X_anatomy: np.ndarray,
        θ: DGPParameters,
    ) -> Tuple[int, float]:
        """
        Generate treatment assignment probability π(W=1|X).
        
        Satisfies positivity: 0.01 < π < 0.99
        Satisfies unconfoundedness: depends only on observed X
        
        Parameters
        ----------
        X_anatomy : np.ndarray
            Patient anatomy representation (100-dim)
        θ : DGPParameters
            Current DGP parameters
        
        Returns
        -------
        W : int
            Treatment assignment (0 or 1)
        π : float
            Propensity score P(W=1|X)
        """
        γ = θ.gamma_confound
        
        if θ.confound_type == "severity":
            # Treat severe patients more aggressively
            logit_π = γ * (2 * θ.D_baseline - 1)
            
        elif θ.confound_type == "location":
            # Prefer treating cortical over subcortical
            logit_π = γ * (2 * θ.cortical_score - 1)
            
        elif θ.confound_type == "network":
            # Prioritize motor/language patients
            critical_network = (
                θ.D_networks.get("motor", 0) or 
                θ.D_networks.get("language", 0)
            )
            logit_π = γ * (2 * critical_network - 1)
            
        else:  # mixed
            critical_network = (
                θ.D_networks.get("motor", 0) or 
                θ.D_networks.get("language", 0)
            )
            logit_π = γ * (
                0.4 * (2 * θ.D_baseline - 1) +
                0.3 * (2 * θ.cortical_score - 1) +
                0.3 * (2 * critical_network - 1)
            )
        
        # Add noise to ensure positivity
        logit_π += np.random.normal(0, 0.3)
        
        # Convert to probability
        π = 1 / (1 + np.exp(-logit_π))
        π = np.clip(π, 0.01, 0.99)
        
        # Treatment assignment
        W = np.random.binomial(1, π)
        
        return W, π
    
    def generate_outcomes(
        self,
        X_anatomy: np.ndarray,
        θ: DGPParameters,
    ) -> Dict[str, float]:
        """
        Generate both counterfactual outcomes Y_0 and Y_1.
        
        This is the GROUND TRUTH that enables causal learning.
        
        KEY DESIGN CHOICE: Train on τ (expectation), not Y (noisy observation)
        - Removes aleatoric noise from target
        - Model learns causal effect, not noise prediction
        - Matches CausalPFN's CEPO framework (Balazadeh et al., 2025)
        
        Parameters
        ----------
        X_anatomy : np.ndarray
            Patient anatomy representation
        θ : DGPParameters
            Current DGP parameters
        
        Returns
        -------
        outcomes : Dict[str, float]
            Y_0, Y_1 (noisy), τ_0, τ_1 (ground truth), CATE
        """
        D = θ.D_baseline
        S = θ.S_treatment
        α = θ.alpha_recovery
        β = θ.beta_treatment
        σ = θ.sigma_noise
        
        # Ground truth expectations (TRAINING TARGETS)
        τ_0 = (1 - D) + α * D
        τ_1 = τ_0 + β * S * D
        
        # Noisy observations (for simulation, not training targets)
        Y_0 = τ_0 + np.random.normal(0, σ)
        Y_0 = np.clip(Y_0, 0, 1)
        
        Y_1 = τ_1 + np.random.normal(0, σ)
        Y_1 = np.clip(Y_1, 0, 1)
        
        return {
            "Y_0": Y_0,
            "Y_1": Y_1,
            "τ_0": τ_0,
            "τ_1": τ_1,
            "CATE": τ_1 - τ_0,
        }
    
    def generate_patient_batch(
        self,
        θ: DGPParameters,
        n_patients: int,
    ) -> List[Dict[str, Any]]:
        """
        Generate a batch of patients from a single DGP.
        
        Each patient has the same θ but different (W, Y).
        
        Parameters
        ----------
        θ : DGPParameters
            DGP specification
        n_patients : int
            Number of patients to generate
        
        Returns
        -------
        batch : List[Dict]
            List of patient records
        """
        records = []
        
        for _ in range(n_patients):
            W, π = self.generate_treatment_propensity(θ.X_template, θ)
            outcomes = self.generate_outcomes(θ.X_template, θ)
            
            # Observed outcome depends on treatment
            Y = outcomes["Y_1"] if W == 1 else outcomes["Y_0"]
            
            record = {
                "X": θ.X_template.tolist(),
                "W": W,
                "Y": Y,
                "τ_0": outcomes["τ_0"],
                "τ_1": outcomes["τ_1"],
                "CATE": outcomes["CATE"],
                "π": π,
                "D_baseline": θ.D_baseline,
                "S_treatment": θ.S_treatment,
                "γ": θ.gamma_confound,
                "β": θ.beta_treatment,
                "α": θ.alpha_recovery,
                "σ": θ.sigma_noise,
                "confound_type": θ.confound_type,
            }
            records.append(record)
        
        return records
    
    def generate_training_data(
        self,
        n_dgps: int,
        n_patients_per_dgp: int,
        stage: int = 1,
        save_path: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Generate complete training data for CausalPFN.
        
        Parameters
        ----------
        n_dgps : int
            Number of DGPs to sample
        n_patients_per_dgp : int
            Patients per DGP
        stage : int
            Training stage (1, 2, or 3)
        save_path : str, optional
            Path to save data
        
        Returns
        -------
        df : pd.DataFrame
            Training dataset
        """
        all_records = []
        
        print(f"\n{'='*60}")
        print(f"Generating Training Data - Stage {stage}")
        print(f"  DGPs: {n_dgps:,}")
        print(f"  Patients/DGP: {n_patients_per_dgp:,}")
        print(f"  Total triplets: {n_dgps * n_patients_per_dgp:,}")
        print(f"{'='*60}\n")
        
        for dgp_idx in trange(n_dgps, desc="Sampling DGPs"):
            θ = self.sample_DGP_from_neuro_prior(stage=stage)
            batch = self.generate_patient_batch(θ, n_patients_per_dgp)
            
            for record in batch:
                record["dgp_idx"] = dgp_idx
            
            all_records.extend(batch)
        
        df = pd.DataFrame(all_records)
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            df.to_parquet(save_path, index=False)
            print(f"\nSaved: {save_path}")
            print(f"  Samples: {len(df):,}")
            print(f"  Size: {os.path.getsize(save_path) / 1e6:.1f} MB")
        
        return df


# =============================================================================
# MAIN
# =============================================================================

def run(args):
    """Main run function."""
    print(f"\n{'#'*60}")
    print("# NEURO-PRIOR DGP SAMPLING")
    print(f"{'#'*60}")
    print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    # Load file paths
    lesion_paths = get_nifti_paths(args.lesionpath)
    disco_paths = get_nifti_paths(args.discopath)
    
    if args.n_samples:
        lesion_paths = lesion_paths[:args.n_samples]
        disco_paths = disco_paths[:args.n_samples]
    
    print(f"Lesions: {len(lesion_paths)}")
    print(f"Disconnectomes: {len(disco_paths)}")
    
    # Load functional networks
    parc_path = os.path.join(args.atlaspath, "functional_parcellation_2mm.nii")
    functional_networks = load_functional_parcellation(parc_path)
    print(f"Networks: {len(functional_networks)}")
    
    # Load ROI pairs
    roi_pairs = {}
    genetics_path = os.path.join(args.atlaspath, "genetics")
    if os.path.exists(genetics_path):
        roi_pairs = load_roi_pairs(genetics_path)
    print(f"ROI pairs: {len(roi_pairs)}")
    
    # Load VAEs
    vae_lesion = None
    vae_disco = None
    
    if args.vae_lesion_path and os.path.exists(args.vae_lesion_path):
        vae_lesion = VAE3D(latent_dim=args.latent_dim)
        vae_lesion.load_state_dict(torch.load(args.vae_lesion_path, map_location=device))
        vae_lesion.to(device).eval()
        print(f"Loaded VAE lesion: {args.vae_lesion_path}")
    
    if args.vae_disco_path and os.path.exists(args.vae_disco_path):
        vae_disco = VAE3D(latent_dim=args.latent_dim)
        vae_disco.load_state_dict(torch.load(args.vae_disco_path, map_location=device))
        vae_disco.to(device).eval()
        print(f"Loaded VAE disco: {args.vae_disco_path}")
    
    # Initialize sampler
    sampler = NeuroPriorSampler(
        lesion_paths=lesion_paths,
        disco_paths=disco_paths,
        functional_networks=functional_networks,
        roi_pairs=roi_pairs,
        vae_lesion=vae_lesion,
        vae_disco=vae_disco,
        latent_dim=args.latent_dim,
        device=device,
    )
    
    # Generate data
    save_path = os.path.join(
        args.savepath,
        f"neuro_prior_stage{args.stage}_dgps{args.n_dgps}_pts{args.n_patients_per_dgp}.parquet"
    )
    
    df = sampler.generate_training_data(
        n_dgps=args.n_dgps,
        n_patients_per_dgp=args.n_patients_per_dgp,
        stage=args.stage,
        save_path=save_path,
    )
    
    # Summary
    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    print(f"Samples: {len(df):,}")
    print(f"Y mean: {df['Y'].mean():.3f}")
    print(f"Treatment rate: {df['W'].mean():.3f}")
    print(f"CATE mean: {df['CATE'].mean():.3f}")
    print(f"γ mean: {df['γ'].mean():.3f}")
    print(f"β mean: {df['β'].mean():.3f}")
    print(f"α mean: {df['α'].mean():.3f}")
    
    print(f"\n{'#'*60}")
    print("# COMPLETE")
    print(f"{'#'*60}")
    
    return df


def command_line_options():
    parser = argparse.ArgumentParser(description="Neuro-Prior DGP Sampling")
    
    parser.add_argument("--lesionpath", type=str, required=True)
    parser.add_argument("--discopath", type=str, required=True)
    parser.add_argument("--atlaspath", type=str, default="atlases")
    parser.add_argument("--savepath", type=str, default="training_data")
    
    parser.add_argument("--vae_lesion_path", type=str, default=None)
    parser.add_argument("--vae_disco_path", type=str, default=None)
    
    parser.add_argument("--n_dgps", type=int, default=1000)
    parser.add_argument("--n_patients_per_dgp", type=int, default=1024)
    parser.add_argument("--stage", type=int, default=1, choices=[1, 2, 3])
    parser.add_argument("--latent_dim", type=int, default=50)
    
    parser.add_argument("--n_samples", type=int, default=None, help="DEBUG: limit patients")
    parser.add_argument("--verbose", action="store_true")
    
    return parser.parse_args()


if __name__ == "__main__":
    args = command_line_options()
    run(args)
