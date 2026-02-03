"""
representation_neuro_prior.py - VAE Training for Neuro-Prior Foundation Model

Adapted from Giles et al. (2025) representation.py for Neuro-Prior project.

Usage:
    python representation_neuro_prior.py \
        --lesionpath /path/to/lesions \
        --discopath /path/to/disconnectomes \
        --savepath /path/to/output \
        --latent_dim 50 \
        --batch_size 10 \
        --max_epoch 32

Output:
    - vae_lesion.pth: Trained VAE encoder for lesion masks
    - vae_disco.pth: Trained VAE encoder for disconnectomes
    - training_log.json: Training metrics and hyperparameters
"""

import argparse
from datetime import datetime
import copy
import os

from nilearn import image
import nibabel as nib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm, trange


# =============================================================================
# MODEL ARCHITECTURE (from Giles et al., 2025)
# =============================================================================


class ResBlock3D(nn.Module):
    """3D ResNet-type convolutional block"""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, downsample: nn.Module = None):
        super(ResBlock3D, self).__init__()
        self.conv1 = nn.Conv3d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class Encoder3D(nn.Module):
    """
    Encoder with 4 layers of 3D ResNet-type convolutional blocks.
    
    Architecture (from Giles et al., 2025 paper, page 12):
    "The AE's encoder was composed of four layers of three-dimensional 
    ResNet-type convolutional blocks, followed by fully connected layers."
    """

    def __init__(self, latent_dim: int, base_channels: int = 32):
        super(Encoder3D, self).__init__()
        self.latent_dim = latent_dim

        # Initial convolution
        self.conv_init = nn.Sequential(
            nn.Conv3d(1, base_channels, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm3d(base_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=2, padding=1),
        )

        # 4 ResNet layers
        self.layer1 = self._make_layer(base_channels, base_channels, blocks=2, stride=1)
        self.layer2 = self._make_layer(base_channels, base_channels * 2, blocks=2, stride=2)
        self.layer3 = self._make_layer(base_channels * 2, base_channels * 4, blocks=2, stride=2)
        self.layer4 = self._make_layer(base_channels * 4, base_channels * 8, blocks=2, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc_input_dim = base_channels * 8

    def _make_layer(self, in_channels: int, out_channels: int, blocks: int, stride: int) -> nn.Sequential:
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(out_channels),
            )
        layers = [ResBlock3D(in_channels, out_channels, stride, downsample)]
        for _ in range(1, blocks):
            layers.append(ResBlock3D(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_init(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x


class Decoder3D(nn.Module):
    """Decoder with transposed convolutions (inverse of encoder)"""

    def __init__(self, latent_dim: int, base_channels: int = 32, output_shape: Tuple[int, int, int] = (91, 109, 91)):
        super(Decoder3D, self).__init__()
        self.latent_dim = latent_dim
        self.base_channels = base_channels
        self.output_shape = output_shape
        self.init_shape = (3, 4, 3)
        self.fc_output_dim = base_channels * 8 * np.prod(self.init_shape)

        self.fc = nn.Linear(latent_dim, self.fc_output_dim)

        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(base_channels * 8, base_channels * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(base_channels * 4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(base_channels * 4, base_channels * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(base_channels * 2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(base_channels * 2, base_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(base_channels),
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(base_channels, base_channels // 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(base_channels // 2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(base_channels // 2, 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.fc(z)
        x = x.view(-1, self.base_channels * 8, *self.init_shape)
        x = self.decoder(x)
        if x.shape[2:] != self.output_shape:
            x = F.interpolate(x, size=self.output_shape, mode="trilinear", align_corners=False)
        return x


class VAE3D(nn.Module):
    """
    3D Variational Autoencoder for lesion/disconnectome encoding.
    
    Architecture matches Giles et al. (2025):
    - Encoder: 4 layers of ResNet-3D blocks → μ, σ
    - Reparameterization trick for sampling
    - Decoder: inverse architecture
    - Loss: BCE reconstruction + β * KL divergence
    """

    def __init__(
        self,
        latent_dim: int = 50,
        base_channels: int = 32,
        input_shape: Tuple[int, int, int] = (91, 109, 91),
        beta: float = 1.0,
    ):
        super(VAE3D, self).__init__()
        self.latent_dim = latent_dim
        self.input_shape = input_shape
        self.beta = beta

        self.encoder = Encoder3D(latent_dim, base_channels)
        self.fc_mu = nn.Linear(self.encoder.fc_input_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.encoder.fc_input_dim, latent_dim)
        self.decoder = Decoder3D(latent_dim, base_channels, input_shape)

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode input to latent distribution parameters (μ, log σ²)"""
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick: z = μ + ε * σ, where ε ~ N(0,1)"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent vector to reconstruction"""
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass: encode → sample → decode"""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar, z

    def loss_function(self, recon_x: torch.Tensor, x: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        VAE loss: reconstruction + β * KL divergence
        
        Returns: (total_loss, recon_loss, kl_loss)
        """
        recon_loss = F.binary_cross_entropy(recon_x, x, reduction="sum")
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        total_loss = recon_loss + self.beta * kl_loss
        return total_loss, recon_loss, kl_loss

    def get_latent(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get μ from encoder (used for downstream tasks).
        
        KEY DESIGN CHOICE: Use μ (not sampled z) for deterministic representations.
        """
        mu, _ = self.encode(x)
        return mu


# =============================================================================
# DATASET
# =============================================================================

class NiftiDataset(Dataset):
    """Dataset for loading 3D NIfTI files"""

    def __init__(self, file_paths: List[str], normalize: bool = True):
        self.file_paths = file_paths
        self.normalize = normalize

    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        img = nib.load(self.file_paths[idx]).get_fdata()
        
        # Normalize to [0, 1]
        if self.normalize:
            img = img.astype(np.float32)
            if img.max() > 0:
                img = img / img.max()
        
        # Add channel dimension: (H, W, D) → (1, H, W, D)
        img = torch.from_numpy(img).float().unsqueeze(0)
        return img


def str2bool(v):
    """Convert string to boolean for argparse"""
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")



# =============================================================================
# DATASET
# =============================================================================

class NiftiDataset(Dataset):
    """Dataset for loading 3D NIfTI files"""

    def __init__(self, file_paths: List[str], normalize: bool = True):
        self.file_paths = file_paths
        self.normalize = normalize

    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        img = nib.load(self.file_paths[idx]).get_fdata()
        
        # Normalize to [0, 1]
        if self.normalize:
            img = img.astype(np.float32)
            if img.max() > 0:
                img = img / img.max()
        
        # Add channel dimension: (H, W, D) → (1, H, W, D)
        img = torch.from_numpy(img).float().unsqueeze(0)
        return img


# =============================================================================
# TRAINING FUNCTIONS
# =============================================================================

def train_epoch(
    model: VAE3D,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
    epoch: int,
) -> Dict[str, float]:
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    total_recon = 0.0
    total_kl = 0.0
    n_samples = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}", leave=False)
    for batch in pbar:
        batch = batch.to(device)
        optimizer.zero_grad()

        recon, mu, logvar, z = model(batch)
        loss, recon_loss, kl_loss = model.loss_function(recon, batch, mu, logvar)

        loss.backward()
        optimizer.step()

        batch_size = batch.size(0)
        total_loss += loss.item()
        total_recon += recon_loss.item()
        total_kl += kl_loss.item()
        n_samples += batch_size

        pbar.set_postfix({
            "loss": f"{loss.item() / batch_size:.4f}",
            "recon": f"{recon_loss.item() / batch_size:.4f}",
            "kl": f"{kl_loss.item() / batch_size:.4f}",
        })

    return {
        "loss": total_loss / n_samples,
        "recon_loss": total_recon / n_samples,
        "kl_loss": total_kl / n_samples,
    }


def validate(model: VAE3D, dataloader: DataLoader, device: str) -> Dict[str, float]:
    """Validate model"""
    model.eval()
    total_loss = 0.0
    total_recon = 0.0
    total_kl = 0.0
    n_samples = 0

    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            recon, mu, logvar, z = model(batch)
            loss, recon_loss, kl_loss = model.loss_function(recon, batch, mu, logvar)

            batch_size = batch.size(0)
            total_loss += loss.item()
            total_recon += recon_loss.item()
            total_kl += kl_loss.item()
            n_samples += batch_size

    return {
        "loss": total_loss / n_samples,
        "recon_loss": total_recon / n_samples,
        "kl_loss": total_kl / n_samples,
    }


def train_vae(
    file_paths: List[str],
    savepath: str,
    model_name: str,
    latent_dim: int = 50,
    batch_size: int = 10,
    min_epoch: int = 16,
    max_epoch: int = 32,
    early_stopping_epochs: int = 4,
    learning_rate: float = 1e-3,
    beta: float = 1.0,
    val_split: float = 0.1,
    device: str = "cuda",
    verbose: bool = True,
) -> Tuple[VAE3D, Dict]:
    """
    Train a VAE on the given data.
    
    Parameters
    ----------
    file_paths : List[str]
        Paths to NIfTI files
    savepath : str
        Directory to save model and logs
    model_name : str
        Name for saved model (e.g., "vae_lesion" or "vae_disco")
    latent_dim : int
        Latent space dimension (default: 50)
    batch_size : int
        Training batch size (default: 10, from Giles)
    min_epoch : int
        Minimum training epochs (default: 16, from Giles)
    max_epoch : int
        Maximum training epochs (default: 32, from Giles)
    early_stopping_epochs : int
        Early stopping patience (default: 4, from Giles)
    learning_rate : float
        Learning rate
    beta : float
        KL divergence weight
    val_split : float
        Validation split fraction
    device : str
        Device for training
    verbose : bool
        Print training progress
    
    Returns
    -------
    model : VAE3D
        Trained model
    history : Dict
        Training history
    """
    os.makedirs(savepath, exist_ok=True)

    # Split data
    n_val = int(len(file_paths) * val_split)
    n_train = len(file_paths) - n_val
    
    indices = np.random.permutation(len(file_paths))
    train_paths = [file_paths[i] for i in indices[:n_train]]
    val_paths = [file_paths[i] for i in indices[n_train:]]

    if verbose:
        print(f"\n{'='*60}")
        print(f"Training {model_name}")
        print(f"{'='*60}")
        print(f"  Total samples: {len(file_paths)}")
        print(f"  Train samples: {len(train_paths)}")
        print(f"  Val samples: {len(val_paths)}")
        print(f"  Latent dim: {latent_dim}")
        print(f"  Batch size: {batch_size}")
        print(f"  Device: {device}")

    # Create datasets and dataloaders
    train_dataset = NiftiDataset(train_paths)
    val_dataset = NiftiDataset(val_paths)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # Get input shape from first sample
    sample = train_dataset[0]
    input_shape = tuple(sample.shape[1:])
    if verbose:
        print(f"  Input shape: {input_shape}")

    # Initialize model
    model = VAE3D(latent_dim=latent_dim, input_shape=input_shape, beta=beta)
    model = model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    history = {"train_loss": [], "val_loss": [], "train_recon": [], "val_recon": [], "train_kl": [], "val_kl": []}
    best_val_loss = float("inf")
    epochs_without_improvement = 0

    for epoch in range(max_epoch):
        # Train
        train_metrics = train_epoch(model, train_loader, optimizer, device, epoch)
        
        # Validate
        val_metrics = validate(model, val_loader, device)

        # Record history
        history["train_loss"].append(train_metrics["loss"])
        history["val_loss"].append(val_metrics["loss"])
        history["train_recon"].append(train_metrics["recon_loss"])
        history["val_recon"].append(val_metrics["recon_loss"])
        history["train_kl"].append(train_metrics["kl_loss"])
        history["val_kl"].append(val_metrics["kl_loss"])

        if verbose:
            print(f"Epoch {epoch+1:2d}/{max_epoch}: "
                  f"train_loss={train_metrics['loss']:.4f}, "
                  f"val_loss={val_metrics['loss']:.4f}")

        # Early stopping check (only after min_epoch)
        if epoch >= min_epoch - 1:
            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                epochs_without_improvement = 0
                # Save best model
                torch.save(model.state_dict(), os.path.join(savepath, f"{model_name}.pth"))
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= early_stopping_epochs:
                    if verbose:
                        print(f"Early stopping at epoch {epoch+1}")
                    break
        else:
            # Before min_epoch, always save if best
            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                torch.save(model.state_dict(), os.path.join(savepath, f"{model_name}.pth"))

    # Load best model
    model.load_state_dict(torch.load(os.path.join(savepath, f"{model_name}.pth")))

    # Save history
    history["best_val_loss"] = best_val_loss
    history["final_epoch"] = epoch + 1
    
    with open(os.path.join(savepath, f"{model_name}_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    if verbose:
        print(f"  Best val loss: {best_val_loss:.4f}")
        print(f"  Model saved: {os.path.join(savepath, f'{model_name}.pth')}")

    return model, history


# =============================================================================
# MAIN FUNCTION
# =============================================================================

def get_nifti_paths(directory: str) -> List[str]:
    """Get all NIfTI file paths from a directory"""
    if not os.path.exists(directory):
        return []
    
    paths = []
    for f in sorted(os.listdir(directory)):
        if f.endswith(".nii") or f.endswith(".nii.gz"):
            paths.append(os.path.join(directory, f))
    return paths


def run(args):
    """Main run function"""
    print(f"\n{'#'*60}")
    print("# NEURO-PRIOR VAE TRAINING")
    print(f"{'#'*60}")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    os.makedirs(args.savepath, exist_ok=True)

    # Training configuration
    config = {
        "latent_dim": args.latent_dim,
        "batch_size": args.batch_size,
        "min_epoch": args.min_epoch,
        "max_epoch": args.max_epoch,
        "early_stopping_epochs": args.early_stopping_epochs,
        "learning_rate": args.learning_rate,
        "beta": args.beta,
        "device": device,
        "timestamp": datetime.now().isoformat(),
    }

    # Save configuration
    with open(os.path.join(args.savepath, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    # Train VAE for lesions
    if args.lesionpath:
        lesion_paths = get_nifti_paths(args.lesionpath)
        if args.n_samples:
            lesion_paths = lesion_paths[:args.n_samples]
        
        if len(lesion_paths) > 0:
            print(f"\nFound {len(lesion_paths)} lesion files")
            vae_lesion, history_lesion = train_vae(
                file_paths=lesion_paths,
                savepath=args.savepath,
                model_name="vae_lesion",
                latent_dim=args.latent_dim,
                batch_size=args.batch_size,
                min_epoch=args.min_epoch,
                max_epoch=args.max_epoch,
                early_stopping_epochs=args.early_stopping_epochs,
                learning_rate=args.learning_rate,
                beta=args.beta,
                device=device,
                verbose=args.verbose,
            )
        else:
            print(f"No lesion files found in {args.lesionpath}")

    # Train VAE for disconnectomes
    if args.discopath:
        disco_paths = get_nifti_paths(args.discopath)
        if args.n_samples:
            disco_paths = disco_paths[:args.n_samples]
        
        if len(disco_paths) > 0:
            print(f"\nFound {len(disco_paths)} disconnectome files")
            vae_disco, history_disco = train_vae(
                file_paths=disco_paths,
                savepath=args.savepath,
                model_name="vae_disco",
                latent_dim=args.latent_dim,
                batch_size=args.batch_size,
                min_epoch=args.min_epoch,
                max_epoch=args.max_epoch,
                early_stopping_epochs=args.early_stopping_epochs,
                learning_rate=args.learning_rate,
                beta=args.beta,
                device=device,
                verbose=args.verbose,
            )
        else:
            print(f"No disconnectome files found in {args.discopath}")

    print(f"\n{'#'*60}")
    print("# TRAINING COMPLETE")
    print(f"{'#'*60}")
    print(f"Output directory: {args.savepath}")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


def command_line_options():
    parser = argparse.ArgumentParser(
        description="Neuro-Prior VAE Training"
    )
    
    # Data paths
    parser.add_argument("--lesionpath", type=str, default="", help="Path to lesion NIfTI files")
    parser.add_argument("--discopath", type=str, default="", help="Path to disconnectome NIfTI files")
    parser.add_argument("--savepath", type=str, default="./vae_models", help="Output directory")
    
    # Model parameters
    parser.add_argument("--latent_dim", type=int, default=50, help="Latent space dimension")
    parser.add_argument("--beta", type=float, default=1.0, help="KL divergence weight")
    
    # Training parameters (from Giles et al., 2025)
    parser.add_argument("--batch_size", type=int, default=10, help="Batch size (default: 10 from Giles)")
    parser.add_argument("--min_epoch", type=int, default=16, help="Minimum epochs (default: 16 from Giles)")
    parser.add_argument("--max_epoch", type=int, default=32, help="Maximum epochs (default: 32 from Giles)")
    parser.add_argument("--early_stopping_epochs", type=int, default=4, help="Early stopping patience (default: 4 from Giles)")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate")
    
    # Debug
    parser.add_argument("--n_samples", type=int, default=None, help="DEBUG: Limit number of samples")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")

    return parser.parse_args()


if __name__ == "__main__":
    args = command_line_options()
    run(args)
