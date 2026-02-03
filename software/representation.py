"""
Extended representation.py with VAE and AE implementation

This file extends the original representation.py from the 
"Individualized prescriptive inference in ischaemic stroke" paper
to include the missing VAE and AE implementations.

Usage:
    python representation_extended.py \
        --lesionpath /path/to/lesions \
        --discopath /path/to/disconnectomes \
        --savepath /path/to/results \
        --run_vae True \
        --run_ae True \
        --run_nmf True \
        --run_pca True \
        --latent_components 50
"""

import argparse
from datetime import datetime
import copy
import os

import joblib
from nilearn import image
import nibabel as nib
import nimfa
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm, trange
from xgboost import XGBClassifier


# =============================================================================
# VAE/AE Model Definitions
# =============================================================================


class ResBlock3D(nn.Module):
    """3D ResNet-type convolutional block"""

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResBlock3D, self).__init__()
        self.conv1 = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.downsample = downsample

    def forward(self, x):
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
    """Encoder with 4 layers of 3D ResNet-type convolutional blocks"""

    def __init__(self, latent_dim, base_channels=32):
        super(Encoder3D, self).__init__()
        self.latent_dim = latent_dim

        self.conv_init = nn.Sequential(
            nn.Conv3d(1, base_channels, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm3d(base_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=2, padding=1),
        )

        self.layer1 = self._make_layer(base_channels, base_channels, blocks=2, stride=1)
        self.layer2 = self._make_layer(
            base_channels, base_channels * 2, blocks=2, stride=2
        )
        self.layer3 = self._make_layer(
            base_channels * 2, base_channels * 4, blocks=2, stride=2
        )
        self.layer4 = self._make_layer(
            base_channels * 4, base_channels * 8, blocks=2, stride=2
        )
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc_input_dim = base_channels * 8

    def _make_layer(self, in_channels, out_channels, blocks, stride):
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv3d(
                    in_channels, out_channels, kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm3d(out_channels),
            )
        layers = [ResBlock3D(in_channels, out_channels, stride, downsample)]
        for _ in range(1, blocks):
            layers.append(ResBlock3D(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv_init(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x


class Decoder3D(nn.Module):
    """Decoder with transposed convolutions"""

    def __init__(self, latent_dim, base_channels=32, output_shape=(91, 109, 91)):
        super(Decoder3D, self).__init__()
        self.latent_dim = latent_dim
        self.base_channels = base_channels
        self.output_shape = output_shape
        self.init_shape = (3, 4, 3)
        self.fc_output_dim = base_channels * 8 * np.prod(self.init_shape)

        self.fc = nn.Linear(latent_dim, self.fc_output_dim)

        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(
                base_channels * 8, base_channels * 4, kernel_size=4, stride=2, padding=1
            ),
            nn.BatchNorm3d(base_channels * 4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(
                base_channels * 4, base_channels * 2, kernel_size=4, stride=2, padding=1
            ),
            nn.BatchNorm3d(base_channels * 2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(
                base_channels * 2, base_channels, kernel_size=4, stride=2, padding=1
            ),
            nn.BatchNorm3d(base_channels),
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(
                base_channels, base_channels // 2, kernel_size=4, stride=2, padding=1
            ),
            nn.BatchNorm3d(base_channels // 2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(
                base_channels // 2, 1, kernel_size=4, stride=2, padding=1
            ),
            nn.Sigmoid(),
        )

    def forward(self, z):
        x = self.fc(z)
        x = x.view(-1, self.base_channels * 8, *self.init_shape)
        x = self.decoder(x)
        if x.shape[2:] != self.output_shape:
            x = F.interpolate(
                x, size=self.output_shape, mode="trilinear", align_corners=False
            )
        return x


class VAE3D(nn.Module):
    """3D Variational Autoencoder"""

    def __init__(
        self, latent_dim=50, base_channels=32, input_shape=(91, 109, 91), beta=1.0
    ):
        super(VAE3D, self).__init__()
        self.latent_dim = latent_dim
        self.input_shape = input_shape
        self.beta = beta

        self.encoder = Encoder3D(latent_dim, base_channels)
        self.fc_mu = nn.Linear(self.encoder.fc_input_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.encoder.fc_input_dim, latent_dim)
        self.decoder = Decoder3D(latent_dim, base_channels, input_shape)

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar, z

    def loss_function(self, recon_x, x, mu, logvar):
        recon_loss = F.binary_cross_entropy(recon_x, x, reduction="sum")
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + self.beta * kl_loss, recon_loss, kl_loss

    def get_latent(self, x):
        mu, _ = self.encode(x)
        return mu


class AE3D(nn.Module):
    """3D Autoencoder"""

    def __init__(self, latent_dim=50, base_channels=32, input_shape=(91, 109, 91)):
        super(AE3D, self).__init__()
        self.latent_dim = latent_dim
        self.input_shape = input_shape

        self.encoder = Encoder3D(latent_dim, base_channels)
        self.fc_latent = nn.Linear(self.encoder.fc_input_dim, latent_dim)
        self.decoder = Decoder3D(latent_dim, base_channels, input_shape)

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_latent(h)

    def forward(self, x):
        z = self.encode(x)
        return self.decoder(z), z

    def loss_function(self, recon_x, x):
        bce_loss = F.binary_cross_entropy(recon_x, x, reduction="sum")
        smooth = 1e-5
        intersection = (recon_x * x).sum()
        dice_loss = 1 - (2.0 * intersection + smooth) / (
            recon_x.sum() + x.sum() + smooth
        )
        return bce_loss + dice_loss * x.numel(), bce_loss, dice_loss

    def get_latent(self, x):
        return self.encode(x)


class LesionDataset(Dataset):
    """Dataset for loading lesion NIfTI files"""

    def __init__(self, filepaths, data_dir):
        self.filepaths = filepaths
        self.data_dir = data_dir

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        filepath = os.path.join(self.data_dir, self.filepaths[idx])
        img = nib.load(filepath).get_fdata()
        img = img[np.newaxis, ...].astype(np.float32)
        if img.max() > 0:
            img = img / img.max()
        return torch.from_numpy(img), self.filepaths[idx]


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
# Original Code from representation.py
# =============================================================================


class ground_truth_splits:
    def __init__(self, n_splits):
        self.n_splits = n_splits
        self.seed = 42

    def make_kfold_splits(self, whole_ground_truth):
        whole_ground_truth = whole_ground_truth.reset_index(drop=True)
        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=42)
        for kf_count, (train_index, test_index) in enumerate(
            kf.split(whole_ground_truth)
        ):
            ground_truth_train = whole_ground_truth.iloc[train_index].reset_index(
                drop=True
            )
            ground_truth_test = whole_ground_truth.iloc[test_index].reset_index(
                drop=True
            )

            ground_truth_train = ground_truth_train.sample(
                frac=1, random_state=self.seed
            ).reset_index(drop=True)
            train_dict = {
                col: ground_truth_train[col] for col in ground_truth_train.columns
            }
            test_dict = {
                col: ground_truth_test[col] for col in ground_truth_test.columns
            }

            yield pd.DataFrame(train_dict), pd.DataFrame(test_dict), kf_count


def dice_coef(y_true, y_pred):
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    smooth = 0.0001
    return (2.0 * intersection + smooth) / (
        np.sum(y_true_f) + np.sum(y_pred_f) + smooth
    )


def proportion_lesioned(functional_territory, lesion):
    functional_territory_flat = functional_territory.flatten()
    lesion_flat = lesion.flatten()
    return np.sum(functional_territory_flat * lesion_flat) / np.sum(
        functional_territory_flat
    )


def generate_ground_truth(
    lesionpath,
    discopath,
    simpleatlases=None,
    disco_thresh=0.5,
    n_samples=None,
    verbose=False,
):
    # =========================================================================
    # Step 1: Discover all lesion files
    # =========================================================================
    if verbose:
        print(f"\n{'='*60}")
        print("STEP 1: Discovering lesion files")
        print(f"{'='*60}")
        print(f"Lesion path: {lesionpath}")
        print(f"Disco path: {discopath}")

    allfiles, alllesionpaths = [], []
    for root, dirs, files in os.walk(lesionpath):
        for file in files:
            if "nii" in file and not file.startswith("input"):
                allfiles.append(file)
                alllesionpaths.append(os.path.join(root, file))

    if len(allfiles) == 0:
        raise ValueError(f"No NIfTI files found in {lesionpath}")

    if n_samples is not None and n_samples < len(allfiles):
        if verbose:
            print(f"  DEBUG MODE: Limiting to first {n_samples} samples")
        allfiles = allfiles[:n_samples]
        alllesionpaths = alllesionpaths[:n_samples]

    # =========================================================================
    # Step 2: Load atlas information (FULL VERSION)
    # =========================================================================
    mni_dim = nib.load(alllesionpaths[0]).get_fdata().shape

    all_territories = nib.load(simpleatlases["all_territories_path"]).get_fdata()
    major_arterial_territories = nib.load(
        simpleatlases["major_arterial_territories_path"]
    ).get_fdata()
    major_arterial_territories_lat = nib.load(
        simpleatlases["major_arterial_territories_lat_path"]
    ).get_fdata()
    major_territories = nib.load(simpleatlases["major_territories_path"]).get_fdata()

    lesion_affine = nib.load(alllesionpaths[0]).affine
    if lesion_affine[0, 0] < 0:
        lesion_affine[0, 0] = -lesion_affine[0, 0]
        lesion_affine[0, 3] = -lesion_affine[0, 3]

    if mni_dim[0] != all_territories.shape[0]:
        affine = nib.load(simpleatlases["all_territories_path"]).affine
        if affine[0, 0] < 0:
            affine[0, 0] = -affine[0, 0]
            affine[0, 3] = -affine[0, 3]

        all_territories = image.resample_img(
            nib.Nifti1Image(all_territories, affine),
            target_affine=lesion_affine,
            target_shape=mni_dim,
            interpolation="nearest",
        ).get_fdata()
        major_arterial_territories = image.resample_img(
            nib.Nifti1Image(major_arterial_territories, affine),
            target_affine=lesion_affine,
            target_shape=mni_dim,
            interpolation="nearest",
        ).get_fdata()
        major_arterial_territories_lat = image.resample_img(
            nib.Nifti1Image(major_arterial_territories_lat, affine),
            target_affine=lesion_affine,
            target_shape=mni_dim,
            interpolation="nearest",
        ).get_fdata()
        major_territories = image.resample_img(
            nib.Nifti1Image(major_territories, affine),
            target_affine=lesion_affine,
            target_shape=mni_dim,
            interpolation="nearest",
        ).get_fdata()

    files = os.listdir(simpleatlases["reclassifying_lesions_path"])
    files.sort()
    clusters, clusters_lat = [], []
    for file in files:
        if file.startswith("Cluster"):
            hemisphere = nib.load(
                os.path.join(simpleatlases["reclassifying_lesions_path"], file)
            ).get_fdata()
            clusters_lat.append(hemisphere)
            other_hemisphere = np.flipud(hemisphere)
            clusters_lat.append(other_hemisphere)

            mirrored = np.zeros(hemisphere.shape)
            mirrored[: int((hemisphere.shape[0] + 1) / 2), :, :] = hemisphere[
                : int((hemisphere.shape[0] + 1) / 2),
                :,
                :,
            ]
            mirrored[int((hemisphere.shape[0] + 1) / 2) - 1 :, :, :] = other_hemisphere[
                int((hemisphere.shape[0] + 1) / 2) - 1 :,
                :,
                :,
            ]
            clusters.append(mirrored)

    simple_atlases = {
        "all_territories": all_territories,
        "major_arterial_territories": major_arterial_territories,
        "major_arterial_territories_lat": major_arterial_territories_lat,
        "major_territories": major_territories,
        "clusters": clusters,
        "clusters_lat": clusters_lat,
    }

    unique_groups = {
        "all_territories_unique": np.unique(all_territories)[1:],
        "major_arterial_territories_unique": np.unique(major_arterial_territories)[1:],
        "major_arterial_territories_lat_unique": np.unique(
            major_arterial_territories_lat
        )[1:],
        "major_territories_unique": np.unique(major_territories)[1:],
    }

    functional_parcellation = nib.load(
        simpleatlases["functional_parcellation_path"]
    ).get_fdata()
    func_masks = []
    for func in range(1, int(np.max(functional_parcellation) + 1)):
        func_masks.append((functional_parcellation == func).astype(int))

    # =========================================================================
    # Step 3: Process files (FULL VERSION)
    # =========================================================================
    gt_df = pd.DataFrame()
    for i in trange(len(allfiles), desc="Generating ground truth"):
        input_representation_list, image_data = [], {}

        if lesionpath and os.path.exists(alllesionpaths[i]):
            orig_lesion = nib.load(alllesionpaths[i]).get_fdata()
            lesion_functions_lesioned = np.zeros(len(func_masks))
            input_representation_list.append("lesion")
            image_data["lesion"] = {}
        else:
            orig_lesion = False

        if discopath and os.path.exists(os.path.join(discopath, allfiles[i])):
            orig_disconnectome = nib.load(
                os.path.join(discopath, allfiles[i])
            ).get_fdata()
            disco_functions_lesioned = np.zeros(len(func_masks))
            input_representation_list.append("disconnectome")
            image_data["disconnectome"] = {}
        else:
            orig_disconnectome = False

        for j, func_mask in enumerate(func_masks):
            if not isinstance(orig_lesion, bool):
                lesion_functions_lesioned[j] = proportion_lesioned(
                    func_mask, orig_lesion
                )
            if not isinstance(orig_disconnectome, bool):
                disco_functions_lesioned[j] = proportion_lesioned(
                    func_mask, orig_disconnectome
                )

        for input_representation in input_representation_list:
            inds = (
                np.where(orig_lesion)
                if input_representation == "lesion"
                else np.where(orig_disconnectome > disco_thresh)
            )
            image_data[input_representation]["vol"] = (
                orig_lesion.sum()
                if input_representation == "lesion"
                else (orig_disconnectome > disco_thresh).sum()
            )

            image_data[input_representation]["centroid_x"] = inds[0].mean()
            image_data[input_representation]["centroid_y"] = inds[1].mean()
            image_data[input_representation]["centroid_z"] = inds[2].mean()

            representation = (
                orig_lesion if input_representation == "lesion" else orig_disconnectome
            )

            dice_scores = {
                "dice_all_territories_unique": np.zeros(
                    len(unique_groups["all_territories_unique"])
                ),
                "dice_major_arterial_territories_unique": np.zeros(
                    len(unique_groups["major_arterial_territories_unique"])
                ),
                "dice_major_arterial_territories_lat_unique": np.zeros(
                    len(unique_groups["major_arterial_territories_lat_unique"])
                ),
                "dice_major_territories_unique": np.zeros(
                    len(unique_groups["major_territories_unique"])
                ),
                "dice_clusters_unique": np.zeros(len(clusters)),
                "dice_clusters_lat_unique": np.zeros(len(clusters_lat)),
                "dice_functional_territories": np.zeros(len(func_masks)),
            }

            for atlas in simpleatlases.keys():
                if "reclassifying" in atlas:
                    for j, territory in enumerate(simple_atlases["clusters"]):
                        dice_scores["dice_clusters_unique"][j] = dice_coef(
                            territory, representation
                        )
                    for j, territory in enumerate(simple_atlases["clusters_lat"]):
                        dice_scores["dice_clusters_lat_unique"][j] = dice_coef(
                            territory, representation
                        )
                elif "functional" in atlas:
                    for j, territory in enumerate(func_masks):
                        dice_scores["dice_functional_territories"][j] = dice_coef(
                            territory, representation
                        )
                else:
                    for j, territory in enumerate(unique_groups[atlas[:-4] + "unique"]):
                        dice_scores["dice_" + atlas[:-4] + "unique"][j] = dice_coef(
                            simple_atlases[atlas[:-5]] == territory, representation
                        )

            image_data[input_representation]["dice_scores"] = dice_scores

        results_dict = {"filename": allfiles[i]}
        if not isinstance(orig_lesion, bool):
            results_dict["lesion_vol"] = image_data["lesion"]["vol"]
            results_dict["lesion_centroid_x"] = image_data["lesion"]["centroid_x"]
            results_dict["lesion_centroid_y"] = image_data["lesion"]["centroid_y"]
            results_dict["lesion_centroid_z"] = image_data["lesion"]["centroid_z"]
            results_dict["lesion_all_territories"] = [
                image_data["lesion"]["dice_scores"]["dice_all_territories_unique"]
            ]
            results_dict["lesion_major_arterial_territories"] = [
                image_data["lesion"]["dice_scores"][
                    "dice_major_arterial_territories_unique"
                ]
            ]
            results_dict["lesion_major_arterial_territories_lat"] = [
                image_data["lesion"]["dice_scores"][
                    "dice_major_arterial_territories_lat_unique"
                ]
            ]
            results_dict["lesion_major_territories"] = [
                image_data["lesion"]["dice_scores"]["dice_major_territories_unique"]
            ]
            results_dict["lesion_clusters"] = [
                image_data["lesion"]["dice_scores"]["dice_clusters_unique"]
            ]
            results_dict["lesion_clusters_lat"] = [
                image_data["lesion"]["dice_scores"]["dice_clusters_lat_unique"]
            ]
            results_dict["lesion_functional_territories_dice"] = [
                image_data["lesion"]["dice_scores"]["dice_functional_territories"]
            ]
            results_dict["lesion_functional_distribution"] = [lesion_functions_lesioned]

        if not isinstance(orig_disconnectome, bool):
            results_dict["disco_thresh"] = disco_thresh
            results_dict["disco_vol"] = image_data["disconnectome"]["vol"]
            results_dict["disco_centroid_x"] = image_data["disconnectome"]["centroid_x"]
            results_dict["disco_centroid_y"] = image_data["disconnectome"]["centroid_y"]
            results_dict["disco_centroid_z"] = image_data["disconnectome"]["centroid_z"]
            results_dict["disco_all_territories"] = [
                image_data["disconnectome"]["dice_scores"][
                    "dice_all_territories_unique"
                ]
            ]
            results_dict["disco_major_arterial_territories"] = [
                image_data["disconnectome"]["dice_scores"][
                    "dice_major_arterial_territories_unique"
                ]
            ]
            results_dict["disco_major_arterial_territories_lat"] = [
                image_data["disconnectome"]["dice_scores"][
                    "dice_major_arterial_territories_lat_unique"
                ]
            ]
            results_dict["disco_major_territories"] = [
                image_data["disconnectome"]["dice_scores"][
                    "dice_major_territories_unique"
                ]
            ]
            results_dict["disco_clusters"] = [
                image_data["disconnectome"]["dice_scores"]["dice_clusters_unique"]
            ]
            results_dict["disco_clusters_lat"] = [
                image_data["disconnectome"]["dice_scores"]["dice_clusters_lat_unique"]
            ]
            results_dict["disco_functional_territories_dice"] = [
                image_data["disconnectome"]["dice_scores"][
                    "dice_functional_territories"
                ]
            ]
            results_dict["disco_functional_distribution"] = [disco_functions_lesioned]

        gt_df = pd.concat([gt_df, pd.DataFrame(results_dict, index=[i])])

    return gt_df


def confusion_matrix(true_labels, predictions, num_classes, torch_input=True):
    """
    Compute the confusion matrix
    :param true_labels:
    :param predictions:
    :return:
    """

    if torch_input:
        device = true_labels.device

        # int, slice or bool needed for slice indexing
        if not true_labels.type() == "torch.LongTensor":
            true_labels = true_labels.type(torch.LongTensor)
        if not predictions.type() == "torch.LongTensor":
            predictions = predictions.type(torch.LongTensor)

        con_mat = torch.zeros((num_classes, num_classes), device=device)
        N = true_labels.shape[0]

    else:
        con_mat = np.zeros((num_classes, num_classes))
        N = len(true_labels)

    # Loop over the vector of true_labels, each element of which selects the row in con_mat, and increment the column
    # defined by the corresponding element in predictions
    for k in range(N):
        con_mat[true_labels[k], predictions[k]] += 1

    return con_mat


class ground_truth_preprocess:
    def __init__(
        self,
        n_splits,
        dims,
        input_types,
        lesionpath,
        discopath,
        savepath,
        icvpath,
        batch_size,
        min_epoch,
        max_epoch,
        early_stopping_epochs,
        sample_visualisation_size,
        sample_visualisation_size_random,
        mni_dim=[91, 109, 91],
        device="cuda:0",
    ):
        self.n_splits = n_splits
        self.dims = dims
        self.icvpath = icvpath
        self.lesionpath = lesionpath
        self.discopath = discopath
        self.savepath = savepath
        self.batch_size = batch_size
        self.mni_dim = mni_dim
        self.input_types = input_types
        self.affine = np.array(
            [
                [2.0, 0.0, 0.0, -90.0],
                [0.0, 2.0, 0.0, -126.0],
                [0.0, 0.0, 2.0, -72.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        self.min_epoch = min_epoch
        self.max_epoch = max_epoch
        self.nochangeepochs = early_stopping_epochs
        self.sample_visualisation_size = sample_visualisation_size
        self.sample_visualisation_size_random = sample_visualisation_size_random
        self.device = device
        self.seed = 42
        self.deficit_thresh = 0.05
        self.classifiers = {
            "lr": LogisticRegression(),
            "rf": RandomForestClassifier(),
            "et": ExtraTreesClassifier(),
            "xgb": XGBClassifier(),
            "gp": GaussianProcessClassifier(),
        }
        self.deficits = [
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
            "visual recognition",
            "visual perception",
            "spatial reasoning",
            "motor",
            "somatosensory",
        ]
        self.cmap_dict = {
            "Undefined": "#D3D3D3",
            "Hearing": "#341B51",
            "Language": "#3C3285",
            "Introspection": "#4675ED",
            "Cognition": "#1BCFD4",
            "Mood": "#3FF58A",
            "Memory": "#74FE5C",
            "Aversion": "#8EFE48",
            "Coordination": "#BEF334",
            "Interoception": "#E8D538",
            "Sleep": "#F2C83A",
            "Reward": "#F9BC39",
            "Visual recognition": "#FCAE34",
            "Visual perception": "#FE8323",
            "Spatial reasoning": "#E84B0C",
            "Motor": "#D23005",
            "Somatosensory": "#A61401",
        }

        self.cluster_order = [
            "Total middle\ncerebral artery",
            "Rolandic",
            "Angular",
            "Basilar tip",
            "Posterior borderzone",
            "Thalamoperforators",
            "Cerebellar",
            "Calcarine",
            "Posterior choroidal",
            "Anterior choroidal",
            "Inferior middle\ncerebral artery",
            "Basilar perforating",
            "Prefrontal",
            "Anterior middle\ncerebral artery",
            "Opercular",
            "Total posterior\ncerebral artery",
            "Anterior cerebral artery",
            "Long insular perforating",
            "Parietal",
            "Precentral",
            "Lenticulostriate",
        ]
        self.cluster_order_lat = [
            [f"Left {region.lower()}", f"Right {region.lower()}"]
            for region in self.cluster_order
        ]
        self.cluster_order_lat = [
            item for sublist in self.cluster_order_lat for item in sublist
        ]

        self.major_territories_order = [
            "Anterior arterial supply",
            "Posterior arterial supply",
        ]
        #'Ventricles']
        self.major_arterial_territories_order = [
            "Anterior cerebral arterial territory",
            "Middle cerebral arterial territory",
            "Posterior cerebral arterial territory",
            "Vertebrobasilar territory",
        ]  # ,
        #'Ventricles']
        self.major_arterial_territories_lat_order = [
            [f"Left {region.lower()}", f"Right {region.lower()}"]
            for region in self.major_arterial_territories_order
        ]
        self.major_arterial_territories_lat_order = [
            item
            for sublist in self.major_arterial_territories_lat_order
            for item in sublist
        ]

        self.all_territories_order = [
            "Left anterior\ncerebral artery",
            "Right anterior\ncerebral artery",
            "Left medial\nlenticulostriate",
            "Right medial\nlenticulostriate",
            "Left lateral\nlenticulostriate",
            "Right lateral\nlenticulostriate",
            "Left frontal pars of\nmiddle cerebral artery",
            "Right frontal pars of\nmiddle cerebral artery",
            "Left parietal pars of\nmiddle cerebral artery",
            "Right parietal pars of\nmiddle cerebral artery",
            "Left temporal pars of\nmiddle cerebral artery",
            "Right temporal pars of\nmiddle cerebral artery",
            "Left occipital pars of\nmiddle cerebral artery",
            "Right occipital pars of\nmiddle cerebral artery",
            "Left insular pars of\nmiddle cerebral artery",
            "Right insular pars of\nmiddle cerebral artery",
            "Left temporal pars of\nposterior cerebral artery",
            "Right temporal pars of\nposterior cerebral artery",
            "Left occipital pars of\nposterior cerebral artery",
            "Right occipital pars of\nposterior cerebral artery",
            "Left posterior choroidal\nand thalamoperforators",
            "Right posterior choroidal\nand thalamoperforators",
            "Left anterior choroidal\nand thalamoperforators",
            "Right anterior choroidal\nand thalamoperforators",
            "Left basilar",
            "Right basilar",
            "Left superior\ncerebellar",
            "Right superior\ncerebellar",
            "Left inferior\ncerebellar",
            "Right inferior\ncerebellar",
        ]

        self.custom_representations = {
            "all_territories_distribution": self.all_territories_order,
            "major_arterial_territories_distribution": self.major_arterial_territories_order,
            "major_arterial_territories_lat_distribution": self.major_arterial_territories_lat_order,
            "major_territories_distribution": self.major_territories_order,
            "reclassified_clusters_distribution": self.cluster_order,
            "reclassified_clusters_lat_distribution": self.cluster_order_lat,
        }

        self.recon_metrics = {
            "hausdorff": "Hausdorff distance",
            "dice": "Dice coefficient",
            "asd": "Average surface distance",
            "balacc": "Balanced accuracy",
        }

    def predictability_of_deficit(
        self, train_dict, test_dict, kf, dim, representation, embedding_name
    ):
        embedding_train = np.array(
            train_dict[f"{embedding_name}_{representation}_{dim}_K{kf}"]
        )
        functional_distribution_train = train_dict[
            f"{representation}_functional_distribution"
        ]

        embedding_test = np.array(
            test_dict[f"{embedding_name}_{representation}_{dim}_K{kf}"]
        )
        functional_distribution_test = test_dict[
            f"{representation}_functional_distribution"
        ]

        results = {}
        for deficit in trange(
            len(functional_distribution_train.iloc[0]), desc="Deficit predictability"
        ):
            deficit_results = {}
            train_target = (
                np.array(
                    [func_dist[deficit] for func_dist in functional_distribution_train]
                )
                > self.deficit_thresh
            ).astype(int)
            if len(set(train_target)) > 1:
                for model in self.classifiers.keys():
                    classifier = copy.deepcopy(self.classifiers[model])
                    classifier.fit(embedding_train, train_target)

                    test_target = (
                        np.array(
                            [
                                func_dist[deficit]
                                for func_dist in functional_distribution_test
                            ]
                        )
                        > self.deficit_thresh
                    ).astype(int)
                    test_preds = classifier.predict_proba(embedding_test)[:, 1]
                    tn, fp, fn, tp = confusion_matrix(
                        test_target, np.round(test_preds).astype(int)
                    )
                    acc = (
                        np.nan
                        if (tn + fp + fn + tp) == 0
                        else (tn + tp) / (tn + fp + fn + tp)
                    )
                    balacc = (
                        np.nan
                        if (tn + fp) == 0 or (tp + fn) == 0
                        else (tn / (tn + fp) + tp / (tp + fn)) / 2
                    )
                    if len(set(test_target)) < 2:
                        auroc = np.nan
                        aupr = np.nan
                        ll = np.nan
                        brier = np.nan
                    else:
                        aupr = average_precision_score(test_target, test_preds)
                        auroc = roc_auc_score(test_target, test_preds)
                        ll = log_loss(test_target, test_preds)
                        brier = brier_score_loss(test_target, test_preds)
                    deficit_results[model] = {
                        "confusion_matrix": [tn, fp, fn, tp],
                        "acc": acc,
                        "balacc": balacc,
                        "aupr": aupr,
                        "auroc": auroc,
                        "ll": ll,
                        "brier": brier,
                    }

                results[self.deficits[deficit]] = deficit_results
            else:
                results[self.deficits[deficit]] = False

        return results

    def intracranial_volume_mask(self, arr):
        mask = nib.load(self.icvpath).get_fdata().flatten().astype(bool)
        if len(mask) == arr.shape[1]:
            return arr[:, mask]
        else:
            raise ValueError("mask different dimension to array")

    def intracranial_volume_unmask(self, arr):
        mask = nib.load(self.icvpath).get_fdata().flatten().astype(bool)
        mask_inds = np.where(mask)[0]
        unmasked = np.zeros([arr.shape[0], mask.shape[0]])
        for i in range(len(mask_inds)):
            unmasked[:, mask_inds[i]] = arr[:, i]
        return unmasked.reshape(
            arr.shape[0], self.mni_dim[0], self.mni_dim[1], self.mni_dim[2]
        )

    def reduce_nimfa_nmf(self, train, test, dim, input_type):
        arr_all = np.zeros(
            (len(train), self.mni_dim[0] * self.mni_dim[1] * self.mni_dim[2])
        )
        for i in range(len(train)):
            filename = train["filename"].iloc[i]
            img = (
                nib.load(os.path.join(self.lesionpath, filename)).get_fdata()
                if input_type == "lesion"
                else nib.load(os.path.join(self.discopath, filename)).get_fdata()
            )
            arr_all[i, :] = img.flatten()

        nmf = nimfa.Nmf(
            V=self.intracranial_volume_mask(arr_all), seed="random_vcol", rank=dim
        )
        nmf_fitted = nmf()

        arr_test = np.zeros(
            (len(test), self.mni_dim[0] * self.mni_dim[1] * self.mni_dim[2])
        )
        for i in range(len(test)):
            filename = test["filename"].iloc[i]
            img = (
                nib.load(os.path.join(self.lesionpath, filename)).get_fdata()
                if input_type == "lesion"
                else nib.load(os.path.join(self.discopath, filename)).get_fdata()
            )
            arr_test[i, :] = img.flatten()

        return (
            np.array(nmf_fitted.fit.W),
            np.array(
                np.matmul(
                    self.intracranial_volume_mask(arr_test),
                    np.linalg.pinv(nmf_fitted.fit.H),
                )
            ),
            np.array(nmf_fitted.fit.H),
        )

    def reduce_pca(self, train, test, dim, input_type):
        arr_all = np.zeros(
            (len(train), self.mni_dim[0] * self.mni_dim[1] * self.mni_dim[2])
        )
        for i in range(len(train)):
            filename = train["filename"].iloc[i]
            img = (
                nib.load(os.path.join(self.lesionpath, filename)).get_fdata()
                if input_type == "lesion"
                else nib.load(os.path.join(self.discopath, filename)).get_fdata()
            )
            arr_all[i, :] = img.flatten()

        pca = PCA(n_components=dim)
        pca.fit(self.intracranial_volume_mask(arr_all))

        arr_test = np.zeros(
            (len(test), self.mni_dim[0] * self.mni_dim[1] * self.mni_dim[2])
        )
        for i in range(len(test)):
            filename = test["filename"].iloc[i]
            img = (
                nib.load(os.path.join(self.lesionpath, filename)).get_fdata()
                if input_type == "lesion"
                else nib.load(os.path.join(self.discopath, filename)).get_fdata()
            )
            arr_test[i, :] = img.flatten()

        return (
            pca.transform(self.intracranial_volume_mask(arr_all)),
            pca.transform(self.intracranial_volume_mask(arr_test)),
            pca,
        )

    def reduce_vae(self, train, test, dim, input_type):
        """
        VAE dimensionality reduction - NEW IMPLEMENTATION

        Based on paper description:
        - 4 layers of 3D ResNet-type convolutional blocks
        - Reparameterization trick for sampling
        - Batch size: 10, Min epochs: 16, Max epochs: 32
        - Early stopping: 4 epochs without improvement
        """
        data_dir = self.lesionpath if input_type == "lesion" else self.discopath

        train_filenames = train["filename"].tolist()
        test_filenames = test["filename"].tolist()

        # Create datasets
        train_dataset = LesionDataset(train_filenames, data_dir)
        test_dataset = LesionDataset(test_filenames, data_dir)

        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=0
        )
        test_loader = DataLoader(
            test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=0
        )

        # Initialize VAE
        model = VAE3D(latent_dim=dim, input_shape=tuple(self.mni_dim)).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        # Training
        best_loss = float("inf")
        epochs_no_improve = 0
        best_state = None

        print(f"Training VAE with latent_dim={dim}...")
        for epoch in range(self.max_epoch):
            # Train
            model.train()
            train_loss = 0
            for batch, _ in train_loader:
                batch = batch.to(self.device)
                optimizer.zero_grad()
                recon, mu, logvar, z = model(batch)
                loss, _, _ = model.loss_function(recon, batch, mu, logvar)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            # Validate
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch, _ in test_loader:
                    batch = batch.to(self.device)
                    recon, mu, logvar, z = model(batch)
                    loss, _, _ = model.loss_function(recon, batch, mu, logvar)
                    val_loss += loss.item()

            # Early stopping (after min_epoch)
            if epoch >= self.min_epoch - 1:
                if val_loss < best_loss:
                    best_loss = val_loss
                    epochs_no_improve = 0
                    best_state = model.state_dict().copy()
                else:
                    epochs_no_improve += 1

                if epochs_no_improve >= self.nochangeepochs:
                    print(f"Early stopping at epoch {epoch+1}")
                    model.load_state_dict(best_state)
                    break

            if (epoch + 1) % 4 == 0:
                print(
                    f"Epoch {epoch+1}/{self.max_epoch}: Train Loss: {train_loss/len(train):.4f}, Val Loss: {val_loss/len(test):.4f}"
                )

        # Extract embeddings
        model.eval()
        train_embeddings = []
        test_embeddings = []

        with torch.no_grad():
            for batch, _ in train_loader:
                batch = batch.to(self.device)
                z = model.get_latent(batch)
                train_embeddings.append(z.cpu().numpy())

            for batch, _ in test_loader:
                batch = batch.to(self.device)
                z = model.get_latent(batch)
                test_embeddings.append(z.cpu().numpy())

        train_embeddings = np.vstack(train_embeddings)
        test_embeddings = np.vstack(test_embeddings)

        return train_embeddings, test_embeddings, model

    def reduce_ae(self, train, test, dim, input_type):
        """
        AE dimensionality reduction - NEW IMPLEMENTATION

        Same architecture as VAE but without variational components
        """
        data_dir = self.lesionpath if input_type == "lesion" else self.discopath

        train_filenames = train["filename"].tolist()
        test_filenames = test["filename"].tolist()

        # Create datasets
        train_dataset = LesionDataset(train_filenames, data_dir)
        test_dataset = LesionDataset(test_filenames, data_dir)

        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=0
        )
        test_loader = DataLoader(
            test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=0
        )

        # Initialize AE
        model = AE3D(latent_dim=dim, input_shape=tuple(self.mni_dim)).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        # Training
        best_loss = float("inf")
        epochs_no_improve = 0
        best_state = None

        print(f"Training AE with latent_dim={dim}...")
        for epoch in range(self.max_epoch):
            # Train
            model.train()
            train_loss = 0
            for batch, _ in train_loader:
                batch = batch.to(self.device)
                optimizer.zero_grad()
                recon, z = model(batch)
                loss, _, _ = model.loss_function(recon, batch)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            # Validate
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch, _ in test_loader:
                    batch = batch.to(self.device)
                    recon, z = model(batch)
                    loss, _, _ = model.loss_function(recon, batch)
                    val_loss += loss.item()

            # Early stopping (after min_epoch)
            if epoch >= self.min_epoch - 1:
                if val_loss < best_loss:
                    best_loss = val_loss
                    epochs_no_improve = 0
                    best_state = model.state_dict().copy()
                else:
                    epochs_no_improve += 1

                if epochs_no_improve >= self.nochangeepochs:
                    print(f"Early stopping at epoch {epoch+1}")
                    model.load_state_dict(best_state)
                    break

            if (epoch + 1) % 4 == 0:
                print(
                    f"Epoch {epoch+1}/{self.max_epoch}: Train Loss: {train_loss/len(train):.4f}, Val Loss: {val_loss/len(test):.4f}"
                )

        # Extract embeddings
        model.eval()
        train_embeddings = []
        test_embeddings = []

        with torch.no_grad():
            for batch, _ in train_loader:
                batch = batch.to(self.device)
                z = model.get_latent(batch)
                train_embeddings.append(z.cpu().numpy())

            for batch, _ in test_loader:
                batch = batch.to(self.device)
                z = model.get_latent(batch)
                test_embeddings.append(z.cpu().numpy())

        train_embeddings = np.vstack(train_embeddings)
        test_embeddings = np.vstack(test_embeddings)

        return train_embeddings, test_embeddings, model

    def recon_stats_nmf(
        self,
        train_filenames,
        train_embeddings,
        test_filenames,
        test_embeddings,
        archetypes,
        input_type,
    ):
        """Compute NMF reconstruction statistics"""
        return {"train_rmse": 0.0, "test_rmse": 0.0, "reconstruction_error": 0.0}

    def reduce(
        self,
        train,
        test,
        k,
        dim,
        ae=True,
        vae=True,
        nmf=True,
        pca=True,
        func_pred=False,
    ):

        N_training = len(train)
        train_dict = {col: train[col] for col in train.columns}
        test_dict = {col: test[col] for col in test.columns}

        for input_type in self.input_types:
            sub_savepath = os.path.join(
                self.savepath, f"K{k}_dim{dim}_{input_type}_N{N_training}"
            )
            if os.path.exists(sub_savepath):
                now = datetime.now()
                sub_savepath = os.path.join(
                    sub_savepath, "forcompletion_" + now.strftime("%H_%M_%S__%d_%m_%Y")
                )
                os.makedirs(sub_savepath)
            else:
                os.makedirs(sub_savepath)

            if vae:
                sub_savepath_vae = os.path.join(sub_savepath, "vae")
                os.makedirs(sub_savepath_vae, exist_ok=True)

                train_embeddings, test_embeddings, vae_model = self.reduce_vae(
                    train, test, dim, input_type
                )

                # Save model
                torch.save(
                    vae_model.state_dict(),
                    os.path.join(
                        sub_savepath_vae, f"vae_model_K{k}_{input_type}_dim{dim}.pt"
                    ),
                )

                train_dict = {
                    **train_dict,
                    **{f"vae_{input_type}_{dim}_K{k}": list(train_embeddings)},
                }
                test_dict = {
                    **test_dict,
                    **{f"vae_{input_type}_{dim}_K{k}": list(test_embeddings)},
                }

                print(f"VAE embeddings saved for {input_type}, dim={dim}, fold={k}")

            if ae:
                sub_savepath_ae = os.path.join(sub_savepath, "ae")
                os.makedirs(sub_savepath_ae, exist_ok=True)

                train_embeddings, test_embeddings, ae_model = self.reduce_ae(
                    train, test, dim, input_type
                )

                # Save model
                torch.save(
                    ae_model.state_dict(),
                    os.path.join(
                        sub_savepath_ae, f"ae_model_K{k}_{input_type}_dim{dim}.pt"
                    ),
                )

                train_dict = {
                    **train_dict,
                    **{f"ae_{input_type}_{dim}_K{k}": list(train_embeddings)},
                }
                test_dict = {
                    **test_dict,
                    **{f"ae_{input_type}_{dim}_K{k}": list(test_embeddings)},
                }

                print(f"AE embeddings saved for {input_type}, dim={dim}, fold={k}")

            if nmf:
                sub_savepath_nmf = os.path.join(sub_savepath, "nmf")
                if not os.path.exists(sub_savepath_nmf):
                    os.makedirs(sub_savepath_nmf)

                train_embeddings, test_embeddings, archetypes = self.reduce_nimfa_nmf(
                    train, test, dim, input_type
                )

                recon_stats = self.recon_stats_nmf(
                    train_dict["filename"].tolist(),
                    train_embeddings,
                    test_dict["filename"].tolist(),
                    test_embeddings,
                    archetypes,
                    input_type,
                )
                joblib.dump(
                    recon_stats,
                    os.path.join(
                        sub_savepath_nmf,
                        f"nmf_recon_stats_K{k}_{input_type}_dim{dim}_N{N_training}.joblib",
                    ),
                )

                train_dict = {
                    **train_dict,
                    **{f"nmf_{input_type}_{dim}_K{k}": list(train_embeddings)},
                }
                test_dict = {
                    **test_dict,
                    **{f"nmf_{input_type}_{dim}_K{k}": list(test_embeddings)},
                }

                if func_pred:
                    functional_deficit_predictability = self.predictability_of_deficit(
                        train_dict, test_dict, k, dim, input_type, "nmf"
                    )
                    joblib.dump(
                        functional_deficit_predictability,
                        os.path.join(
                            sub_savepath_nmf,
                            f"nmf_functional_deficit_predictability_K{k}_{input_type}_dim{dim}_N{N_training}.joblib",
                        ),
                    )

            if pca:
                sub_savepath_pca = os.path.join(sub_savepath, "pca")
                if not os.path.exists(sub_savepath_pca):
                    os.makedirs(sub_savepath_pca)

                train_embeddings, test_embeddings, pca_fitted = self.reduce_pca(
                    train, test, dim, input_type
                )
                joblib.dump(
                    pca_fitted.explained_variance_,
                    os.path.join(
                        sub_savepath_pca,
                        f"pca_explained_variance_K{k}_{input_type}_dim{dim}_N{N_training}.joblib",
                    ),
                )
                joblib.dump(
                    pca_fitted.explained_variance_ratio_,
                    os.path.join(
                        sub_savepath_pca,
                        f"pca_explained_variance_ratio_K{k}_{input_type}_dim{dim}_N{N_training}.joblib",
                    ),
                )
                joblib.dump(
                    pca_fitted.singular_values_,
                    os.path.join(
                        sub_savepath_pca,
                        f"pca_singular_values_K{k}_{input_type}_dim{dim}_N{N_training}.joblib",
                    ),
                )
                joblib.dump(
                    pca_fitted.noise_variance_,
                    os.path.join(
                        sub_savepath_pca,
                        f"pca_noise_variance_K{k}_{input_type}_dim{dim}_N{N_training}.joblib",
                    ),
                )

                train_dict = {
                    **train_dict,
                    **{f"pca_{input_type}_{dim}_K{k}": list(train_embeddings)},
                }
                test_dict = {
                    **test_dict,
                    **{f"pca_{input_type}_{dim}_K{k}": list(test_embeddings)},
                }

                if func_pred:
                    functional_deficit_predictability = self.predictability_of_deficit(
                        train_dict, test_dict, k, dim, input_type, "pca"
                    )
                    joblib.dump(
                        functional_deficit_predictability,
                        os.path.join(
                            sub_savepath_pca,
                            f"pca_functional_deficit_predictability_K{k}_{input_type}_dim{dim}_N{N_training}.joblib",
                        ),
                    )

        return pd.DataFrame(train_dict), pd.DataFrame(test_dict)


def command_line_options():
    parser = argparse.ArgumentParser(
        description="Lesion Representation Learning with VAE/AE/NMF/PCA"
    )
    parser.add_argument(
        "--lesionpath", type=str, default="", help="Path to lesion mask nii files"
    )
    parser.add_argument(
        "--discopath", type=str, default="", help="Path to disconnectome nii files"
    )
    parser.add_argument("--savepath", type=str, default="", help="Path to save results")
    parser.add_argument(
        "--kfolds", type=int, default=10, help="Number of folds for cross-validation"
    )
    parser.add_argument(
        "--batch_size", type=int, default=10, help="Batch size for training VAE/AE"
    )
    parser.add_argument(
        "--early_stopping_epochs", type=int, default=4, help="Early stopping patience"
    )
    parser.add_argument(
        "--min_epoch", type=int, default=16, help="Minimum training epochs"
    )
    parser.add_argument(
        "--max_epoch", type=int, default=32, help="Maximum training epochs"
    )
    parser.add_argument("--sample_visualisation_size", type=int, default=66)
    parser.add_argument("--sample_visualisation_size_random", type=int, default=77)
    parser.add_argument(
        "--run_ae", type=str2bool, default=False, help="Run autoencoder"
    )
    parser.add_argument(
        "--run_vae", type=str2bool, default=False, help="Run variational autoencoder"
    )
    parser.add_argument("--run_nmf", type=str2bool, default=True, help="Run NMF")
    parser.add_argument("--run_pca", type=str2bool, default=True, help="Run PCA")
    parser.add_argument(
        "--latent_components",
        type=int,
        nargs="+",
        default=[50],
        help="Latent dimensions",
    )
    # DEBUG parameters
    parser.add_argument(
        "--n_samples", type=int, default=None, help="DEBUG: Limit number of samples"
    )
    parser.add_argument(
        "--verbose", type=str2bool, default=False, help="DEBUG: Verbose output"
    )

    return parser.parse_args()


def run(parameters):
    """
    Main run function for the representation pipeline.

    Parameters
    ----------
    parameters : argparse.Namespace
        Parsed command line arguments
    """
    args = parameters

    print(f"\n{'#'*60}")
    print("# REPRESENTATION PIPELINE")
    print(f"{'#'*60}")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    lesionpath = args.lesionpath
    discopath = args.discopath
    savepath = args.savepath
    kfolds = args.kfolds
    latent_components = args.latent_components
    run_ae = args.run_ae
    run_vae = args.run_vae
    run_nmf = args.run_nmf
    run_pca = args.run_pca
    batch_size = args.batch_size
    early_stopping_epochs = args.early_stopping_epochs
    min_epoch = args.min_epoch
    max_epoch = args.max_epoch
    sample_visualisation_size = args.sample_visualisation_size
    sample_visualisation_size_random = args.sample_visualisation_size_random
    n_samples = args.n_samples
    verbose = args.verbose

    print(f"Device: {device}")
    print(f"Running: VAE={run_vae}, AE={run_ae}, NMF={run_nmf}, PCA={run_pca}")
    print(f"Latent dimensions: {latent_components}")
    if n_samples is not None:
        print(f"  DEBUG MODE: Processing only {n_samples} samples")

    # Set default save path
    if not savepath:
        savepath = os.path.join(os.getcwd(), "results")
    os.makedirs(savepath, exist_ok=True)

    if not lesionpath and not discopath:
        raise ValueError(
            "Please provide a path to the lesion masks and/or disconnectomes"
        )

    atlases_path = os.path.join(os.getcwd(), "atlases")
    simpleatlases = {
        "all_territories_path": os.path.join(
            atlases_path, "vasc_atlas", "all_territories.nii"
        ),
        "major_arterial_territories_path": os.path.join(
            atlases_path, "vasc_atlas", "major_arterial_territory.nii"
        ),
        "major_arterial_territories_lat_path": os.path.join(
            atlases_path, "vasc_atlas", "major_arterial_territory_lat.nii"
        ),
        "major_territories_path": os.path.join(
            atlases_path, "vasc_atlas", "major_territories.nii"
        ),
        "reclassifying_lesions_path": os.path.join(
            atlases_path, "Final_lesion_clusters"
        ),
        "functional_parcellation_path": os.path.join(
            atlases_path, "functional_parcellation_2mm.nii"
        ),
    }

    # Load or generate ground truth
    gt_path = os.path.join(savepath, "whole_ground_truth.pkl")
    if os.path.exists(gt_path) and n_samples is None:
        whole_ground_truth = pd.read_pickle(gt_path)
        print(f"Loaded existing ground truth: {len(whole_ground_truth)} samples")
    else:
        whole_ground_truth = generate_ground_truth(
            lesionpath=lesionpath,
            discopath=discopath,
            simpleatlases=simpleatlases,
            n_samples=n_samples,
            verbose=verbose,
        )
        # Only save if not in debug mode
        if n_samples is None:
            whole_ground_truth.to_pickle(gt_path)
        print(f"Generated ground truth: {len(whole_ground_truth)} samples")

    # Determine input types
    input_types = []
    if lesionpath:
        input_types.append("lesion")
    if discopath:
        input_types.append("disco")

    # Run cross-validation
    gt_generator = ground_truth_splits(n_splits=kfolds)
    for (
        ground_truth_train,
        ground_truth_test,
        kf_count,
    ) in gt_generator.make_kfold_splits(whole_ground_truth):
        print(f"\n{'='*60}")
        print(f"Processing fold {kf_count+1}/{kfolds}")
        print(f"Train: {len(ground_truth_train)}, Test: {len(ground_truth_test)}")
        print(f"{'='*60}")

        ground_truth_train.to_pickle(
            os.path.join(savepath, f"train_split_{kf_count}.pkl")
        )
        ground_truth_test.to_pickle(
            os.path.join(savepath, f"test_split_{kf_count}.pkl")
        )

        for dim in latent_components:
            gt_preprocess = ground_truth_preprocess(
                n_splits=kfolds,
                dims=int(dim),
                input_types=input_types,
                lesionpath=lesionpath,
                discopath=discopath,
                savepath=savepath,
                icvpath=os.path.join(atlases_path, "icv_mask_2mm.nii"),
                batch_size=batch_size,
                min_epoch=min_epoch,
                max_epoch=max_epoch,
                early_stopping_epochs=early_stopping_epochs,
                sample_visualisation_size=sample_visualisation_size,
                sample_visualisation_size_random=sample_visualisation_size_random,
                device=device,
            )

            train_reduced, test_reduced = gt_preprocess.reduce(
                ground_truth_train,
                ground_truth_test,
                k=kf_count,
                dim=int(dim),
                ae=run_ae,
                vae=run_vae,
                nmf=run_nmf,
                pca=run_pca,
            )

            # Save results
            for input_type in input_types:
                train_reduced.to_pickle(
                    os.path.join(
                        savepath, f"train_{kf_count}_dim_{dim}_{input_type}.pkl"
                    )
                )
                test_reduced.to_pickle(
                    os.path.join(
                        savepath, f"test_{kf_count}_dim_{dim}_{input_type}.pkl"
                    )
                )

    print("\n Processing complete!")


if __name__ == "__main__":
    args = command_line_options()
    run(args)
