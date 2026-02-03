# =============================================================================
# Download Atlases Module
# =============================================================================
"""
Module for downloading and preprocessing vascular territory atlases for stroke lesion analysis.

References:
    Liu et al. (2023). Digital 3D brain MRI arterial territories atlas.
    Scientific Data, 10(1), 1-12. https://doi.org/10.1038/s41597-022-01923-0

Usage:
    from download_atlases import AtlasDownloader, run
    
    # As module
    downloader = AtlasDownloader(output_dir="./atlases", nitrc_dir="./NITRC")
    downloader.download_and_process()
    
    # From command line
    python download_atlases.py --outputpath ./atlases --nitrcpath ./NITRC
"""

import os
import glob
import gzip
import shutil
import argparse
from typing import Tuple, List, Optional, Dict

import numpy as np
import nibabel as nib
from nilearn import image


# =============================================================================
# Constants
# =============================================================================
TARGET_SHAPE = (91, 109, 91)

COMPRESSED_FILES = [
    "functional_parcellation_2mm.nii.gz",
    "icv_mask_2mm.nii.gz"
]

VASCULAR_ATLAS_FILES = [
    "all_territories.nii",
    "major_arterial_territory.nii",
    "major_arterial_territory_lat.nii",
    "major_territories.nii"
]


# =============================================================================
# Utility Functions
# =============================================================================
def load_and_resample_to_lesion_space(
    atlas_path: str,
    reference_path: str
) -> Tuple[nib.Nifti1Image, np.ndarray, np.ndarray]:
    """
    Load NIfTI atlas and resample to match lesion mask space exactly.
    Handles 4D inputs by taking first volume.
    
    Parameters
    ----------
    atlas_path : str
        Path to the atlas NIfTI file.
    reference_path : str
        Path to the reference image defining target space.
        
    Returns
    -------
    Tuple[nib.Nifti1Image, np.ndarray, np.ndarray]
        Resampled image, data array, and affine matrix.
    """
    atlas_img = nib.load(atlas_path)
    atlas_data = atlas_img.get_fdata()
    
    # Handle 4D: take first volume
    if atlas_data.ndim == 4:
        atlas_data = atlas_data[:, :, :, 0]
        atlas_img = nib.Nifti1Image(atlas_data, atlas_img.affine)
    
    # Load reference (lesion or ICV mask) for target space
    ref_img = nib.load(reference_path)
    
    # Resample atlas to reference space
    resampled = image.resample_to_img(
        atlas_img,
        ref_img,
        interpolation="nearest"
    )
    
    return resampled, resampled.get_fdata().astype(np.int16), resampled.affine


def lateralize_atlas(data: np.ndarray, n_labels: int = 4) -> np.ndarray:
    """
    Split atlas by hemisphere: left=[1,n], right=[n+1,2n].
    
    Parameters
    ----------
    data : np.ndarray
        Atlas data array.
    n_labels : int
        Number of labels per hemisphere.
        
    Returns
    -------
    np.ndarray
        Lateralized atlas data.
    """
    mid = data.shape[0] // 2
    out = np.zeros_like(data)
    out[:mid] = data[:mid]
    out[mid:] = np.where(data[mid:] > 0, data[mid:] + n_labels, 0)
    return out


def classify_circulation(data: np.ndarray) -> np.ndarray:
    """
    Map territories to anterior (1) vs posterior (2) circulation.
    
    Parameters
    ----------
    data : np.ndarray
        Major arterial territory data.
        
    Returns
    -------
    np.ndarray
        Circulation classification (1=anterior, 2=posterior).
    """
    out = np.zeros_like(data)
    out[np.isin(data, [1, 2])] = 1  # ACA + MCA -> Anterior
    out[np.isin(data, [3, 4])] = 2  # PCA + VB -> Posterior
    return out


# =============================================================================
# Atlas Downloader Class
# =============================================================================
class AtlasDownloader:
    """
    Downloads and preprocesses vascular territory atlases for stroke lesion analysis.
    
    Attributes
    ----------
    output_dir : str
        Directory where atlases will be saved.
    nitrc_dir : str
        Directory containing NITRC arterial atlas files.
    vasc_atlas_dir : str
        Directory for vascular territory atlases.
    target_shape : tuple
        Expected shape for all atlas files.
    """
    
    def __init__(
        self,
        output_dir: str,
        nitrc_dir: str,
        source_atlas_dir: Optional[str] = None,
        target_shape: Tuple[int, int, int] = TARGET_SHAPE
    ):
        """
        Initialize the AtlasDownloader.
        
        Parameters
        ----------
        output_dir : str
            Directory where atlases will be saved.
        nitrc_dir : str
            Directory containing NITRC arterial atlas files.
        source_atlas_dir : str, optional
            Source directory to sync atlases from (e.g., from repository).
        target_shape : tuple
            Expected shape for all atlas files.
        """
        self.output_dir = output_dir
        self.nitrc_dir = nitrc_dir
        self.source_atlas_dir = source_atlas_dir
        self.vasc_atlas_dir = os.path.join(output_dir, "vasc_atlas")
        self.target_shape = target_shape
        
        os.makedirs(self.output_dir, exist_ok=True)
    
    def sync_from_source(self) -> bool:
        """
        Sync atlas directory from source repository if available.
        
        Returns
        -------
        bool
            True if sync was performed or directory exists.
        """
        if self.source_atlas_dir and os.path.exists(self.source_atlas_dir):
            if not os.path.exists(self.output_dir):
                shutil.copytree(self.source_atlas_dir, self.output_dir)
                print(f"Synced atlas directory to {self.output_dir}")
                return True
            else:
                print(f"Atlas directory already exists at {self.output_dir}")
                return True
        else:
            os.makedirs(self.output_dir, exist_ok=True)
            if self.source_atlas_dir:
                print(f"Source atlas directory not found: {self.source_atlas_dir}")
            print(f"Created atlas directory at {self.output_dir}")
            return False
    
    def decompress_files(self, verbose: bool = True) -> Dict[str, bool]:
        """
        Decompress required .nii.gz files to .nii format.
        
        Parameters
        ----------
        verbose : bool
            Print progress messages.
            
        Returns
        -------
        Dict[str, bool]
            Dictionary mapping filenames to decompression status.
        """
        results = {}
        
        for gz_name in COMPRESSED_FILES:
            gz_path = os.path.join(self.output_dir, gz_name)
            nii_path = os.path.join(self.output_dir, gz_name.replace(".gz", ""))
            
            if os.path.exists(gz_path) and not os.path.exists(nii_path):
                try:
                    with gzip.open(gz_path, 'rb') as f_in, open(nii_path, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
                    if verbose:
                        print(f"Decompressed: {gz_name}")
                    results[gz_name] = True
                except Exception as e:
                    if verbose:
                        print(f"ERROR decompressing {gz_name}: {e}")
                    results[gz_name] = False
            elif os.path.exists(nii_path):
                if verbose:
                    print(f"Already exists: {os.path.basename(nii_path)}")
                results[gz_name] = True
            else:
                if verbose:
                    print(f"WARNING: Missing source file {gz_name}")
                results[gz_name] = False
        
        return results
    
    def get_reference_path(self) -> Optional[str]:
        """Get path to reference image (ICV mask)."""
        ref_path = os.path.join(self.output_dir, "icv_mask_2mm.nii")
        if os.path.exists(ref_path):
            return ref_path
        return None
    
    def build_vascular_atlases(self, verbose: bool = True) -> Dict[str, bool]:
        """
        Build vascular territory atlases resampled to lesion space.
        
        Parameters
        ----------
        verbose : bool
            Print progress messages.
            
        Returns
        -------
        Dict[str, bool]
            Dictionary mapping atlas names to creation status.
        """
        os.makedirs(self.vasc_atlas_dir, exist_ok=True)
        results = {}
        
        reference_path = self.get_reference_path()
        if not reference_path:
            print(f"ERROR: Reference image not found. Run decompress_files() first.")
            return results
        
        # 1. Full parcellation (30 territories)
        full_atlas_path = os.path.join(self.nitrc_dir, "ArterialAtlas.nii")
        if os.path.exists(full_atlas_path):
            try:
                full_img, _, _ = load_and_resample_to_lesion_space(full_atlas_path, reference_path)
                output_path = os.path.join(self.vasc_atlas_dir, "all_territories.nii")
                nib.save(full_img, output_path)
                if verbose:
                    print(f"all_territories.nii: {full_img.shape}")
                results["all_territories.nii"] = True
            except Exception as e:
                if verbose:
                    print(f"ERROR creating all_territories.nii: {e}")
                results["all_territories.nii"] = False
        else:
            if verbose:
                print(f"WARNING: {full_atlas_path} not found")
            results["all_territories.nii"] = False
        
        # 2. Major territories (ACA, MCA, PCA, VB)
        level2_path = os.path.join(self.nitrc_dir, "ArterialAtlas_level2.nii")
        if os.path.exists(level2_path):
            try:
                level2_img, level2_data, affine = load_and_resample_to_lesion_space(
                    level2_path, reference_path
                )
                
                # Save major arterial territory
                output_path = os.path.join(self.vasc_atlas_dir, "major_arterial_territory.nii")
                nib.save(level2_img, output_path)
                if verbose:
                    print(f"major_arterial_territory.nii: {level2_img.shape}")
                results["major_arterial_territory.nii"] = True
                
                # 3. Lateralized version (L/R hemisphere split)
                lat_data = lateralize_atlas(level2_data, n_labels=4)
                output_path = os.path.join(self.vasc_atlas_dir, "major_arterial_territory_lat.nii")
                nib.save(nib.Nifti1Image(lat_data, affine), output_path)
                if verbose:
                    print(f"major_arterial_territory_lat.nii: labels {np.unique(lat_data)}")
                results["major_arterial_territory_lat.nii"] = True
                
                # 4. Anterior vs posterior circulation
                circ_data = classify_circulation(level2_data)
                output_path = os.path.join(self.vasc_atlas_dir, "major_territories.nii")
                nib.save(nib.Nifti1Image(circ_data, affine), output_path)
                if verbose:
                    anterior_voxels = np.sum(circ_data == 1)
                    posterior_voxels = np.sum(circ_data == 2)
                    print(f"major_territories.nii: anterior={anterior_voxels}, posterior={posterior_voxels} voxels")
                results["major_territories.nii"] = True
                
            except Exception as e:
                if verbose:
                    print(f"ERROR processing level2 atlas: {e}")
                results["major_arterial_territory.nii"] = False
                results["major_arterial_territory_lat.nii"] = False
                results["major_territories.nii"] = False
        else:
            if verbose:
                print(f"WARNING: {level2_path} not found")
            results["major_arterial_territory.nii"] = False
            results["major_arterial_territory_lat.nii"] = False
            results["major_territories.nii"] = False
        
        return results
    
    def download_and_process(self, verbose: bool = True) -> bool:
        """
        Execute full download and processing pipeline.
        
        Parameters
        ----------
        verbose : bool
            Print progress messages.
            
        Returns
        -------
        bool
            True if all steps completed successfully.
        """
        if verbose:
            print("=" * 50)
            print("Step 1: Sync Atlas Directory")
            print("=" * 50)
        self.sync_from_source()
        
        if verbose:
            print("\n" + "=" * 50)
            print("Step 2: Decompress Required Files")
            print("=" * 50)
        decompress_results = self.decompress_files(verbose=verbose)
        
        if verbose:
            print("\n" + "=" * 50)
            print("Step 3: Build Vascular Territory Atlases")
            print("=" * 50)
        build_results = self.build_vascular_atlases(verbose=verbose)
        
        if verbose:
            print("\nAtlas preprocessing complete.")
        
        # Check if all critical files were created
        all_success = all(decompress_results.values()) and all(build_results.values())
        return all_success
    
    def get_atlas_files(self) -> List[str]:
        """Get list of all expected atlas file paths."""
        files = []
        
        # Vascular atlas files
        for filename in VASCULAR_ATLAS_FILES:
            files.append(os.path.join(self.vasc_atlas_dir, filename))
        
        # Base atlas files
        for gz_name in COMPRESSED_FILES:
            nii_name = gz_name.replace(".gz", "")
            files.append(os.path.join(self.output_dir, nii_name))
        
        return files
    
    def verify(self, verbose: bool = True) -> Dict[str, dict]:
        """
        Verify all atlas files exist and have correct shapes.
        
        Parameters
        ----------
        verbose : bool
            Print verification results.
            
        Returns
        -------
        Dict[str, dict]
            Dictionary mapping filenames to verification info.
        """
        results = {}
        
        files_to_verify = self.get_atlas_files()
        
        if verbose:
            print(f"\n{'FILE':<45} | {'SHAPE':<15} | {'STATUS'}")
            print("-" * 75)
        
        for path in files_to_verify:
            filename = os.path.basename(path)
            
            if os.path.exists(path):
                try:
                    img = nib.load(path)
                    shape = img.shape
                    is_compatible = shape == self.target_shape
                    status = "OK" if is_compatible else "SHAPE MISMATCH"
                    
                    results[filename] = {
                        "exists": True,
                        "shape": shape,
                        "compatible": is_compatible,
                        "path": path
                    }
                    
                    if verbose:
                        print(f"{filename:<45} | {str(shape):<15} | {status}")
                        
                except Exception as e:
                    results[filename] = {
                        "exists": True,
                        "error": str(e),
                        "compatible": False,
                        "path": path
                    }
                    if verbose:
                        print(f"{filename:<45} | {'ERROR':<15} | {e}")
            else:
                results[filename] = {
                    "exists": False,
                    "compatible": False,
                    "path": path
                }
                if verbose:
                    print(f"{filename:<45} | {'NOT FOUND':<15} | CHECK PATH")
        
        return results
    
    def get_atlas_paths(self) -> Dict[str, str]:
        """
        Get dictionary of atlas paths for use in other modules.
        
        Returns
        -------
        Dict[str, str]
            Dictionary mapping atlas names to file paths.
        """
        return {
            "all_territories_path": os.path.join(self.vasc_atlas_dir, "all_territories.nii"),
            "major_arterial_territories_path": os.path.join(self.vasc_atlas_dir, "major_arterial_territory.nii"),
            "major_arterial_territories_lat_path": os.path.join(self.vasc_atlas_dir, "major_arterial_territory_lat.nii"),
            "major_territories_path": os.path.join(self.vasc_atlas_dir, "major_territories.nii"),
            "functional_parcellation_path": os.path.join(self.output_dir, "functional_parcellation_2mm.nii"),
            "icv_mask_path": os.path.join(self.output_dir, "icv_mask_2mm.nii")
        }


# =============================================================================
# CLI Interface
# =============================================================================
def command_line_options() -> Tuple[Tuple[str, str, str], bool]:
    """
    Parse command line arguments.
    
    Returns
    -------
    tuple
        (paths, verify_only)
        - paths: (outputpath, nitrcpath, sourcepath)
        - verify_only: bool
    """
    parser = argparse.ArgumentParser(
        description="Download and preprocess vascular territory atlases for stroke lesion analysis.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process atlases with NITRC source
  python download_atlases.py --outputpath ./atlases --nitrcpath ./NITRC
  
  # Sync from repository and process
  python download_atlases.py --outputpath ./atlases --nitrcpath ./NITRC --sourcepath ./repo/atlases
  
  # Verify existing atlases
  python download_atlases.py --outputpath ./atlases --nitrcpath ./NITRC --verify

References:
  Liu et al. (2023). Digital 3D brain MRI arterial territories atlas.
  Scientific Data, 10(1), 1-12. https://doi.org/10.1038/s41597-022-01923-0
        """
    )
    
    # Path arguments
    parser.add_argument("--outputpath", type=str, default="",
                        help="Path to save atlas files (default: ./atlases)")
    parser.add_argument("--nitrcpath", type=str, default="",
                        help="Path to NITRC arterial atlas files")
    parser.add_argument("--sourcepath", type=str, default="",
                        help="Path to source atlas directory (optional, for syncing)")
    
    # Mode flags
    parser.add_argument("--verify", action="store_true",
                        help="Only verify existing atlases")
    
    args = parser.parse_args()
    
    paths = (args.outputpath, args.nitrcpath, args.sourcepath)
    
    return (paths, args.verify)


def run(parameters: Tuple) -> Dict[str, str]:
    """
    Main execution function for atlas download and processing.
    
    Parameters
    ----------
    parameters : tuple
        Output from command_line_options():
        (paths, verify_only)
        
    Returns
    -------
    Dict[str, str]
        Dictionary of atlas paths for use in other modules.
    """
    paths, verify_only = parameters
    outputpath, nitrcpath, sourcepath = paths
    
    # Set default paths if not provided
    if not outputpath:
        outputpath = os.path.join(os.getcwd(), "atlases")
        print(f"Using default output path: {outputpath}")
    
    if not nitrcpath:
        nitrcpath = os.path.join(os.getcwd(), "NITRC")
        print(f"Using default NITRC path: {nitrcpath}")
    
    # Create downloader
    downloader = AtlasDownloader(
        output_dir=outputpath,
        nitrc_dir=nitrcpath,
        source_atlas_dir=sourcepath if sourcepath else None
    )
    
    # Print configuration
    print("=" * 50)
    print("Atlas Download Configuration")
    print("=" * 50)
    print(f"Output path: {outputpath}")
    print(f"Vascular atlas dir: {downloader.vasc_atlas_dir}")
    print(f"NITRC path: {nitrcpath}")
    print(f"Source path: {sourcepath if sourcepath else 'Not specified'}")
    print(f"Target shape: {downloader.target_shape}")
    print("=" * 50)
    
    # Verify only mode
    if verify_only:
        print("\nVerification Mode")
        downloader.verify(verbose=True)
        return downloader.get_atlas_paths()
    
    # Execute full pipeline
    success = downloader.download_and_process(verbose=True)
    
    # Final verification
    print("\n" + "=" * 50)
    print("Final Verification")
    print("=" * 50)
    downloader.verify(verbose=True)
    
    if success:
        print("\nAll atlas files processed successfully!")
    else:
        print("\nWARNING: Some atlas files could not be processed.")
        print("Check that NITRC atlas files exist in the specified path.")
    
    return downloader.get_atlas_paths()


if __name__ == "__main__":
    parameters = command_line_options()
    run(parameters)