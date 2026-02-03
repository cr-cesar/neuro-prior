# =============================================================================
# Disconnectomes Module
# =============================================================================
"""
Module for generating disconnectomes from lesion masks using the Functionnectome library.

Usage:
    from disconnectomes import DisconnectomeGenerator, run
    
    # Como módulo
    generator = DisconnectomeGenerator(
        lesions_dir="path/to/lesions",
        output_dir="path/to/output",
        priors_h5="path/to/priors.h5"
    )
    generator.process_all(batch_size=100, n_proc=4)
    
    # Desde línea de comandos
    python disconnectomes.py --lesionpath ./lesions --outputpath ./disconnectomes --priorspath ./priors.h5
"""

import os
import glob
import pickle
import argparse
from typing import Optional, Set, List, Tuple

import nibabel as nib
import numpy as np
from tqdm import tqdm

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

from Functionnectome import quickDisco
from Functionnectome import functionnectome as fun


class DisconnectomeGenerator:
    """
    Generates disconnectomes from lesion masks using anatomical priors.
    
    Attributes
    ----------
    lesions_dir : str
        Directory containing lesion mask files (.nii or .nii.gz).
    output_dir : str
        Directory where disconnectomes will be saved.
    priors_h5 : str
        Path to the HDF5 file containing anatomical priors.
    progress_file : str
        Path to the pickle file tracking processed files.
    """
    
    def __init__(
        self,
        lesions_dir: str,
        output_dir: str,
        priors_h5: str,
        progress_file: Optional[str] = None
    ):
        self.lesions_dir = lesions_dir
        self.output_dir = output_dir
        self.priors_h5 = priors_h5
        self.progress_file = progress_file or os.path.join(output_dir, "disconn_progress.pkl")
        
        os.makedirs(self.output_dir, exist_ok=True)
        self._validate_paths()
    
    def _validate_paths(self) -> None:
        """Validate that required paths exist."""
        if not os.path.exists(self.lesions_dir):
            raise FileNotFoundError(f"Lesions directory not found: {self.lesions_dir}")
        if not os.path.exists(self.priors_h5):
            raise FileNotFoundError(f"Priors file not found: {self.priors_h5}")
    
    @staticmethod
    def download_priors(output_dir: str, prior_name: str = 'V2.P.WB - Whole brain, Probabilistic') -> str:
        """Download anatomical priors from the Functionnectome repository."""
        os.makedirs(output_dir, exist_ok=True)
        fun.Download_H5(output_dir, prior_name)
        
        priors_h5 = os.path.join(output_dir, "priors_full_proba_3T_comp9_thr0p01.h5")
        
        if not os.path.exists(priors_h5):
            raise RuntimeError(f"Failed to download priors to {priors_h5}")
        
        return priors_h5
    
    @staticmethod
    def get_optimal_n_proc() -> int:
        """Determine optimal number of processes based on system resources."""
        cpu_cores = os.cpu_count() or 1
        
        if PSUTIL_AVAILABLE:
            ram_gb = round(psutil.virtual_memory().total / (1024**3))
            if ram_gb > 20:
                return cpu_cores
            else:
                return min(2, cpu_cores)
        else:
            return min(4, cpu_cores)
    
    def generate_single(
        self,
        lesion_path: str,
        output_path: str,
        n_proc: int = 1
    ) -> np.ndarray:
        """Generate a disconnectome for a single lesion mask."""
        quickDisco.probaMap_fromMask(
            roiFile=lesion_path,
            priorsLoc=self.priors_h5,
            priors_type='h5',
            outFile=output_path,
            proc=n_proc,
            maxVal=True
        )
        
        disconn_img = nib.load(output_path)
        return disconn_img.get_fdata()
    
    def _load_progress(self) -> Set[str]:
        """Load previously processed files from progress file."""
        if os.path.exists(self.progress_file):
            with open(self.progress_file, 'rb') as f:
                return pickle.load(f)
        return set()
    
    def _save_progress(self, processed: Set[str]) -> None:
        """Save progress to pickle file."""
        with open(self.progress_file, 'wb') as f:
            pickle.dump(processed, f)
    
    def get_lesion_files(self) -> List[str]:
        """Get sorted list of all lesion files."""
        return sorted(glob.glob(os.path.join(self.lesions_dir, "*.nii*")))
    
    def get_pending_files(self) -> List[str]:
        """Get list of lesion files that haven't been processed yet."""
        processed = self._load_progress()
        lesion_files = self.get_lesion_files()
        return [f for f in lesion_files if os.path.basename(f) not in processed]
    
    def process_batch(
        self,
        batch_size: int = 100,
        n_proc: Optional[int] = None,
        save_interval: int = 50,
        verbose: bool = True
    ) -> int:
        """Process a batch of lesion files."""
        if n_proc is None:
            n_proc = self.get_optimal_n_proc()
        
        processed = self._load_progress()
        pending = self.get_pending_files()
        
        if verbose:
            print(f"Previously processed: {len(processed)} files")
            print(f"Pending: {len(pending)} files")
            print(f"Using N_PROC: {n_proc}")
        
        batch = pending[:batch_size]
        success_count = 0
        
        iterator = tqdm(batch, desc="Generating disconnectomes") if verbose else batch
        
        for lesion_path in iterator:
            filename = os.path.basename(lesion_path)
            output_path = os.path.join(self.output_dir, filename)
            
            if not os.path.exists(output_path):
                try:
                    self.generate_single(lesion_path, output_path, n_proc=n_proc)
                    processed.add(filename)
                    success_count += 1
                except Exception as e:
                    if verbose:
                        print(f"\nError processing {filename}: {e}")
            else:
                processed.add(filename)
                success_count += 1
            
            if len(processed) % save_interval == 0:
                self._save_progress(processed)
        
        self._save_progress(processed)
        
        if verbose:
            total = len(self.get_lesion_files())
            print(f"\nBatch completed: {len(processed)} / {total} total files")
            print(f"Remaining: {total - len(processed)} files")
        
        return success_count
    
    def process_all(
        self,
        batch_size: int = 100,
        n_proc: Optional[int] = None,
        save_interval: int = 50,
        verbose: bool = True
    ) -> int:
        """Process all pending lesion files."""
        total_processed = 0
        
        while self.get_pending_files():
            processed = self.process_batch(
                batch_size=batch_size,
                n_proc=n_proc,
                save_interval=save_interval,
                verbose=verbose
            )
            total_processed += processed
            
            if processed == 0:
                break
        
        return total_processed
    
    def verify_results(self, verbose: bool = True) -> dict:
        """Verify the generated disconnectomes."""
        lesion_files = self.get_lesion_files()
        disconn_files = sorted(glob.glob(os.path.join(self.output_dir, "*.nii*")))
        
        stats = {
            "total_lesions": len(lesion_files),
            "total_disconnectomes": len(disconn_files),
            "completion_rate": len(disconn_files) / len(lesion_files) * 100 if lesion_files else 0
        }
        
        if disconn_files and lesion_files:
            sample_disconn = nib.load(disconn_files[0])
            sample_lesion = nib.load(lesion_files[0])
            
            disconn_data = sample_disconn.get_fdata()
            
            stats["lesion_shape"] = sample_lesion.shape
            stats["disconn_shape"] = sample_disconn.shape
            stats["shapes_match"] = sample_lesion.shape == sample_disconn.shape
            stats["value_range"] = (float(disconn_data.min()), float(disconn_data.max()))
        
        if verbose:
            print(f"Disconnectomes generated: {stats['total_disconnectomes']}")
            print(f"Completion rate: {stats['completion_rate']:.1f}%")
            if "shapes_match" in stats:
                print(f"\nShape verification:")
                print(f"  Lesion shape: {stats['lesion_shape']}")
                print(f"  Disconn shape: {stats['disconn_shape']}")
                print(f"  Match: {'✓ Good' if stats['shapes_match'] else '✗ Mismatch'}")
                print(f"  Value range: [{stats['value_range'][0]:.4f}, {stats['value_range'][1]:.4f}]")
        
        return stats


# =============================================================================
# CLI Interface
# =============================================================================
def command_line_options() -> Tuple[Tuple[str, str, str], Tuple[int, int, int], bool, bool]:
    """
    Parse command line arguments.
    
    Returns
    -------
    tuple
        (paths, batch_params, download_only, verify_only)
        - paths: (lesionpath, outputpath, priorspath)
        - batch_params: (batch_size, n_proc, save_interval)
        - download_only: bool
        - verify_only: bool
    """
    parser = argparse.ArgumentParser(
        description="Generate disconnectomes from lesion masks using Functionnectome.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all lesions
  python disconnectomes.py --lesionpath ./lesions --outputpath ./disconnectomes --priorspath ./priors.h5
  
  # Process with specific batch size and parallelism
  python disconnectomes.py --lesionpath ./lesions --outputpath ./disconnectomes --priorspath ./priors.h5 --batch_size 200 --n_proc 8
  
  # Download priors first
  python disconnectomes.py --download_priors --outputpath ./priors_dir
  
  # Verify existing results
  python disconnectomes.py --lesionpath ./lesions --outputpath ./disconnectomes --priorspath ./priors.h5 --verify
        """
    )
    
    # Path arguments
    parser.add_argument("--lesionpath", type=str, default="", 
                        help="Path to lesion mask nii files")
    parser.add_argument("--outputpath", type=str, default="", 
                        help="Path to save disconnectomes")
    parser.add_argument("--priorspath", type=str, default="", 
                        help="Path to HDF5 priors file")
    
    # Processing parameters
    parser.add_argument("--batch_size", type=int, default=100, 
                        help="Number of files to process per batch (default: 100)")
    parser.add_argument("--n_proc", type=int, default=0, 
                        help="Number of parallel processes (0 = auto-detect, default: 0)")
    parser.add_argument("--save_interval", type=int, default=50, 
                        help="Save progress every N files (default: 50)")
    
    # Mode flags
    parser.add_argument("--download_priors", action="store_true", 
                        help="Download priors to outputpath and exit")
    parser.add_argument("--verify", action="store_true", 
                        help="Only verify existing results")
    
    args = parser.parse_args()
    
    paths = (args.lesionpath, args.outputpath, args.priorspath)
    batch_params = (args.batch_size, args.n_proc if args.n_proc > 0 else None, args.save_interval)
    
    return (paths, batch_params, args.download_priors, args.verify)


def run(parameters: Tuple) -> None:
    """
    Main execution function for disconnectome generation.
    
    Parameters
    ----------
    parameters : tuple
        Output from command_line_options():
        (paths, batch_params, download_only, verify_only)
    """
    paths, batch_params, download_only, verify_only = parameters
    lesionpath, outputpath, priorspath = paths
    batch_size, n_proc, save_interval = batch_params
    
    # Set default output path if not provided
    if not outputpath:
        outputpath = os.path.join(os.getcwd(), "disconnectomes")
        print(f"Using default output path: {outputpath}")
    
    if not os.path.exists(outputpath):
        os.makedirs(outputpath)
    
    # Download priors mode
    if download_only:
        print(f"Downloading priors to: {outputpath}")
        priors_path = DisconnectomeGenerator.download_priors(outputpath)
        print(f"Priors saved at: {priors_path}")
        print(f"File exists: {os.path.exists(priors_path)}")
        return
    
    # Validate required paths for processing
    if not lesionpath:
        raise ValueError("Please provide --lesionpath for lesion masks")
    if not priorspath:
        raise ValueError("Please provide --priorspath for HDF5 priors file")
    
    # Create generator
    generator = DisconnectomeGenerator(
        lesions_dir=lesionpath,
        output_dir=outputpath,
        priors_h5=priorspath
    )
    
    # Print configuration
    print("=" * 50)
    print("Disconnectome Generation Configuration")
    print("=" * 50)
    print(f"Lesions path: {lesionpath}")
    print(f"Output path: {outputpath}")
    print(f"Priors path: {priorspath}")
    print(f"Priors exists: {os.path.exists(priorspath)}")
    print(f"Batch size: {batch_size}")
    print(f"N_PROC: {n_proc if n_proc else 'auto-detect'}")
    print(f"Save interval: {save_interval}")
    print("=" * 50)
    
    # Verify only mode
    if verify_only:
        generator.verify_results(verbose=True)
        return
    
    # Process all lesions
    total_lesions = len(generator.get_lesion_files())
    pending = len(generator.get_pending_files())
    print(f"\nTotal lesions: {total_lesions}")
    print(f"Pending: {pending}")
    
    if pending == 0:
        print("All files already processed!")
        generator.verify_results(verbose=True)
        return
    
    # Run processing
    generator.process_all(
        batch_size=batch_size,
        n_proc=n_proc,
        save_interval=save_interval,
        verbose=True
    )
    
    # Final verification
    print("\n" + "=" * 50)
    print("Final Verification")
    print("=" * 50)
    generator.verify_results(verbose=True)


if __name__ == "__main__":
    parameters = command_line_options()
    run(parameters)