# =============================================================================
# Download Lesions Module
# =============================================================================
"""
Module for downloading lesion masks from the individualized_prescriptive_inference repository.

Usage:
    from download_lesions import LesionDownloader, run
    
    # As module
    downloader = LesionDownloader(output_dir="./lesions")
    lesion_files = downloader.download()
    
    # From command line
    python download_lesions.py --outputpath ./data --url https://example.com/lesions.zip
"""

import os
import glob
import zipfile
import subprocess
import argparse
from urllib.request import urlretrieve
from typing import Tuple, List, Optional


# Default URL for lesion masks
DEFAULT_LESIONS_URL = "https://github.com/high-dimensional/individualized_prescriptive_inference/raw/main/lesions.zip"


class LesionDownloader:
    """
    Downloads and extracts lesion mask files from a remote repository.
    
    Attributes
    ----------
    output_dir : str
        Directory where lesions will be extracted.
    url : str
        URL to the lesions zip file.
    zip_filename : str
        Name of the zip file.
    """
    
    def __init__(
        self,
        output_dir: str,
        url: str = DEFAULT_LESIONS_URL,
        zip_filename: str = "lesions.zip"
    ):
        """
        Initialize the LesionDownloader.
        
        Parameters
        ----------
        output_dir : str
            Directory where lesions will be extracted.
        url : str
            URL to the lesions zip file.
        zip_filename : str
            Name of the zip file.
        """
        self.output_dir = output_dir
        self.url = url
        self.zip_filename = zip_filename
        self.zip_path = os.path.join(output_dir, zip_filename)
        self.lesions_dir = os.path.join(output_dir, "lesions")
        
        os.makedirs(self.output_dir, exist_ok=True)
    
    def _download_with_urlretrieve(self) -> bool:
        """Download using urllib urlretrieve."""
        try:
            print(f"Downloading from: {self.url}")
            print(f"Saving to: {self.zip_path}")
            urlretrieve(self.url, self.zip_path)
            return True
        except Exception as e:
            print(f"urlretrieve failed: {e}")
            return False
    
    def _download_with_wget(self) -> bool:
        """Download using wget as fallback."""
        try:
            print("Trying wget as fallback...")
            subprocess.run(
                ["wget", "-q", "--show-progress", self.url, "-O", self.zip_path],
                check=True
            )
            return True
        except Exception as e:
            print(f"wget failed: {e}")
            return False
    
    def _download_with_curl(self) -> bool:
        """Download using curl as fallback."""
        try:
            print("Trying curl as fallback...")
            subprocess.run(
                ["curl", "-L", "-o", self.zip_path, self.url],
                check=True
            )
            return True
        except Exception as e:
            print(f"curl failed: {e}")
            return False
    
    def download_zip(self, force: bool = False) -> bool:
        """
        Download the lesions zip file.
        
        Parameters
        ----------
        force : bool
            If True, download even if zip file already exists.
            
        Returns
        -------
        bool
            True if download successful or file already exists.
        """
        if os.path.exists(self.zip_path) and not force:
            print(f"Zip file already exists: {self.zip_path}")
            return True
        
        # Try multiple download methods
        if self._download_with_urlretrieve():
            return True
        if self._download_with_wget():
            return True
        if self._download_with_curl():
            return True
        
        raise RuntimeError(f"Failed to download lesions from {self.url}")
    
    def _extract_with_zipfile(self) -> bool:
        """Extract using Python zipfile module."""
        try:
            print(f"Extracting to: {self.output_dir}")
            with zipfile.ZipFile(self.zip_path, "r") as zip_ref:
                zip_ref.extractall(self.output_dir)
            return True
        except Exception as e:
            print(f"zipfile extraction failed: {e}")
            return False
    
    def _extract_with_unzip(self) -> bool:
        """Extract using unzip command as fallback."""
        try:
            print("Trying unzip command as fallback...")
            subprocess.run(
                ["unzip", "-q", "-o", self.zip_path, "-d", self.output_dir],
                check=True
            )
            return True
        except Exception as e:
            print(f"unzip failed: {e}")
            return False
    
    def extract_zip(self) -> bool:
        """
        Extract the lesions zip file.
        
        Returns
        -------
        bool
            True if extraction successful.
        """
        if not os.path.exists(self.zip_path):
            raise FileNotFoundError(f"Zip file not found: {self.zip_path}")
        
        # Try multiple extraction methods
        if self._extract_with_zipfile():
            return True
        if self._extract_with_unzip():
            return True
        
        raise RuntimeError(f"Failed to extract {self.zip_path}")
    
    def get_lesion_files(self) -> List[str]:
        """Get sorted list of all lesion files."""
        return sorted(glob.glob(os.path.join(self.lesions_dir, "*.nii*")))
    
    def lesions_exist(self) -> bool:
        """Check if lesion files already exist."""
        return os.path.exists(self.lesions_dir) and len(self.get_lesion_files()) > 0
    
    def download(self, force: bool = False, keep_zip: bool = True) -> List[str]:
        """
        Download and extract lesion masks.
        
        Parameters
        ----------
        force : bool
            If True, download and extract even if files already exist.
        keep_zip : bool
            If True, keep the zip file after extraction.
            
        Returns
        -------
        List[str]
            List of paths to lesion files.
        """
        if self.lesions_exist() and not force:
            lesion_files = self.get_lesion_files()
            print(f"Lesions already exist: {len(lesion_files)} files found")
            return lesion_files
        
        # Download
        self.download_zip(force=force)
        
        # Extract
        self.extract_zip()
        
        # Clean up zip if requested
        if not keep_zip and os.path.exists(self.zip_path):
            os.remove(self.zip_path)
            print(f"Removed zip file: {self.zip_path}")
        
        lesion_files = self.get_lesion_files()
        print(f"Total lesions found: {len(lesion_files)}")
        
        return lesion_files
    
    def verify(self, verbose: bool = True) -> dict:
        """
        Verify the downloaded lesion files.
        
        Parameters
        ----------
        verbose : bool
            Print verification results.
            
        Returns
        -------
        dict
            Verification statistics.
        """
        lesion_files = self.get_lesion_files()
        
        stats = {
            "lesions_dir": self.lesions_dir,
            "lesions_dir_exists": os.path.exists(self.lesions_dir),
            "total_files": len(lesion_files),
            "zip_exists": os.path.exists(self.zip_path)
        }
        
        if lesion_files:
            # Check a sample file
            try:
                import nibabel as nib
                sample = nib.load(lesion_files[0])
                stats["sample_shape"] = sample.shape
                stats["sample_dtype"] = str(sample.get_data_dtype())
            except ImportError:
                stats["sample_shape"] = "nibabel not installed"
            except Exception as e:
                stats["sample_shape"] = f"Error: {e}"
        
        if verbose:
            print("=" * 50)
            print("Lesion Download Verification")
            print("=" * 50)
            print(f"Lesions directory: {stats['lesions_dir']}")
            print(f"Directory exists: {stats['lesions_dir_exists']}")
            print(f"Total files: {stats['total_files']}")
            print(f"Zip file exists: {stats['zip_exists']}")
            if "sample_shape" in stats:
                print(f"Sample shape: {stats['sample_shape']}")
            if "sample_dtype" in stats:
                print(f"Sample dtype: {stats['sample_dtype']}")
            print("=" * 50)
        
        return stats


# =============================================================================
# CLI Interface
# =============================================================================
def command_line_options() -> Tuple[Tuple[str, str], bool, bool, bool]:
    """
    Parse command line arguments.
    
    Returns
    -------
    tuple
        (paths, force, keep_zip, verify_only)
        - paths: (outputpath, url)
        - force: bool
        - keep_zip: bool
        - verify_only: bool
    """
    parser = argparse.ArgumentParser(
        description="Download lesion masks from remote repository.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download to default location
  python download_lesions.py --outputpath ./data
  
  # Download from custom URL
  python download_lesions.py --outputpath ./data --url https://example.com/lesions.zip
  
  # Force re-download
  python download_lesions.py --outputpath ./data --force
  
  # Download and remove zip file
  python download_lesions.py --outputpath ./data --remove_zip
  
  # Verify existing download
  python download_lesions.py --outputpath ./data --verify
        """
    )
    
    # Path arguments
    parser.add_argument("--outputpath", type=str, default="",
                        help="Path to save lesion files (default: current directory)")
    parser.add_argument("--url", type=str, default=DEFAULT_LESIONS_URL,
                        help="URL to lesions zip file")
    
    # Mode flags
    parser.add_argument("--force", action="store_true",
                        help="Force re-download even if files exist")
    parser.add_argument("--remove_zip", action="store_true",
                        help="Remove zip file after extraction")
    parser.add_argument("--verify", action="store_true",
                        help="Only verify existing download")
    
    args = parser.parse_args()
    
    paths = (args.outputpath, args.url)
    
    return (paths, args.force, not args.remove_zip, args.verify)


def run(parameters: Tuple) -> List[str]:
    """
    Main execution function for lesion download.
    
    Parameters
    ----------
    parameters : tuple
        Output from command_line_options():
        (paths, force, keep_zip, verify_only)
        
    Returns
    -------
    List[str]
        List of paths to lesion files.
    """
    paths, force, keep_zip, verify_only = parameters
    outputpath, url = paths
    
    # Set default output path if not provided
    if not outputpath:
        outputpath = os.getcwd()
        print(f"Using default output path: {outputpath}")
    
    if not os.path.exists(outputpath):
        os.makedirs(outputpath)
    
    # Create downloader
    downloader = LesionDownloader(
        output_dir=outputpath,
        url=url
    )
    
    # Print configuration
    print("=" * 50)
    print("Lesion Download Configuration")
    print("=" * 50)
    print(f"Output path: {outputpath}")
    print(f"Lesions dir: {downloader.lesions_dir}")
    print(f"URL: {url}")
    print(f"Force: {force}")
    print(f"Keep zip: {keep_zip}")
    print("=" * 50)
    
    # Verify only mode
    if verify_only:
        downloader.verify(verbose=True)
        return downloader.get_lesion_files()
    
    # Download and extract
    lesion_files = downloader.download(force=force, keep_zip=keep_zip)
    
    # Final verification
    print()
    downloader.verify(verbose=True)
    
    return lesion_files


if __name__ == "__main__":
    parameters = command_line_options()
    run(parameters)