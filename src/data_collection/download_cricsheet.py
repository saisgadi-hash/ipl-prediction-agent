"""
Download match data from Cricsheet.org.

Cricsheet provides free, ball-by-ball data for cricket matches in JSON format.
This script downloads IPL, international T20, and other T20 league data.

HOW TO RUN:
    python -m src.data_collection.download_cricsheet

WHAT IT DOES:
    1. Downloads ZIP files containing match data from Cricsheet
    2. Extracts the JSON files into data/raw/ folders
    3. Prints a summary of what was downloaded

BEGINNER NOTES:
    - requests.get(url) sends an HTTP GET request (like visiting a URL in your browser)
    - zipfile.ZipFile opens a .zip file so we can extract its contents
    - os.makedirs creates folders, exist_ok=True means "don't error if it already exists"
"""

import os
import sys
import zipfile
import requests
from tqdm import tqdm

# Add project root to path so we can import config
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from config import RAW_DATA_DIR, DATA_SOURCES


def download_file(url: str, save_path: str) -> bool:
    """
    Download a file from a URL and save it locally.

    Args:
        url: The web address to download from
        save_path: Where to save the file on your computer

    Returns:
        True if download was successful, False otherwise

    BEGINNER NOTE:
        - response.status_code == 200 means "OK, the server sent us the file"
        - We write in binary mode ("wb") because ZIP files are binary, not text
        - tqdm gives us a nice progress bar while downloading
    """
    try:
        print(f"  Downloading from: {url}")
        response = requests.get(url, stream=True)

        if response.status_code != 200:
            print(f"  ERROR: Server returned status {response.status_code}")
            return False

        # Get the file size for the progress bar
        total_size = int(response.headers.get("content-length", 0))
        block_size = 8192  # Download in 8KB chunks

        # Save the file with a progress bar
        with open(save_path, "wb") as f:
            with tqdm(total=total_size, unit="B", unit_scale=True, desc="  Progress") as pbar:
                for chunk in response.iter_content(chunk_size=block_size):
                    f.write(chunk)
                    pbar.update(len(chunk))

        file_size_mb = os.path.getsize(save_path) / (1024 * 1024)
        print(f"  Saved: {save_path} ({file_size_mb:.1f} MB)")
        return True

    except requests.exceptions.ConnectionError:
        print(f"  ERROR: Could not connect to {url}. Check your internet connection.")
        return False
    except Exception as e:
        print(f"  ERROR: {e}")
        return False


def extract_zip(zip_path: str, extract_to: str) -> int:
    """
    Extract a ZIP file to a folder.

    Returns:
        Number of files extracted

    BEGINNER NOTE:
        - zipfile.ZipFile opens the archive (like double-clicking a .zip on your computer)
        - extractall() pulls out all the files inside it
    """
    os.makedirs(extract_to, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        file_list = zip_ref.namelist()
        zip_ref.extractall(extract_to)

    json_files = [f for f in file_list if f.endswith(".json")]
    print(f"  Extracted {len(json_files)} match files to {extract_to}")
    return len(json_files)


def download_ipl_data():
    """Download all IPL match data from Cricsheet."""
    print("\n" + "=" * 60)
    print("STEP 1: Downloading IPL Match Data")
    print("=" * 60)

    cricsheet = DATA_SOURCES.get("cricsheet", {})
    url = cricsheet.get("ipl_url", "https://cricsheet.org/downloads/ipl_json.zip")

    zip_path = str(RAW_DATA_DIR / "ipl_json.zip")
    extract_path = str(RAW_DATA_DIR / "ipl_matches")

    if download_file(url, zip_path):
        count = extract_zip(zip_path, extract_path)
        print(f"\n  SUCCESS: {count} IPL matches ready for analysis!")
        return count
    return 0


def download_t20i_data():
    """Download international T20 match data."""
    print("\n" + "=" * 60)
    print("STEP 2: Downloading International T20 Data")
    print("=" * 60)

    cricsheet = DATA_SOURCES.get("cricsheet", {})
    url = cricsheet.get("t20i_url", "https://cricsheet.org/downloads/t20s_json.zip")

    zip_path = str(RAW_DATA_DIR / "t20i_json.zip")
    extract_path = str(RAW_DATA_DIR / "t20i_matches")

    if download_file(url, zip_path):
        count = extract_zip(zip_path, extract_path)
        print(f"\n  SUCCESS: {count} T20I matches ready!")
        return count
    return 0


def download_other_leagues():
    """Download data from other T20 leagues (BBL, PSL, CPL, SA20, The Hundred)."""
    print("\n" + "=" * 60)
    print("STEP 3: Downloading Other T20 League Data")
    print("=" * 60)

    cricsheet = DATA_SOURCES.get("cricsheet", {})
    leagues = {
        "BBL (Big Bash League)": cricsheet.get("bbl_url", "https://cricsheet.org/downloads/bbl_json.zip"),
        "PSL (Pakistan Super League)": cricsheet.get("psl_url", "https://cricsheet.org/downloads/psl_json.zip"),
        "CPL (Caribbean Premier League)": cricsheet.get("cpl_url", "https://cricsheet.org/downloads/cpl_json.zip"),
        "SA20 (South Africa)": cricsheet.get("sa20_url", "https://cricsheet.org/downloads/sa20_json.zip"),
        "The Hundred": cricsheet.get("hundred_url", "https://cricsheet.org/downloads/hundred_json.zip"),
    }

    total = 0
    for league_name, url in leagues.items():
        print(f"\n--- {league_name} ---")
        short_name = league_name.split("(")[0].strip().lower().replace(" ", "_")
        zip_path = str(RAW_DATA_DIR / f"{short_name}_json.zip")
        extract_path = str(RAW_DATA_DIR / "other_leagues" / short_name)

        if download_file(url, zip_path):
            count = extract_zip(zip_path, extract_path)
            total += count

    print(f"\n  TOTAL: {total} matches from other T20 leagues")
    return total


def download_all():
    """
    Download ALL cricket data from Cricsheet.

    This is the main function you run. It downloads:
    1. All IPL matches (2008 - present)
    2. All international T20 matches
    3. Other T20 league matches (BBL, PSL, CPL, SA20, The Hundred)
    """
    print("\n" + "#" * 60)
    print("#  IPL PREDICTION AGENT - DATA DOWNLOAD")
    print("#  Downloading cricket data from Cricsheet.org")
    print("#" * 60)

    ipl_count = download_ipl_data()
    t20i_count = download_t20i_data()
    league_count = download_other_leagues()

    print("\n" + "=" * 60)
    print("DOWNLOAD SUMMARY")
    print("=" * 60)
    print(f"  IPL matches:              {ipl_count}")
    print(f"  International T20 matches: {t20i_count}")
    print(f"  Other T20 league matches:  {league_count}")
    print(f"  TOTAL:                     {ipl_count + t20i_count + league_count}")
    print("\nAll data saved to: data/raw/")
    print("Next step: Run  python -m src.data_collection.parse_matches")


if __name__ == "__main__":
    download_all()
