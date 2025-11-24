# data_prep.py
import os
import gzip
import requests
import pandas as pd
import numpy as np
import colorcet as cc
from pathlib import Path
import GEOparse
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

def download_url(url, outfile):
    """Download a file with a progress print."""
    print(f"   Downloading {outfile.name} from {url}...")
    response = requests.get(url, stream=True)
    response.raise_for_status()
    with open(outfile, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    print(f"   Saved to {outfile}")

def load_methylation_data(data_file_path, meta_data_file):
    """Load and align methylation data with metadata."""
    print("   Loading metadata...")
    reference_meta = pd.read_excel(meta_data_file)
    
    print("   Reading genomic matrix (this may take a moment)...")
    with gzip.open(data_file_path, 'rt') as f:
        header = f.readline().strip().replace('"', '').split('\t')
        ncol = len(header)

    cols_keep = [header[0]] + header[1:ncol:2]
    dtype_dict = {col: np.float32 for col in cols_keep[1:]}
    dtype_dict[cols_keep[0]] = str
    
    data_df = pd.read_csv(
        data_file_path,
        usecols=cols_keep,
        comment="!",
        sep='\t',
        index_col="ID_REF",
        dtype=dtype_dict
    )
    data_df = data_df.T

    # Match sample names
    reference_meta.set_index('ID', inplace=True)
    reference_meta = reference_meta.loc[data_df.index]
    reference_meta.reset_index(inplace=True, names='ID')

    return data_df, reference_meta

def main():
    print("=== Starting Data Preparation ===")
    
    # 1. Setup Directories
    data_dir = Path('./data/raw')
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # 2. handle Metadata Excel File
    # Check if it exists locally (uploaded to repo), otherwise download
    meta_filename = "41467_2020_20603_MOESM4_ESM.xlsx"
    if os.path.exists(meta_filename):
        print(f"1. Found metadata file locally: {meta_filename}")
    else:
        print(f"1. Metadata file not found locally. Downloading...")
        url_meta = "https://huggingface.co/datasets/bemert/GSE140686_GPL13534/resolve/main/41467_2020_20603_MOESM4_ESM.xlsx"
        download_url(url_meta, Path(meta_filename))

    # 3. Handle GEO Data
    print("2. Checking for raw GEO data...")
    geo_file = data_dir / "GSE140686_GPL13534_matrix_processed.txt.gz"
    
    if not geo_file.exists():
        print("   Downloading GEO series info...")
        # We use GEOparse to get the URL, then manually download to control path
        gse = GEOparse.get_GEO("GSE140686", destdir="./", how="quick")
        url_geo = gse.metadata['supplementary_file'][0]
        download_url(url_geo, geo_file)
    else:
        print(f"   Found existing GEO file at {geo_file}")

    # 4. Process Data
    print("3. Processing DataFrames...")
    df, meta = load_methylation_data(geo_file, meta_filename)
    
    # Filter Missing Values (Probes)
    print("   Filtering missing values (Probes)...")
    cpg_nan_counts = df.isna().sum()
    df = df.loc[:, cpg_nan_counts == 0]
    print(f"   Final shape: {df.shape[0]} samples, {df.shape[1]} probes")

    # 5. Feature Engineering (Diagnosis & Colors)
    print("4. Mapping Diagnoses and Colors...")
    dx_map = [
        ('Rhabdomyosarcoma (alveolar)', 'RMS-A', 'Skeletal muscle tumors'),
        ('Rhabdomyosarcoma (embryonal)', 'RMS-E', 'Skeletal muscle tumors'),
        ('Embryonal rhabdomyosarcoma', 'RMS-E', 'Skeletal muscle tumors'),
        ('Rhabdomyosarcoma (spindle cell)', 'RMS-S', 'Skeletal muscle tumors'),
        ('Synovial sarcoma', 'SS', 'Tumors of uncertain differentiation'),
        ('Myxoid liposarcoma', 'MLPS', 'Adipocytic tumors'),
        ('Small blue round cell tumour with BCOR alteration', 'BCOR', 'Undifferentiated small round cell tumors'),
        ('Small blue round cell tumour with CIC alteration', 'CIC', 'Undifferentiated small round cell tumors'),
        ('Ewing´s sarcoma', 'EWS', 'Undifferentiated small round cell tumors'),
        ('Fibrous dysplasia', 'FD', 'Other mesenchymal tumors of bone'),
        ('Chordoma', 'CHD', 'Notochordal tumors'),
        ('Osteosarcoma (high-grade)', 'OS-HG', 'Osteogenic tumors'),
        ('Undifferentiated pleomorphic sarcoma', 'UPS', 'Undifferentiated sarcomas'),
        ('Chondrosarcoma', 'CHS', 'Chondrogenic tumors'),
        ('Chondrosarcoma (mesenchymal)', 'MCHS', 'Chondrogenic tumors'),
        ('Extraskeletal myxoid chondrosarcoma', 'EMC', 'Tumors of uncertain differentiation'),
        ('Desmoplastic small round cell tumour', 'DSRCT', 'Tumors of uncertain differentiation'),
        ('Alveolar soft part sarcoma', 'ASPS', 'Tumors of uncertain differentiation'),
        ('Primitive neuroectodermal tumour', 'EWS', 'Undifferentiated small round cell tumors'),
        ('Solitary fibrous tumour', 'SFT', 'Fibroblastic/myofibroblastic tumors'),
        ('Malignant peripheral nerve sheath tumour', 'MPNST', 'Peripheral nerve sheath tumors'),
        ('Angiosarcoma', 'AS', 'Vascular tumors'),
        ('Gastroinstestinal stromal tumour', 'GIST', 'Gastrointestinal stromal tumor'),
        ('Undifferentiated sarcoma', 'US', 'Undifferentiated sarcomas'),
        ('Control (reactive tissue)', 'CTRL', 'Non-neoplastic'),
        ('Chordoma (de-differentiated)', 'DDCHD', 'Notochordal tumors'),
        ('Lipoma', 'LP', 'Adipocytic tumors'),
        ('Dermatofibrosarcoma protuberans', 'DFSP', 'Fibroblastic/myofibroblastic tumors'),
        ('Giant cell tumour of bone', 'GCTB', 'Osteoclastic giant cell rich tumors'),
        ('Sarcoma not otherwise specified', 'SARC-NOS', 'Undifferentiated sarcomas'),
        ('Osteoblastoma', 'OBL', 'Osteogenic tumors'),
        ('Desmoid-type fibromatosis', 'DES', 'Fibroblastic/myofibroblastic tumors'),
        ('Low-grade fibromyxoid sarcoma', 'LGFMS', 'Fibroblastic/myofibroblastic tumors'),
        ('Leiomyosarcoma', 'LMS', 'Smooth muscle tumors'),
        ('Malignant tumour not otherwise specified', 'MAL-NOS', 'Undifferentiated sarcomas'),
        ('Neurofibroma', 'NF', 'Peripheral nerve sheath tumors'),
        ('Neurofibroma (plexiform)', 'NF-P', 'Peripheral nerve sheath tumors'),
        ('Schwannoma', 'SCHW', 'Peripheral nerve sheath tumors'),
        ('Infantile fibrosarcoma', 'IFS', 'Fibroblastic/myofibroblastic tumors'),
        ('Kaposi sarcoma', 'KS', 'Vascular tumors'),
        ('Melanoma', 'MEL', 'Melanocytic tumors'),
        ('Sclerosing epithelioid sarcoma', 'SEF', 'Fibroblastic/myofibroblastic tumors'),
        ('Myxofibrosarcoma', 'MFS', 'Fibroblastic/myofibroblastic tumors'),
        ('Pleomorphic liposarcoma', 'PLPS', 'Adipocytic tumors'),
        ('Dedifferentiated liposarcoma', 'DDLPS', 'Adipocytic tumors'),
    ]

    df_diagnosis = pd.DataFrame(dx_map, columns=['Diagnosis', 'Dx', 'WHO_differentiation'])
    
    # Create colors
    unique_dx = df_diagnosis['Dx'].unique()
    palette = list(cc.glasbey[:len(unique_dx)])
    dx_to_color = dict(zip(unique_dx, palette))
    df_diagnosis['Color_dx'] = df_diagnosis['Dx'].map(dx_to_color)

    unique_WHO = df_diagnosis['WHO_differentiation'].unique()
    palette_WHO = list(cc.glasbey[:len(unique_WHO)])
    WHO_to_color = dict(zip(unique_WHO, palette_WHO))
    df_diagnosis['Color_WHO'] = df_diagnosis['WHO_differentiation'].map(WHO_to_color)

    # Merge
    meta_complete = pd.merge(meta, df_diagnosis, how='left', on='Diagnosis')

    # 6. Save to Disk
    print("5. Saving to Parquet...")
    df.to_parquet('methylation_data.parquet')
    meta_complete.to_parquet('metadata.parquet')
    
    print("=== Success! Data preparation complete. ===")

if __name__ == "__main__":
    main()