# process_data.py
import gzip
import pandas as pd
import numpy as np
import colorcet as cc
from pathlib import Path
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# ================= CONFIGURATION =================
INPUT_DIR = Path('./data/raw')
META_FILE = INPUT_DIR / "41467_2020_20603_MOESM4_ESM.xlsx"
GEO_FILE = INPUT_DIR / "GSE140686_GPL13534_matrix_processed.txt.gz"

OUTPUT_DATA = "methylation_data.parquet"
OUTPUT_META = "metadata.parquet"
# =================================================

def load_and_align(data_path, meta_path):
    """Load methylation data and align with metadata."""
    print("   [LOADING] Metadata...")
    # Expecting the file downloaded by download_data.py
    if not meta_path.exists():
        raise FileNotFoundError(f"Could not find {meta_path}.")
    
    reference_meta = pd.read_excel(meta_path)
    
    print("   [LOADING] Methylation data (this is may take a couple minutes)...")
    if not data_path.exists():
        raise FileNotFoundError(f"Could not find {data_path}.")

    # Read header first to determine columns
    with gzip.open(data_path, 'rt') as f:
        header = f.readline().strip().replace('"', '').split('\t')
        ncol = len(header)

    # Logic to skip every other column (p-values) often found in GEO methylation files
    cols_keep = [header[0]] + header[1:ncol:2]
    dtype_dict = {col: np.float32 for col in cols_keep[1:]}
    dtype_dict[cols_keep[0]] = str
    
    data_df = pd.read_csv(
        data_path,
        usecols=cols_keep,
        comment="!",
        sep='\t',
        index_col="ID_REF",
        dtype=dtype_dict
    )
    data_df = data_df.T

    # Match sample names
    print("   Matching methylation data samples to metadata...")
    reference_meta.set_index('ID', inplace=True)
    #Drop original Colours column if exists
    if 'Colour' in reference_meta.columns:
        reference_meta.drop(columns=['Colour'], inplace=True)
    # Intersect indices
    common_ids = data_df.index.intersection(reference_meta.index)
    
    data_df = data_df.loc[common_ids]
    reference_meta = reference_meta.loc[common_ids]
    reference_meta.reset_index(inplace=True, names='ID')

    return data_df, reference_meta

def main():
    print("=== Starting Data Processing ===")

    # Load Data
    try:
        df, meta = load_and_align(GEO_FILE, META_FILE)
        print(f"   Data Dimensions: {df.shape[0]} samples x {df.shape[1]} probes")
        print(f"   First 5 rows of metadata:\n{meta.head()}\n")
    except Exception as e:
        print(f"Unable to load data. ERROR: {e}")
        return

    
    # Filter Missing Values
    print("   Removing probes with missing values...")
    cpg_nan_counts = df.isna().sum()
    print(f'Number of CpG probes with missing values: {np.sum(cpg_nan_counts > 0)}')
    cpg_nan_counts = df.isna().sum()
    df = df.loc[:, cpg_nan_counts == 0]
    print(f"   Final Dimensions: {df.shape[0]} samples x {df.shape[1]} probes")

    # Adding to metadata (Diagnosis & Colors)
    print("   Adding diagnosis codes and color palettes...")
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
    
    unique_dx = df_diagnosis['Dx'].unique()
    palette = list(cc.glasbey[:len(unique_dx)])
    dx_to_color = dict(zip(unique_dx, palette))
    df_diagnosis['Color_dx'] = df_diagnosis['Dx'].map(dx_to_color)

    unique_WHO = df_diagnosis['WHO_differentiation'].unique()
    palette_WHO = list(cc.glasbey[:len(unique_WHO)])
    WHO_to_color = dict(zip(unique_WHO, palette_WHO))
    df_diagnosis['Color_WHO'] = df_diagnosis['WHO_differentiation'].map(WHO_to_color)

    meta_complete = pd.merge(meta, df_diagnosis, how='left', on='Diagnosis')

    # 4. Save to Disk
    print("   [SAVING] Writing processed data files...")
    df.to_parquet(OUTPUT_DATA, engine="pyarrow", compression="zstd")
    meta_complete.to_parquet(OUTPUT_META, engine="pyarrow", compression="zstd")

    print(f"===  Data ready for App. ===")

if __name__ == "__main__":
    main()