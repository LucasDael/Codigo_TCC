from __future__ import annotations

import zipfile
from pathlib import Path

import pandas as pd

from config import ZIP_PATH


def read_csv_from_zip(zf: zipfile.ZipFile, filename: str) -> pd.DataFrame:
    """Le CSV do ZIP sem extrair todos os arquivos."""
    with zf.open(filename) as f:
        return pd.read_csv(f, low_memory=False)


def read_input_csv(filename: str) -> pd.DataFrame:
    """Le do ZIP quando existir, ou cai para o CSV no diretorio atual."""
    zip_path = ZIP_PATH
    if zip_path.exists():
        try:
            with zipfile.ZipFile(zip_path) as zf:
                if filename in zf.namelist():
                    return read_csv_from_zip(zf, filename)
        except zipfile.BadZipFile:
            pass

    local_path = Path(filename)
    if local_path.exists():
        return pd.read_csv(local_path, low_memory=False)

    raise FileNotFoundError(f"Nao encontrei '{filename}' nem {zip_path.resolve()}")


def safe_numeric(df: pd.DataFrame, columns: list[str]) -> None:
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")