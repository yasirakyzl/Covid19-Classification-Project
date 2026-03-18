import pandas as pd
import logging
from . import config

# Setup module-level logger
logger = logging.getLogger(__name__)

def load_data(filepath: str) -> pd.DataFrame:
    """
    CSV dosyasını yükler, veri setinin boyutlarını ve ilk 5 satırını loglar.

    Args:
        filepath (str): Ham CSV dosyasının dosya yolu.

    Returns:
        pd.DataFrame: Yüklenen veriyi içeren pandas DataFrame nesnesi.

    Raises:
        FileNotFoundError: Eğer belirtilen yoldaki dosya bulunamazsa.
    """
    logger.info(f"Yükleniyor: {filepath}")
    
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        logger.error(f"HATA: Veri dosyası bulunamadı -> {filepath}")
        raise FileNotFoundError(f"Veri dosyası belirtilen dizinde yok: {filepath} Lütfen 'data/raw/' klasörünü kontrol edin.")

    logger.info(f"Veri başarıyla yüklendi. Shape: {df.shape}")
    logger.info("İlk 5 satır:\n%s", df.head().to_string())
    
    return df


def get_overview(df: pd.DataFrame) -> None:
    """
    DataFrame'in genel dtypes bilgisini, eşsiz değer sayılarını, 
    DATE_DIED ve PNEUMONIA kolonlarının value_counts değerlerini loglar.

    Args:
        df (pd.DataFrame): İncelenecek veri seti.
    """
    logger.info("Veri Tipleri (dtypes):\n%s", df.dtypes.to_string())
    logger.info("Eşsiz değer sayıları (nunique):\n%s", df.nunique().to_string())
    
    if "DATE_DIED" in df.columns:
        logger.info("DATE_DIED (Ölüm Tarihi) dağılımı:\n%s", df["DATE_DIED"].value_counts().to_string())
    else:
        logger.info("DATE_DIED kolonu bulunamadı.")
        
    if "PNEUMONIA" in df.columns:
        logger.info("PNEUMONIA dağılımı:\n%s", df["PNEUMONIA"].value_counts().to_string())
    else:
        logger.info("PNEUMONIA kolonu bulunamadı.")
