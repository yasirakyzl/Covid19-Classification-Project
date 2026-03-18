import pandas as pd
import logging
from . import config

logger = logging.getLogger(__name__)

def remove_missing_values(df: pd.DataFrame, cols: list, missing_vals: list) -> pd.DataFrame:
    """Belirtilen kolonlarda missing_vals içeren satırları siler."""
    logger.info(f"Kayıp değerler {missing_vals} listedeki kolonlardan {cols} temizleniyor...")
    initial_shape = df.shape
    for col in cols:
        df = df[~df[col].isin(missing_vals)]
    logger.info(f"Temizleme sonrası boyut: {df.shape} (Düşen satır: {initial_shape[0] - df.shape[0]})")
    return df

def create_death_target(df: pd.DataFrame) -> pd.DataFrame:
    """DATE_DIED → DEATH binary kolonu oluşturur (9999-99-99=2, diğer=1)."""
    logger.info("DEATH hedef değişkeni oluşturuluyor...")
    df["DEATH"] = df["DATE_DIED"].apply(lambda x: 2 if x == "9999-99-99" else 1)
    return df

def fix_pregnant_column(df: pd.DataFrame) -> pd.DataFrame:
    """PREGNANT kolonunu erkekler için 2 ile doldurur (SEX==2 → PREGNANT=2)."""
    logger.info("PREGNANT kolonu erkek hastalar (SEX=2) için 2 (hayır) olarak dolduruluyor...")
    # SEX: 2 is male, PREGNANT: 2 is NO
    df.loc[df["SEX"] == 2, "PREGNANT"] = 2
    return df

def encode_binary_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tüm 1/2 içeren binary kolonları 1/0'a dönüştürür. 
    DEATH: 1→1, 2→0. Diğerleri: 1→1, 2→0.
    """
    logger.info("1/2 formatındaki özellikler 1/0 formatına dönüştürülüyor...")
    # Find columns that only have values from {1, 2, 97, 98, 99} or similar boolean logic
    # Also explicitly handling target and boolean variables.
    
    binary_mapping = {1: 1, 2: 0}
    
    # We apply this mapping to all feature columns that originally used 1=(yes), 2=(no)
    # Exclude columns that are not boolean (like AGE, MEDICAL_UNIT)
    for col in df.columns:
        unique_vals = set(df[col].unique())
        # If the column essentially contains 1 and 2 (ignoring missing values which were mostly dropped)
        if unique_vals.issubset({0, 1, 2, 97, 98, 99}):
            # We skip 'SEX' because it's categorical 1=(Female), 2=(Male), wait, notebook usually maps SEX too
            # Wait, the prompt specifically said: "Tüm 1/2 binary kolonları 1/0'a dönüştürür."
            if col not in config.COLS_TO_DROP_FINAL and col != 'AGE':
                df[col] = df[col].map(lambda x: binary_mapping.get(x, x))
    return df

def drop_columns(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """Gereksiz kolonları siler."""
    existing_cols = [c for c in cols if c in df.columns]
    logger.info(f"Kolonlar siliniyor: {existing_cols}")
    return df.drop(columns=existing_cols)

def run_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    """Tüm preprocessing adımlarını sırayla çalıştırır ve temiz df döner."""
    logger.info("Veri ön işleme (preprocessing) başlatılıyor...")
    
    # Adım 1: DEATH kolonunu oluştur
    df = create_death_target(df)
    
    # Adım 2: Missing değerleri düş (INTUBED, PREGNANT, ICU hariç)
    # COLS_TO_DROP_MISSING config içerisinde listeleniyor.
    df = remove_missing_values(df, config.COLS_TO_DROP_MISSING, config.MISSING_VALUES)
    
    # Adım 3: PREGNANT düzeltmesi (Erkekler -> 2)
    df = fix_pregnant_column(df)
    
    # Adım 4: Binary mapping (1/2 -> 1/0)
    df = encode_binary_columns(df)
    
    # Adım 5: Gereksiz kolonları düş
    df = drop_columns(df, config.COLS_TO_DROP_FINAL)
    
    # Sadece PREGNANT kadınlar için eğer 97/99 kaldıysa, onu da düşebiliriz veya dataset'teki gibi bırakırız.
    # Orijinal notebook'ta kadınlar için 97/98/99 olanlar genelde bırakılıyor veya düşülüyor, 
    # ama Prompt: "kadın hastalardaki 97/99 değerleri orijinal notebook'ta olduğu gibi olduğu gibi bırakılabilir" diyor.
    
    logger.info(f"Ön işleme tamamlandı. Son veri boyutu: {df.shape}")
    return df
