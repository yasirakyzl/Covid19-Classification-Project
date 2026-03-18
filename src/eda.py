import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import logging
from pathlib import Path
import matplotlib
from . import config

# Eğer headless (UI sız) ortamdaysak:
matplotlib.use('Agg')

logger = logging.getLogger(__name__)

def plot_target_distribution(df: pd.DataFrame, save_path: str) -> None:
    """DEATH kolonunun countplot'unu çizer."""
    plt.figure(figsize=(6, 4))
    sns.countplot(x="DEATH", data=df, palette="viridis")
    plt.title("Target Olarak Seçilen DEATH Dağılımı")
    plt.xlabel("0: Yaşıyor, 1: Öldü")
    plt.ylabel("Kişi Sayısı")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    logger.info(f"plot_target_distribution kaydedildi: {save_path}")

def plot_age_distribution(df: pd.DataFrame, save_path: str) -> None:
    """AGE histogramını çizer."""
    plt.figure(figsize=(10, 5))
    sns.histplot(df["AGE"], bins=30, kde=True, color="skyblue")
    plt.title("Yaş (AGE) Dağılımı")
    plt.xlabel("Age")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    logger.info(f"plot_age_distribution kaydedildi: {save_path}")

def plot_sex_pregnancy(df: pd.DataFrame, save_path: str) -> None:
    """SEX-PREGNANT bar chart."""
    plt.figure(figsize=(8, 6))
    if "SEX" in df.columns and "PREGNANT" in df.columns:
        sns.countplot(data=df, x="SEX", hue="PREGNANT", palette="Set2")
        plt.title("Cinsiyet (SEX) ve Hamilelik (PREGNANT) Dağılımı")
    else:
        plt.title("SEX veya PREGNANT kolonu bulunamadı.")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    logger.info(f"plot_sex_pregnancy kaydedildi: {save_path}")

def plot_correlation_heatmap(df: pd.DataFrame, save_path: str) -> None:
    """Korelasyon heatmap."""
    plt.figure(figsize=(16, 12))
    corr = df.corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)
    plt.title("Feature Korelasyon Matrisi Heatmap")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    logger.info(f"plot_correlation_heatmap kaydedildi: {save_path}")

def plot_feature_death_rates(df: pd.DataFrame, binary_cols: list, save_path: str) -> None:
    """Her binary feature için ölüm oranını gösteren bar chart."""
    rates = {}
    for col in binary_cols:
        # Binary cols genelde 1 ve 0'dır, 1 olanların DEATH ortalaması -> ölüm oranı
        if col in df.columns and "DEATH" in df.columns:
            rate = df[df[col] == 1]["DEATH"].mean()
            rates[col] = rate
    
    if rates:
        plt.figure(figsize=(12, 6))
        rates_series = pd.Series(rates).sort_values(ascending=False)
        sns.barplot(x=rates_series.values, y=rates_series.index, palette="mako")
        plt.title("Binary Feature'lar için Ölüm Oranları (DEATH Rate)")
        plt.xlabel("Ölüm Oranı")
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        logger.info(f"plot_feature_death_rates kaydedildi: {save_path}")

def run_eda(df: pd.DataFrame, figures_dir: str) -> None:
    """Tüm EDA fonksiyonlarını çalıştırır."""
    logger.info("Keşifçi Veri Analizi (EDA) grafikleri çizilmeye başlanıyor...")
    
    figures_path = Path(figures_dir)
    figures_path.mkdir(parents=True, exist_ok=True)
    
    plot_target_distribution(df, str(figures_path / "01_target_distribution.png"))
    plot_age_distribution(df, str(figures_path / "02_age_distribution.png"))
    plot_sex_pregnancy(df, str(figures_path / "03_sex_pregnancy.png"))
    plot_correlation_heatmap(df, str(figures_path / "04_correlation_heatmap.png"))
    
    # Binary olan, numeric olmayan ve 1/0 değerleri alan kolonları seçeceğiz
    binary_cols = [col for col in df.columns if set(df[col].dropna().unique()).issubset({0, 1}) and col != "DEATH"]
    
    plot_feature_death_rates(df, binary_cols, str(figures_path / "05_feature_death_rates.png"))
    
    logger.info(f"EDA tamamlandı, çıktılar {figures_dir} dizinine kaydedildi.")
