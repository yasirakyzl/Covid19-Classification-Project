import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from sklearn.feature_selection import SelectKBest, chi2
import matplotlib

# Headless environment
matplotlib.use('Agg')

logger = logging.getLogger(__name__)

def select_features(X: pd.DataFrame, y: pd.Series, k: int) -> tuple:
    """
    SelectKBest ve chi2 algoritmasını kullanarak en iyi `k` özelliği seçer.
    
    Args:
        X (pd.DataFrame): Özellik veri matrisi.
        y (pd.Series): Hedef değişken.
        k (int): Seçilecek özellik sayısı.
        
    Returns:
        tuple: (X_selected_df, selected_feature_names, selector_object)
    """
    logger.info(f"Feature selection başlıyor... Hedeflenen feature sayısı: {k}")
    
    # Tüm feature'ların isimlerini al
    feature_names = X.columns.tolist()
    
    # SelectKBest'i tanımla ve fit et
    selector = SelectKBest(score_func=chi2, k=k)
    X_selected = selector.fit_transform(X, y)
    
    # Seçilen özelliklerin isimlerini bul
    mask = selector.get_support()
    selected_features = [feature_names[i] for i in range(len(feature_names)) if mask[i]]
    
    logger.info(f"Seçilen özellikler: {selected_features}")
    
    # Yeni DataFrame oluştur
    X_selected_df = pd.DataFrame(X_selected, columns=selected_features)
    
    return X_selected_df, selected_features, selector

def plot_feature_importance(selector, feature_names: list, save_path: str) -> None:
    """
    Chi2 skorlarını bar chart olarak çizer ve diske kaydeder.
    
    Args:
        selector: Eğitilmiş SelectKBest nesnesi.
        feature_names (list): Orijinal özellik isimleri.
        save_path (str): Grafiğin kaydedileceği konum.
    """
    logger.info("Feature importance (chi2 scores) grafiği çiziliyor...")
    
    # Skorları ve isimleri al
    scores = selector.scores_
    
    # Skorlarla isimleri eşleştirip DataFrame yap
    importance_df = pd.DataFrame({"Feature": feature_names, "Score": scores})
    importance_df = importance_df.sort_values(by="Score", ascending=False)
    
    plt.figure(figsize=(12, 8))
    sns.barplot(x="Score", y="Feature", data=importance_df, palette="viridis")
    plt.title("Özellik Önem Skorları (Chi-Square)")
    plt.xlabel("Chi2 Skoru")
    plt.ylabel("Özellikler")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    
    logger.info(f"Feature importance grafiği kaydedildi: {save_path}")
