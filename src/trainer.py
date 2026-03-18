import pandas as pd
import logging
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import joblib

logger = logging.getLogger(__name__)

def undersample(X: pd.DataFrame, y: pd.Series, random_state: int) -> tuple:
    """
    Minority class = 1 (death). Majority class'tan aynı sayıda sample alır.
    
    Args:
        X (pd.DataFrame): Özellikler.
        y (pd.Series): Hedef değişken.
        random_state (int): Rastgelelik sabiti.
        
    Returns:
        tuple: (X_balanced, y_balanced)
    """
    logger.info("Undersampling işlemi başlatılıyor...")
    
    # DataFrame ve target'ı birleştir
    df = X.copy()
    df["DEATH"] = y
    
    # Sınıfları ayır (1: Minority - Öldü, 0: Majority - Yaşadı)
    # DEATH = 1 corresponds to death according to original notebook when mapped as 1:1, 2:0
    minority_class = df[df["DEATH"] == 1]
    majority_class = df[df["DEATH"] == 0]
    
    logger.info(f"Sınıf dağılımı (Öncesi): Yaşıyor (0): {len(majority_class)}, Öldü (1): {len(minority_class)}")
    
    # Majority class'tan minority class kadar sample al
    majority_downsampled = majority_class.sample(n=len(minority_class), random_state=random_state)
    
    # Birleştir ve shuffle et
    balanced_df = pd.concat([minority_class, majority_downsampled]).sample(frac=1, random_state=random_state).reset_index(drop=True)
    
    logger.info(f"Sınıf dağılımı (Sonrası): Toplam {len(balanced_df)} satır.")
    
    # X ve y olarak ayır
    y_balanced = balanced_df["DEATH"]
    X_balanced = balanced_df.drop("DEATH", axis=1)
    
    return X_balanced, y_balanced

def split_data(X: pd.DataFrame, y: pd.Series, test_size: float, random_state: int) -> tuple:
    """
    Veriyi train ve test olarak ayırır.
    
    Args:
        X (pd.DataFrame): Özellikler.
        y (pd.Series): Hedef değişken.
        test_size (float): Test setinin oranı.
        random_state (int): Rastgelelik sabiti.
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    logger.info(f"Veri %{test_size*100} test olacak şekilde bölünüyor...")
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

def train_logistic_regression(X_train: pd.DataFrame, y_train: pd.Series, max_iter: int, random_state: int):
    """
    Logistic Regression modeli eğitir.
    
    Args:
        X_train (pd.DataFrame): Eğitim özellikleri.
        y_train (pd.Series): Eğitim hedef değişkeni.
        max_iter (int): Maksimum itersayon sayısı.
        random_state (int): Rastgelelik sabiti.
        
    Returns:
        model: Eğitilmiş sklearn LogisticRegression nesnesi.
    """
    logger.info("Logistic Regression modeli eğitiliyor...")
    model = LogisticRegression(max_iter=max_iter, random_state=random_state)
    model.fit(X_train, y_train)
    logger.info("Model eğitimi tamamlandı.")
    return model

def save_model(model, path: str) -> None:
    """
    Modeli .pkl uzantılı dosya olarak kaydeder.
    
    Args:
        model: Eğitilmiş sklearn modeli.
        path (str): Kayıt dizini ve dosya adı.
    """
    joblib.dump(model, path)
    logger.info(f"Model kaydedildi: {path}")
