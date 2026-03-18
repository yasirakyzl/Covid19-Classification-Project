import logging
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from pathlib import Path
import matplotlib

# Headless mode
matplotlib.use('Agg')

logger = logging.getLogger(__name__)

def plot_confusion_matrix(y_test, y_pred, save_path: str) -> None:
    """
    Seaborn heatmap ile confusion matrix çizer.
    
    Args:
        y_test: Gerçek değerler.
        y_pred: Tahmin edilen değerler.
        save_path: Grafiğin kaydedileceği dosya yolu.
    """
    logger.info("Confusion matrix grafiği oluşturuluyor...")
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title("Confusion Matrix")
    plt.xlabel("Tahmin Edilen Sınıf")
    plt.ylabel("Gerçek Sınıf")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    logger.info(f"Confusion matrix kaydedildi: {save_path}")

def plot_roc_curve(model, X_test, y_test, save_path: str) -> None:
    """
    ROC Eğrisini (Receiver Operating Characteristic) çizer.
    Eğer y_test içerisinde (1, 2) var ise {2:1, 1:0} mapping'i yapar.
    
    Args:
        model: Eğitilmiş sklearn modeli.
        X_test: Test özellikleri.
        y_test: Test etiketleri.
        save_path: Grafiğin kaydedileceği dosya yolu.
    """
    logger.info("ROC Curve grafiği oluşturuluyor...")
    
    # Eğer y_test hala (1, 2) formatındaysa maple, değilse (1, 0) olduğu gibi kullan
    y_true = y_test.copy()
    if set(y_true.unique()).issubset({1, 2}):
        logger.info("y_test değerleri (1, 2) formatında tespit edildi, (0, 1) formatına mapleniyor {2:1, 1:0}")
        y_true = y_true.map({2: 1, 1: 0})
        
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    
    logger.info(f"ROC curve kaydedildi: {save_path}")

def evaluate_model(model, X_test, y_test, figures_dir: str) -> dict:
    """
    Modeli değerlendirir (Accuracy, CLS Report, Confusion Matrix, ROC).
    
    Args:
        model: Eğitilmiş model.
        X_test: Test özellikleri.
        y_test: Test etiketleri.
        figures_dir: Grafiklerin kaydedileceği dizin.
        
    Returns:
        dict: Değerlendirme metrikleri (accuracy, classification_report)
    """
    logger.info("Model test seti üzerinde değerlendiriliyor...")
    
    # Klasörü oluştur
    figures_path = Path(figures_dir)
    figures_path.mkdir(parents=True, exist_ok=True)
    
    y_pred = model.predict(X_test)
    
    # Metrikleri hesapla
    acc_score = accuracy_score(y_test, y_pred)
    cls_report = classification_report(y_test, y_pred)
    
    logger.info(f"Accuracy Score: {acc_score:.4f}")
    logger.info(f"Classification Report:\n{cls_report}")
    
    # Grafikleri çiz ve kaydet
    cm_path = str(figures_path / "07_confusion_matrix.png")
    roc_path = str(figures_path / "08_roc_curve.png")
    
    plot_confusion_matrix(y_test, y_pred, cm_path)
    plot_roc_curve(model, X_test, y_test, roc_path)
    
    return {
        "accuracy": acc_score,
        "classification_report": cls_report
    }
