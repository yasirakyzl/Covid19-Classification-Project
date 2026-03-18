import logging
from src import config
from src import data_loader
from src import preprocessor
from src import eda
from src import feature_selector
from src import trainer
from src import evaluator

def main():
    """Ana ML veri boru hattını (pipeline) çalıştırır."""
    logger = logging.getLogger("main")
    logger.info("COVID-19 ML Pipeline Başlatılıyor...")
    
    # 1. Config'den ayarları al, output dizinlerini yap (zaten config içinde mevcut)
    # 2. Veri Yükle
    logger.info("--- ADIM 1: Veri Yükleme ---")
    df = data_loader.load_data(config.DATA_PATH)
    data_loader.get_overview(df)
    
    # 3. Temizle
    logger.info("--- ADIM 2: Veri Ön İşleme (Preprocessing) ---")
    df_clean = preprocessor.run_preprocessing(df)
    
    # 4. Grafikleri Kaydet
    logger.info("--- ADIM 3: Keşifçi Veri Analizi (EDA) ---")
    eda.run_eda(df_clean, config.OUTPUT_FIGURES_DIR)
    
    # 5. Özellik Seç
    logger.info("--- ADIM 4: Özellik Seçimi (Feature Selection) ---")
    # Hedef değişkene göre X ve y'yi ayır
    y = df_clean[config.TARGET_COLUMN]
    X = df_clean.drop(columns=[config.TARGET_COLUMN])
    
    X_selected, selected_features, selector = feature_selector.select_features(X, y, config.K_BEST_FEATURES)
    feature_selector.plot_feature_importance(selector, X.columns.tolist(), str(config.OUTPUT_FIGURES_DIR / "06_feature_importance.png"))
    
    # 6. Undersample -> Split -> Eğit -> Kaydet
    logger.info("--- ADIM 5: Model Eğitimi (Training) ---")
    X_balanced, y_balanced = trainer.undersample(X_selected, y, config.RANDOM_STATE)
    X_train, X_test, y_train, y_test = trainer.split_data(X_balanced, y_balanced, config.TEST_SIZE, config.RANDOM_STATE)
    
    model = trainer.train_logistic_regression(X_train, y_train, config.MODEL_MAX_ITER, config.RANDOM_STATE)
    
    model_path = config.OUTPUT_MODELS_DIR / "logistic_regression_covid.pkl"
    trainer.save_model(model, str(model_path))
    
    # 7. Değerlendir
    logger.info("--- ADIM 6: Değerlendirme (Evaluation) ---")
    metrics = evaluator.evaluate_model(model, X_test, y_test, config.OUTPUT_FIGURES_DIR)
    
    logger.info("COVID-19 ML Pipeline Başarıyla Tamamlandı!")

if __name__ == "__main__":
    main()
