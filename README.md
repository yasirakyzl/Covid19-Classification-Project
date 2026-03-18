# COVID-19 Death Prediction - Modular Machine Learning Pipeline

Bu proje, Meksika Hükümeti'nin yayınladığı 1 milyonu aşkın hastanın verilerini içeren COVID-19 veri seti üzerinden hastaların hayatta kalma (ölüm) durumunu tahmin eden makine öğrenimi modelinin **production-ready (üretime hazır)** ve **modüler** Python versiyonudur.

Önceden Jupyter Notebook üzerinde geliştirilmiş analiz ve modelleme kodları; parçalı, tekrar kullanılabilir ve test edilebilir yapılar halinde modüllere ayrıştırılmıştır. (Clean Code & Separation of Concerns prensipleri uygulanmıştır.)

## Veri Seti (Dataset) Bilgisi
* **Satır Sayısı**: ~1.048.575
* **Özellik (Feature) Sayısı**: 21
* **Kaynak**: Meksika Hükümeti (Kaggle üzerinde sunulmuştur)
* **Hedef Değişken (Target)**: `DEATH` (1: Öldü, 0: Yaşadı)

### Özellikler (Features) Tablosu

| Kolon Adı | Açıklama |
| --- | --- |
| **USMER** | Tıbbi birimin seviyesi (1, 2, 3) |
| **MEDICAL_UNIT** | Kurum tipi |
| **SEX** | Cinsiyet (1: Kadın, 2: Erkek) |
| **PATIENT_TYPE** | Hasta tipi (1: Eve gönderildi, 2: Hastaneye yatırıldı) |
| **DATE_DIED** | Ölüm tarihi (Eğer '9999-99-99' ise hasta ölmemiştir) |
| **INTUBED** | Entübe edilip edilmediği |
| **PNEUMONIA** | Zatürre durumu |
| **AGE** | Hastanın yaşı |
| **PREGNANT** | Hamilelik durumu |
| **DIABETES** | Diyabet rahatsızlığı |
| **COPD** | KOAH rahatsızlığı durumu |
| **ASTHMA** | Astım rahatsızlığı durumu |
| **INMSUPR** | İmmünsüpresyon durumu |
| **HIPERTENSION** | Hipertansiyon hastalığı durumu |
| **OTHER_DISEASE** | Diğer hastalıkların durumu |
| **CARDIOVASCULAR** | Kalp ve damar rahatsızlığı |
| **OBESITY** | Obezite durumu |
| **RENAL_CHRONIC** | Kronik böbrek hastalığı |
| **TOBACCO** | Tütün (sigara) kullanım durumu |
| **CLASIFFICATION_FINAL** | COVID test sonuç sınıflandırması |
| **ICU** | Yoğun bakım (Intensive Care Unit) yatış durumu |

*(Not: INTUBED ve ICU gibi eksik verisi çok olan özellikler analiz aşamasında veri sızıntısını önlemek için düşülmüştür.)*

## Proje Klasör Yapısı

```
covid19-ml-project/
│
├── data/
│   └── raw/                    # Ham veri klasörü (CovidData.csv buraya yerleştirilir)
│
├── outputs/
│   ├── figures/                # EDA grafikleri, ROC ve Confusion Matrix (PNG olarak kaydedilir)
│   └── models/                 # Eğitilen model .pkl formatında bu klasöre kaydedilir
│
├── src/
│   ├── __init__.py
│   ├── config.py               # Hyperparametreler, dizin ayarları ve sabitler
│   ├── data_loader.py          # Verinin pandas DataFrame'e yüklenmesi
│   ├── preprocessor.py         # Temizleme, dönüşüm ve eksik/hatalı verilerin manipülatif işlemi
│   ├── eda.py                  # Keşifçi veri analizi grafiklerini çizen modül
│   ├── feature_selector.py     # SelectKBest ile Chi-Square (chi2) skor tabanında özellik seçimi
│   ├── trainer.py              # Veri dengelemesi (Undersampling) ve model (Logistic Regression) eğitimi
│   └── evaluator.py            # Modeli test ederek metrikleri hesaplama
│
├── main.py                     # Tüm işlemleri (pipeline) baştan sona yöneten ana Python projesi
├── requirements.txt            # Kurulması gereken dış kütüphaneler listesi
└── README.md                   # Proje kullanım dokümantasyonu
```

## Kurulum ve Çalıştırma Yönergeleri

### 1. Kütüphanelerin Yüklenmesi
Sanal ortam (virtual environment) oluşturduktan veya genel ortama kütüphaneleri yüklemek için aşağıdaki komutu çalıştırın:
```bash
pip install -r requirements.txt
```

### 2. Ham Veri Dosyası (Data)
Dosya boyutu çok büyük olduğu için GitHub'a eklenmemiştir. `CovidData.csv` dosyasını `data/raw/` klasörü içerisine yerleştirin.

### 3. Pipeline'i Başlatma (Training)
Tüm süreci modüler halde çalıştırmak, veriyi ön işleme sokmak, grafik çıkartmak ve modeli eğitmek için ana script'i çalıştırın:
```bash
python main.py
```

## Model Sonuçları ve Başarısı
* **Algoritma**: Logistic Regression (max_iterations=1000)
* **Veri Dengeleme**: Undersampling yöntemi (%50 Yaşayan / %50 Ölen dengesi sağlandı)
* **Accuracy (Doğruluk)**: ~%90 civarında
* **Ek Metrikler**: Sensitivity, Specificity değerleri ROC Curve grafiklerinde (AUC ölçümü) oldukça yüksek gözlemlenmiştir.

---
*Bu proje gelişmiş bir Modüler Python Eğitimi uygulaması için yapay zeka ajanları tarafından Agentic Framework standartlarında geliştirilmiştir.*
