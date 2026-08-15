# deniz_saka_makine_-renmesi_-devi
# Müşteri Churn Tahmini

## Projenin Amacı
Bu projenin amacı, müşterilerin abonelikten ayrılıp ayrılmayacağını (churn) temel makine öğrenmesi yöntemleri kullanarak tahmin etmektir.
Projede müşteri yaşı, gelir, abonelik süresi, destek talebi sayısı, şehir ve üyelik tipi gibi değişkenler kullanılmıştır. Hedef değişken olan `churn` değeri:
* `0`: Müşteri ayrılmadı
* `1`: Müşteri ayrıldı
Veri seti Python kullanılarak 150 müşteri için oluşturulmuştur.

## Kullanılan Kütüphaneler
Projede aşağıdaki Python kütüphaneleri kullanılmıştır:
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Seaborn

## Uygulanan İşlemler
Projede temel makine öğrenmesi akışı uygulanmıştır:
1. Veri setinin oluşturulması
2. Veri setinin incelenmesi
3. Eksik değer kontrolü
4. Öznitelik üretimi
5. Kategorik değişkenlerin One-Hot Encoding ile dönüştürülmesi
6. Sayısal değişkenlerin StandardScaler ile ölçeklenmesi
7. Verinin train, validation ve test kümelerine ayrılması
8. Logistic Regression modelinin eğitilmesi
9. KNN modelinin eğitilmesi
10. Modellerin validation sonuçlarına göre karşılaştırılması
11. Seçilen modelin test verisi üzerinde değerlendirilmesi
12. Accuracy, Precision, Recall, F1-Score ve Confusion Matrix değerlerinin hesaplanması

## Üretilen Öznitelikler
Projede iki yeni öznitelik oluşturulmuştur:
* `gelir_grubu`: Müşterinin gelirini düşük, orta veya yüksek olarak sınıflandırır.
* `destek_talebi_var_mi`: Müşterinin en az bir destek talebi olup olmadığını gösterir.

## Modeller
Projede iki farklı sınıflandırma modeli kullanılmıştır:
* Logistic Regression
* K-Nearest Neighbors (KNN)

Validation sonuçları:
| Model               | Validation Accuracy |
| ------------------- | ------------------: |
| Logistic Regression |              73.33% |
| KNN                 |              66.67% |
Validation sonucuna göre **Logistic Regression** modeli seçilmiştir.

## Test Sonuçları
Seçilen Logistic Regression modelinin test sonuçları:
* Accuracy: 70%
* Precision: 0%
* Recall: 0%
* F1-Score: 0%

Confusion Matrix:
```text
[[21  1]
 [ 8  0]]
```

## Sonuç
Validation sonuçlarına göre Logistic Regression modeli KNN modelinden daha başarılı olmuştur. Logistic Regression'ın validation accuracy değeri %73.33, KNN modelinin ise %66.67 olarak bulunmuştur.
Test verisinde Logistic Regression modelinin accuracy değeri %70 olmuştur. Ancak precision, recall ve F1-score değerlerinin 0 olması, modelin churn eden müşterileri yeterince iyi tespit edemediğini göstermektedir. Bunun nedeni veri setinin basit şekilde oluşturulmuş olması ve churn değerlerinin rastgele belirlenmesi olabilir.

## Nasıl Çalıştırılır?
Öncelikle gerekli kütüphaneler yüklenmelidir:
```bash
pip install -r requirements.txt
```
Daha sonra Python dosyası çalıştırılabilir:
```bash
python "makine öğrenmesi ilk ödev.py"
```
Program çalıştırıldığında veri inceleme sonuçları, model karşılaştırmaları, test metrikleri ve confusion matrix ekrana yazdırılır.
