import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.float_format', lambda x: '%.2f' % x)

from  sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import  train_test_split, cross_val_score

df = pd.read_csv("datasets/advertising.csv")

X= df.drop('sales', axis=1)
y= df[["sales"]]


###################
#Model
###################

#Modelleme basamağı bir öncekinden biraz daha farklı olacak  veri setini train ve test olarak ayırmış olacağız

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1)

#X_train.shape =>  (160, 3)
#y_train.shape =>(160, 1)

#X_test.shape =>  (40, 1)

reg_model = LinearRegression().fit(X_train, y_train)

#sabit
reg_model.intercept_

#coefficients(w- weights)
reg_model.coef_


###############
#Tahmin
###############

#Aşağıdaki gözlem değerlerine göre satışın beklenen değeri nedir?

# TV: 30
# radio: 10
# newspaper: 40

# 2.90
# 0.0468431, 0.17854434, 0.00258619

#Sales = 2.90+ 0.04*TV + 0.17*radio + 0.002*newspaper
dene = 2.90+ 0.04*30 + 0.17*10 + 0.002*40
print(dene)

#FONSKİYONEL ŞEKİLDE YAPIMI

yeni_veri = pd.DataFrame({'TV': [30], 'radio': [10], 'newspaper': [40]})  #yeni veriyi oluşturup dataframe 'e dönüştürdük


reg_model.predict(yeni_veri)


######################
#Tahmin Başarısını değerlendirme
######################

#Train RMSE
y_pred = reg_model.predict(X_train)
print("train hatası : " ,np.sqrt(mean_squared_error(y_train, y_pred)))
#1.73

#TRAİN RKARE(RSQUARE)
reg_model.score(X_train, y_train)
#0.89
#Veri setindeki bağımsız değişkenlerin bağımlı değişkenleri açıklaması %90 civarında yani oldukça yüksek bir oranda açıklama yetebeğine sahipler


#Test RMSE
y_pred = reg_model.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred))
#1.41

#Train hatası test hatasından yüksek çıktı. Normalde test hatası train hatasından yüksek çıkar.
#Buarada beklenti dışı ve güzel bir durum var

#10 Katlı Cv RMSE
np.mean(np.sqrt(-cross_val_score(reg_model,
                                 X,
                                 y,
                                 cv=10,
                                 scoring="neg_mean_squared_error")))
#-cross_val_score'u "neg_mean_squared_error"u veriyor yani neatifi. Oyüzden - ile çarptık önemli
#Çünkü - hata olmaz bize ortalama hata lazım

#1.69 geldi
# 10 katlı cv de test hatası 1.69 iken diğerinde 1.71 bu veri seti için 10 katlı cv daha güvnilir

