#Gerekli kütüphanelerimizi ekleyelim
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures


#Veri setimizi df isimli dataframe'mimizin içine aktarıyoruz.
df = pd.read_csv("polynomial.csv", sep=";")


#Bu uygulamanın sonucunda veri setimizi tekrar bir inceleyelim.
plt.scatter(df["deneyim"], df["maas"])
plt.xlabel("Deneyim Yılı")
plt.ylabel("Maaş")
#plt.savefig("sonuc1.png", dpi=300)
#plt.show()


#Bir önceki uygulamadan görüldüğü gibi veriler, doğrusal bir düzlemde dağılmıyor.
#Bundan dolayı burada linear regression kullanıldığında doğru olmayan sonuçlar elde ederiz.
#Bunu şimdi ekranda grafik üzerinden görelim.
reg1 = LinearRegression()
reg1.fit(df[["deneyim"]], df["maas"])
plt.xlabel("Deneyim Yılı")
plt.ylabel("Maaş")
plt.scatter(df["deneyim"], df["maas"])
xekseni = df["deneyim"]
yekseni = reg1.predict(df[["deneyim"]])
#plt.plot(xekseni,yekseni,color="green", label="Linear Regression")
#plt.legend()
#plt.savefig("sonuc2.png", dpi=300)
#plt.show()
#Sonuçta görüdüğü üzere makine öğrenmesi modeli çok kötü bir sonuç veriyor.


#Bundan dolayı veri setimize göre doğru makine öğrenmesi modelini seçmemiz gerekir.
#Bu örnekte Polynomial Linear Regression modelinin uygun olduğuna karar verdik.
#Şimdi uygulamasını yapalım
#Polynomial Regression'u kullanabilmek için modelimizi çağırıyoruz.
#Fonksiyon çağrılırken polinomun derecesini de belirtiyoruz.
polynomial_regression = PolynomialFeatures(degree=4) #Burada degree polinomun derecesidir.
x_polynominal = polynomial_regression.fit_transform(df[["deneyim"]])


#Linear Regression modelinden yeni bir nesne oluşturup, modelimizi eğitiyoryuz.
reg2 = LinearRegression()
reg2.fit(x_polynominal, df["maas"])


#Şimdi yeni modelimizin nasıl sonuç verdiğini grafik üzerinde görelim.
y_head = reg2.predict(x_polynominal)
plt.plot(df["deneyim"], y_head, color="red", label="Polynomial Linear Regression")
plt.legend()
plt.scatter(df["deneyim"], df["maas"])
#plt.savefig("sonuc5.png", dpi=300)
#plt.show()
#Sonuc3
#Görüldüğü üzere bu veri seti için polynomial Linear regression modeli daha uygun.
#Fakat modelimizi daha uyumlu olması için biraz güncelleyeceğiz.
#Bunun için polinomun derecesini değiştireceğiz.


#Şimdi degree değerini 3 yapalım.
#Sonuc4
#Görüldüğü gibi daha uyumlu oldu ama halen istediğimiz seviyede değil.


#Şimdi degree'yi 4 yapalım.
#Sonuc5
#Şu anda çok daha iyi oldu. Burada aklımıza degree değerini çok büyük veririm diye gelebilir.
#Fakat her veri seti 10 satır boyutunda değil. Bu yüzden büyük vermeniz işlemciniz yoracak ve
#bilgisayarınızın kasmasına yol açacaktır.
#Bundan dolayı degree değerini olabilecek optimum seviyede tutmak çok önemlidir.


#Şimdi asıl yapmamız gerekn işi yapalım.
#Yeni bir pozisyon oluşturup vermemiz gereken maasş miktarını yapay zekaya hesaplatalım.
x_polynominal1 = polynomial_regression.fit_transform([[4.5]])
print(reg2.predict(x_polynominal1))
#Sonuç olarak 10958 dolar olarak tavsiye etti.
#Grafiğimize tekrar bakacak olursak ne kadar doğru bir tahmin yaptığını görebiliriz.