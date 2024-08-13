import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

df = pd.read_csv("polynomial.csv", sep=";")


plt.scatter(df["deneyim"], df["maas"])
plt.xlabel("Deneyim Yılı")
plt.ylabel("Maaş")
plt.savefig("sonuc1.png", dpi=300)
plt.show()


reg1 = LinearRegression()
reg1.fit(df[["deneyim"]], df["maas"])
plt.xlabel("Deneyim Yılı")
plt.ylabel("Maaş")
plt.scatter(df["deneyim"], df["maas"])
xekseni = df["deneyim"]
yekseni = reg1.predict(df[["deneyim"]])
plt.plot(xekseni,yekseni,color="green", label="Linear Regression")
plt.legend()
plt.savefig("sonuc2.png", dpi=300)
plt.show()



polynomial_regression = PolynomialFeatures(degree=4) #Burada degree polinomun derecesidir.
x_polynominal = polynomial_regression.fit_transform(df[["deneyim"]])


reg2 = LinearRegression()
reg2.fit(x_polynominal, df["maas"])

y_head = reg2.predict(x_polynominal)
plt.plot(df["deneyim"], y_head, color="red", label="Polynomial Linear Regression")
plt.legend()
plt.scatter(df["deneyim"], df["maas"])

x_polynominal1 = polynomial_regression.fit_transform([[4.5]])
print(reg2.predict(x_polynominal1))
