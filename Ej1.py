import numpy as np
import random
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import plotly.graph_objects as go
from sklearn.metrics import r2_score
import time

#Regresión con Scikit Learn
def reg(X,Y):
    poly = PolynomialFeatures(degree=2, include_bias=False)
    poly_features = poly.fit_transform(X.reshape(-1, 1))
    poly_reg_model = LinearRegression()
    poly_reg_model.fit(poly_features, Y)
    return round(poly_reg_model.coef_[1],4),round(poly_reg_model.coef_[0],4),round(poly_reg_model.intercept_,4)

# Función de costo
def mse(X,Y,a,b,c):
    m=len(X)
    suma=0
    for i in range(m):
        suma+=(a*X[i]**2+b*X[i]+c-Y[i])**2
    return suma/m

# Derivadas de función costo
def dmse(X,Y,a,b,c):
    da=0
    db=0
    dc=0
    m=len(X)
    for i in range(m):
        da+=X[i]**2*(a*X[i]**2+b*X[i]+c-Y[i])
        db+=X[i]*(a*X[i]**2+b*X[i]+c-Y[i])
        dc+=a*X[i]**2+b*X[i]+c-Y[i]
    return da/m,db/m,dc/m

#Gradiente descendente
def descgrad(X,Y,alpha):
    random.seed(11)
    inival=random.random()
    a=inival
    b=inival
    c=inival
    while mse(X,Y,a,b,c)>0.05:
        da,db,dc=dmse(X,Y,a,b,c)
        a-=alpha*da
        b-=alpha*db
        c-=alpha*dc
    return round(a,4),round(b,4),round(c,4)

#Datos
X=np.array([2.1,6,9.1,7.5])
Y=np.array([8.9,1.2,15.7,6.5])

#Parámetros obtenidos
startreg = time.time()
a1,b1,c1=reg(X,Y)
endreg = time.time()
startdesc = time.time()
a2,b2,c2=descgrad(X,Y,0.0005)
enddesc = time.time()

#Función Cuadrática
def func(x,a,b,c):
    return a*x**2+b*x+c

#Salidas
print("Con Scikit Learn se obtienen los parámetros en {}seg a={}, b={}, c={}, R^2={}"
                                        .format(endreg-startreg,a1,b1,c1,round(r2_score(Y,func(X,a1,b1,c1)),6)))
print("Con el método de Gradiente descendente se obtienen los parámetros en {}seg a={}, b={}, c={}, R^2={}"
                                        .format(enddesc-startdesc,a2,b2,c2,round(r2_score(Y,func(X,a2,b2,c2)),6)))

#Gráficas
fig =go.Figure()
fig.add_trace(go.Scatter(x=X,y=Y,line_color='black',line_width=3,
                        mode='markers',marker_symbol='x-open',marker_size=14,name="Mediciones"))
fig.add_trace(go.Scatter(x=np.arange(0,10,0.1),y=func(np.arange(0,10,0.1),a1,b1,c1),
                        line_color="blue",line_width=3,mode='lines',name="Scikit-Learn"))
fig.add_trace(go.Scatter(x=np.arange(0,10,0.1),y=func(np.arange(0,10,0.1),a2,b2,c2),
                        line_color="red",line_width=3,mode='lines',name="Gradiente descendente"))
fig.show()
fig.write_image("ej1.png",height=800,width=1000)