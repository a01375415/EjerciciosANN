import numpy as np
import random
import time

#Función de costo
def MSE(out,y):
    return 1/2*(out-y)**2

#Función de activación (Sigmoide)
def sigmoide(h):
    return 1/(1+np.exp(-h))

#Derivada de la función de activación (Sigmoide)
def dsigmoide(h):
    return np.exp(-h)/(1+np.exp(-h))**2

#Cálculo de ecuaciones de la red
def calc():
    global h1,h2,sh1,sh2,o1,o2,so1,so2
    h1=w1*i1+w2*i2+b1
    sh1=sigmoide(h1)
    h2=w3*i1+w4*i2+b1
    sh2=sigmoide(h2)
    o1=w5*sh1+w6*sh2+b2
    so1=sigmoide(o1)
    o2=w7*sh1+w8*sh2+b2
    so2=sigmoide(o2)

#Tasa de aprendizaje
alpha=0.4

#Datos
i1=0.05
i2=0.1
y1=0.01
y2=0.09

#Valor aleatorio
random.seed(11)
inival=random.random()

#Valores inicializados
w1=inival
w2=inival
w3=inival
w4=inival
w5=inival
w6=inival
w7=inival
w8=inival
b1=inival
b2=inival
calc()

#Minimización por gradiente descendente
start = time.time()
while MSE(so1,y1)>1e-12 or MSE(so2,y2)>1e-12:
    calc()
    dw1=((so1-y1)*dsigmoide(o1)*w5*dsigmoide(h1)*i1
                +(so2-y2)*dsigmoide(o2)*w7*dsigmoide(h1)*i1)
    dw2=((so1-y1)*dsigmoide(o1)*w5*dsigmoide(h1)*i2
                +(so2-y2)*dsigmoide(o2)*w7*dsigmoide(h1)*i2)
    dw3=((so1-y1)*dsigmoide(o1)*w6*dsigmoide(h1)*i1
                +(so2-y2)*dsigmoide(o2)*w8*dsigmoide(h1)*i1)
    dw4=((so1-y1)*dsigmoide(o1)*w6*dsigmoide(h1)*i2
                +(so2-y2)*dsigmoide(o2)*w8*dsigmoide(h1)*i2)
    dw5=((so1-y1)*dsigmoide(o1)*sh1)
    dw6=((so1-y1)*dsigmoide(o1)*sh2)
    dw7=((so2-y2)*dsigmoide(o2)*sh1)
    dw8=((so2-y2)*dsigmoide(o2)*sh2)
    db1=((so1-y1)*dsigmoide(o1)*(w5*dsigmoide(h1)+w6*dsigmoide(h2))
                +(so2-y2)*dsigmoide(o2)*(w7*dsigmoide(h1)+w8*dsigmoide(h2)))
    db2=((so1-y1)*dsigmoide(o1)+(so2-y2)*dsigmoide(o2))

    w1=w1-alpha*dw1
    w2=w2-alpha*dw2
    w3=w3-alpha*dw3
    w4=w4-alpha*dw4
    w5=w5-alpha*dw5
    w6=w6-alpha*dw6
    w7=w7-alpha*dw7
    w8=w8-alpha*dw8
    b1=b1-alpha*db1
    b2=b2-alpha*db2

#Salidas
print("w1: {}, w2: {}, w3: {}, w4: {}, w5: {}, w6: {}, w7: {}, w8: {}, b1: {}, b2: {}".format(round(w1,4),round(w2,4),round(w3,4),round(w4,4),round(w5,4)
                                                                                        ,round(w6,4),round(w7,4),round(w8,4),round(b1,4),round(b2,4)))
calc()
print("So1: {}".format(round(so1,6)))
print("So2: {}".format(round(so2,6)))
end = time.time()
print("Tiempo de ejecución: {} seg".format(end - start))