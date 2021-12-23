#!/usr/bin/env python
# coding: utf-8

# In[1]:


#question 2 part a
from pandas import *
from sympy import *
import math

def newton_interpolation_func(vec1, vec2):
    F = [[0] * len(vec1)  for i in range(len(vec1))]
    for j in range(len(vec2)):
        F[j][0] = vec2[j]
    for i in range(1, len(vec1)):
        for j in range(1, len(vec1)):
            F[i][j] = (F[i][j-1] - F[i-1][j-1]) / (vec1[i] - vec1[i-j])
    
    x = symbols('x')
    pn = F[0][0]
    for i in range(1, len(vec1)):
        l = 1
        for j in range(0, i):
             l *= (x - vec1[j])
        pn += F[i][i]*l
        
    divided_diff = [[0] * len(vec1)  for i in range(len(vec1))]
    for i in range(len(vec1)):
        for j in range(i, len(vec1)):
            divided_diff[j][i] = F[j][i]
    table = DataFrame(divided_diff, columns=['divided differences ' + str(i) for i in range(len(vec1))])
    returnlist = [pn, table] 
    return returnlist #return list that the first object in the list is the polynom and second is the divided differnces


# In[187]:


#question 2 part b

# n = 5
n = 5
vectorX = [0] * n
vectorY = [0] * n
for i in range(n):
    vectorX[i] = -5 + 10 * (i) / (n - 1)
    vectorY[i] = 1 / (1 + vectorX[i]**2)

list5 = newton_interpolation_func(vectorX, vectorY)
print("the polynom n = 5:\n", simplify(list5[0]))
print("the divided differences:\n", list5[1])


# n = 10
n = 10
vectorX = [0] * n
vectorY = [0] * n
for i in range(n):
    vectorX[i] = -5 + 10 * (i) / (n - 1)
    vectorY[i] = 1 / (1 + vectorX[i]**2)

list10 = newton_interpolation_func(vectorX, vectorY)
print("the polynom n = 10:\n", simplify(list10[0]))

# n = 15
n = 15
vectorX = [0] * n
vectorY = [0] * n
for i in range(n):
    vectorX[i] = -5 + 10 * (i) / (n - 1)
    vectorY[i] = 1 / (1 + vectorX[i]**2)

list15 = newton_interpolation_func(vectorX, vectorY)
print("the polynom n = 15:\n", simplify(list15[0]))

# n = 20
n = 20
vectorX = [0] * n
vectorY = [0] * n
for i in range(n):
    vectorX[i] = -5 + 10 * (i) / (n - 1)
    vectorY[i] = 1 / (1 + vectorX[i]**2)
list20 = newton_interpolation_func(vectorX, vectorY)
print("the polynom n = 20:\n", simplify(list20[0]))

#question 2 part c
#the functin (in red) and the interpolation (in blue) in one graph 

x = symbols('x')
f = 1 / (1 + x**2)
pn5 = list5[0]
pn10 = list10[0]
pn15 = list15[0]
pn20 = list20[0]

# between [-5, 5]

p1 = plot(f, pn5, pn10, pn15, pn20, (x, -5, 5), show=False)
p1[0].line_color = 'red'
#p1.show()

#between [-2.5, 2.5]
p2 = plot(f, pn5, pn10, pn15, pn20, (x, -2.5, 2.5), show=False)
p2[0].line_color = 'red'
#p2.show()

#question 2 part d
f_func = lambdify(x, f)
g5 = lambdify(x, pn5)
g10 = lambdify(x, pn10)
g15 = lambdify(x, pn15)
g20 = lambdify(x, pn20)
print(f_func(1 + 10**0.5))
print(simplify(g5(1 + 10**0.5)))
print(simplify(g10(1 + 10**0.5)))
print(simplify(g15(1 + 10**0.5)))
print(simplify(g20(1 + 10**0.5)))

p2 = plot(f, pn5, pn10, pn15, pn20, (x, 1 + sqrt(10) - 0.5, 1 + sqrt(10) + 0.5 ), show=False)
p2[0].line_color = 'red'
#p2.show()


# In[29]:


import numpy as np
from sympy import *
import math

A = np.array([[1, 0, 0], [2, 6, 1], [0, 0, 1]])
B = np.array([0, (3/1) * (15-5) - (3/2) * (5 - 10), 0])
C = A.dot(B)
print(C)
c1 = (C[1])
print(c1)
b1 = (1) * (15 - 5) - (1/3) * (2 * c1)
print(b1)
d1 = (-c1)/(3*1)
print(d1)
x = symbols('x')
s1 = 5 + b1*(x - 2) + c1*((x-2)**2) + d1*((x-2)**3)
print(s1)
l1 = lambdify(x, s1)
print(l1(2))
b0 = (1/2) * (5 - 10) - (2/3) * (c1)
d0 = c1 / 6
s0 = 10 + b0*x + d0*(x**3)
print(s0)
l0 = lambdify(x, s0)
print(l0(2))

