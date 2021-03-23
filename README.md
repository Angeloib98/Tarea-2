# Tarea-2 Regresión polinomial con descenso de gradiente

## Estudiantes:
Angelo M. Isaac Bonilla - 2016093978
Sofía Acerbi Martini - 2017111691
            
## Tabla de contenidos
* [Información general](#información-general)
* [Trayectorias de minimización en el espacio paramétrico](#trayectorias-de-minimización-en-el-espacio-paramétrico)
* [Evolución de la hipótesis](#apartado4)
* [Evolución del error J](#apartado5)
* [Aproximaciones para órdenes distintos de polinomios](#aproximaciones-para-órdenes-distintos-de-polinomios)

## Información General
El código presentado pretende analizar la regresión polinomial a través
de cinco distintos métodos para el cálculo del descenso de gradiente.

Los métodos disponibles son: batch gradient descent, vanilla stochastic 
gradient descent, stochastic gradient descent with momentum, stochastic
gradient descent with RMSprop y stochastic gradient descent with ADAM.

Todos los métodos excepto el "batch" usan minilotes con reemplazo.
En todos los apartados excepto el último se utiliza normalización "normal".
Para calcular las aproximaciones para órdenes superiores a dos se utiliza 
el normalizador "minmax" esto con el fin de evitar que con órdenes polinomiales 
altos se produzcan errores.

Todas las 20 gráficas se despliegan automáticamente al correr el código testdescent.m.
Para modificar alguno de los parámetros refiérase a la sección correspondiente.

## Trayectorias de minimización en el espacio paramétrico
Esta sección del código (identificada como "apartado 3") calcula la evolución de los valores de theta
para cada uno de los cinco métodos. Las figuras que genera se identifican con los números del 1 al 5.

### Parámetros editables
t0 es un vector fila que indica el punto inicial utilizado en el aprendizaje, en esta sección en 
particular se trabaja con polinomios de orden dos, por lo que el theta tendrá tres valores.

l_rate 

maxiter

epsilon

## Aproximaciones para órdenes distintos de polinomios
