# Tarea-2 Regresión polinomial con descenso de gradiente

## Estudiantes:
Angelo M. Isaac Bonilla - 2016093978

Sofía Acerbi Martini - 2017111691
            
## Tabla de contenidos
* [Información general](#información-general)
* [Trayectorias de minimización en el espacio paramétrico](#trayectorias-de-minimización-en-el-espacio-paramétrico)
* [Evolución de la hipótesis](#evolución-de-la-hipótesis)
* [Evolución del error J](#evolución-del-error-J)
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

Todas las 20 gráficas se despliegan automáticamente al correr el código *testdescent.m*.
Para modificar alguno de los parámetros refiérase a la sección correspondiente.

## Trayectorias de minimización en el espacio paramétrico
Esta sección del código (identificada como "apartado 3") calcula la evolución de los valores de theta
para cada uno de los cinco métodos, cuando se tiene el caso particular de aproximaciones cuadráticas. 
Las figuras que genera se identifican con los números del 1 al 5.

### Parámetros editables
Los parámetros son específicos para cada método, no se pueden cambiar de forma global.

*t0* es un vector fila que indica el punto inicial utilizado en el aprendizaje, en esta sección en 
particular se trabaja con polinomios de orden dos, por lo que *t0* tendrá tres valores.

*l_rate* corresponde a la tasa de aprendizaje.

*maxiter* es el número de iteraciones máximas que se desea ejecutar.

*epsilon* es el valor utilizado para calcular el error entre una iteración y la siguiente.

*minibatch* es la variable que contiene el tamaño de los minibatch para los métodos "stochastic". Se
suele recomendar utilizar *0.5·rows(Xo)*.

*beta* y *beta2* son parámetros utilizados para la optimización, se recomienda no cambiarlos.


## Evolución de la hipótesis

## Evolución del error J

## Aproximaciones para órdenes distintos de polinomios
Esta sección del código (identificada como "apartado 7") realiza una comparación entre las aproximaciones
finales para regresiones con polinomios de diversos órdenes.







