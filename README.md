# Tarea-2 Regresión polinomial con descenso de gradiente

## Estudiantes:
            Angelo M. Isaac Bonilla - 2016093978
            Sofía Acerbi Martini - 2017111691
            
## Tabla de contenidos
* [Información general](#información-general)
* [Trayectorias de minimización en el espacio paramétrico](#apartado3)
* [Evolución de la hipótesis](#apartado4)
* [Evolución del error J](#apartado5)
* [Aproximaciones para órdenes distintos de polinomios](#apartado7)
* []

## Información General
El código presentado pretende analizar la regresión polinomial a través
de cinco distintos métodos para el cálculo del descenso de gradiente.

Los métodos disponibles son: batch gradient descent, vanilla stochastic 
gradient descent, stochastic gradient descent with momentum, stochastic
gradient descent with RMSprop y stochastic gradient descent with ADAM.

Todos los métodos excepto el “batch” usan minilotes con reemplazo.
En todos los apartados excepto el último se utiliza normalización "normal".
Para calcular las aproximaciones para órdenes superiores a dos se utiliza 
el normalizador 'minmax' esto con el fin de evitar que con órdenes polinomiales 
altos se produzcan errores.