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

Todas las 20 gráficas se despliegan automáticamente al correr el archivo *testdescent.m*. En este último 
archivo es posible modificar los parámetros, para ello refiérase a la sección correspondiente.

## Trayectorias de minimización en el espacio paramétrico
Esta sección del código (identificada como "apartado 3" en *testdescent.m*) calcula la evolución de los valores de theta
para cada uno de los cinco métodos, cuando se tiene el caso particular de aproximaciones cuadráticas. 
Las figuras que genera se identifican con los números del 1 al 5.

### Parámetros editables
Los parámetros son específicos para cada método, no se pueden cambiar de forma global. Para modificarlos
se tienen que editar las constantes que se encuentran debajo del nombre de cada método, según lo siguiente:

`t0` es un vector fila que indica el punto inicial utilizado en el aprendizaje, en esta sección en 
particular se trabaja con polinomios de orden dos, por lo que *t0* será un vector con tres entradas.

`l_rate` corresponde a la tasa de aprendizaje.

`maxiter` es el número de iteraciones máximas que se desean ejecutar.

`epsilon` es el valor utilizado para calcular el error entre el theta de una iteración y el de la siguiente.

`minibatch` es la variable que contiene el tamaño de los minibatch para los métodos "stochastic". Se
suele recomendar utilizar `0.5*rows(Xo)`.

`beta` y `beta2` son parámetros utilizados para la optimización, se recomienda no cambiarlos.


## Evolución de la hipótesis
Esta sección del código (identificada como "apartado 4" en *testdescent.m*) ...

## Evolución del error J
Esta sección del código (identificada como "apartado 5" en *testdescent.m*) ...

## Aproximaciones para órdenes distintos de polinomios
Esta sección del código (identificada como "apartado 7" en *testdescent.m*) realiza una comparación entre las aproximaciones
finales para regresiones con polinomios de órdenes distintos. Las figuras que genera se identifican con 
los números del 16 al 20.

### Parámetros editables
`orden` es un vector fila que contiene los varios valores de órden del polinomio que se desean evaluar.
Es posible modificar los 4 propuestos sin necesidad de mayores cambios.

También es posible graficar más de 4 órdenes polinomiales distintos, hasta un máximo de 10 en
un único gráfico. De ser así, el cálculo se realiza de forma automática, pero se deberán agregar las siguientes 
instrucciones para mostrar más de 4 curvas:

```
plot(areas,precios_poly(:,n),'color',"linewidth",0.5);
hold on;
```
Donde `n` es el número de la curva y `'color'` el color de la curva. 
Para agregar la respectiva leyenda al gráfico se utiliza:

```
cstrcat("n=", mat2str(orden(n)))

```