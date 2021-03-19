% Main entry code
% This is the script called to start the evaluation process

1;
pkg load optim;

# Data stored each sample in a row, where the last row is the label
D=load("escazu40.dat");

# Extract the areas and the prices
Xo=D(:,1);
Yo=D(:,4);

## Normalizar los datos
normalizer_type = "normal";

nx = normalizer(normalizer_type);
X = nx.fit_transform(Xo);  
ny = normalizer(normalizer_type);
Y = ny.fit_transform(Yo);

########################################################################                         
## Apartado 3: Trayectorias de minimización en el espacio paramétrico ##
## para el caso particular de aproximaciones cuadráticas              ##
########################################################################

## Método Batch Gradient Descent ##
t0 = [-1 -0.2 -0.3];
l_rate=0.004;
maxiter=300;
epsilon=0.005;
method="batch";

[thetas_batch,errors_batch]=descentpoly(@loss,@gradloss,t0,X,Y,l_rate,
                            "method",method,
                            "epsilon",epsilon,
                            "maxiter",maxiter);
                           
figure(1,"name","Trayectoria de minimización en el espacio paramétrico - Método Batch Gradient Descent");
hold off;
plot3(thetas_batch(:,1),thetas_batch(:,2),thetas_batch(:,3),"k-");
hold on;
plot3(thetas_batch(:,1),thetas_batch(:,2),thetas_batch(:,3),"ob");
xlabel('{\theta_0}'); ylabel('{\theta_1}'); zlabel('{\theta_2}');
grid;

## Método Stochastic Gradient Descent ##
t0 = [-1 -0.2 -0.3];
l_rate=0.003;
maxiter=500;
epsilon=0.005;
minibatch=10; %%0.5*rows(Xo);
method="stochastic";

[thetas_stoch,errors_stoch]=descentpoly(@loss,@gradloss,t0,X,Y,l_rate,
                            "method",method,
                            "maxiter",maxiter,
                            "epsilon",epsilon,
                            "minibatch",minibatch);
                           
figure(2,"name","Trayectoria de minimización en el espacio paramétrico - Método Stochastic Gradient Descent");
hold off;
plot3(thetas_stoch(:,1),thetas_stoch(:,2),thetas_stoch(:,3),"k-");
hold on;
plot3(thetas_stoch(:,1),thetas_stoch(:,2),thetas_stoch(:,3),"ob");
xlabel('{\theta_0}'); ylabel('{\theta_1}'); zlabel('{\theta_2}');
grid;

## Método Stochastic Gradient Descent with Momentum ##
t0 = [-1 -0.2 -0.3];
l_rate=0.003;
maxiter=500;
epsilon=0.005;
minibatch=10;
method="momentum";

[thetas_mom,errors_mom]=descentpoly(@loss,@gradloss,t0,X,Y,l_rate,
                            "method",method,
                            "maxiter",maxiter,
                            "epsilon",epsilon,
                            "minibatch",minibatch);
                           
figure(3,"name","Trayectoria de minimización en el espacio paramétrico - Stoch. Gradient Descent with Momentum");
hold off;
plot3(thetas_mom(:,1),thetas_mom(:,2),thetas_mom(:,3),"k-");
hold on;
plot3(thetas_mom(:,1),thetas_mom(:,2),thetas_mom(:,3),"ob");
xlabel('{\theta_0}'); ylabel('{\theta_1}'); zlabel('{\theta_2}');
grid;

## Método Stochastic Gradient Descent with RMSprop ##
t0 = [-1 -0.2 -0.3];
l_rate=0.005;
beta=0.9;
maxiter=500;
epsilon=1e-7;
minibatch=10; %%0.5*rows(Xo);
method="rmsprop";

[thetas_rms,errors_rms]=descentpoly(@loss,@gradloss,t0,X,Y,l_rate,
                            "method",method,
                            "beta", beta,
                            "maxiter",maxiter,
                            "epsilon",epsilon,
                            "minibatch",minibatch);
                           
figure(4,"name","Trayectoria de minimización en el espacio paramétrico - Stoch. Gradient Descent with RMSprop");
hold off;
plot3(thetas_rms(:,1),thetas_rms(:,2),thetas_rms(:,3),"k-");
hold on;
plot3(thetas_rms(:,1),thetas_rms(:,2),thetas_rms(:,3),"ob");
xlabel('{\theta_0}'); ylabel('{\theta_1}'); zlabel('{\theta_2}');
grid;

## Método Stochastic Gradient Descent with Adam ##
t0 = [-1 -0.2 -0.3];
l_rate=0.003;
beta=0.9;
maxiter=1000;
epsilon=0.001;
minibatch=10; %%0.5*rows(Xo);
method="adam";

[thetas_adam,errors_adam]=descentpoly(@loss,@gradloss,t0,X,Y,l_rate,
                            "method",method,
                            "beta", beta,
                            "maxiter",maxiter,
                            "epsilon",epsilon,
                            "minibatch",minibatch);
                           
figure(5,"name","Trayectoria de minimización en el espacio paramétrico - Stoch. Gradient Descent with Adam");
hold off;
plot3(thetas_adam(:,1),thetas_adam(:,2),thetas_adam(:,3),"k-");
hold on;
plot3(thetas_adam(:,1),thetas_adam(:,2),thetas_adam(:,3),"ob");
xlabel('{\theta_0}'); ylabel('{\theta_1}'); zlabel('{\theta_2}');
grid;


########################################################################                         
## Apartado 4:  ##
########################################################################


## Apartado 6: 

## Apartado 7: 

