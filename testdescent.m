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

[thetas_batch,errors_batch]=descentpoly(@loss,@gradloss,t0,X,Y,l_rate,
                            "method","batch",
                            "epsilon",epsilon,
                            "maxiter",maxiter);
                           
figure(1,"name","Trayectoria de minimización en el espacio paramétrico");
hold off;
plot3(thetas_batch(:,1),thetas_batch(:,2),thetas_batch(:,3),"k-");
hold on;
plot3(thetas_batch(:,1),thetas_batch(:,2),thetas_batch(:,3),"ob");
t0_str=cstrcat("Iniciado en ", mat2str(t0));
title({"Trayectoria de minimización en el espacio paramétrico";
              t0_str; "Método Batch Gradient Descent"},
              "fontsize", 20);
xlabel('{\theta_0}',"fontsize", 20);
ylabel('{\theta_1}',"fontsize", 20); 
zlabel('{\theta_2}',"fontsize", 20);
grid;

## Método Stochastic Gradient Descent ##
t0 = [-1 -0.2 -0.3];
l_rate=0.001;
maxiter=500;
epsilon=0.001;
minibatch=20; %%0.5*rows(Xo);

[thetas_stoch,errors_stoch]=descentpoly(@loss,@gradloss,t0,X,Y,l_rate,
                            "method","stochastic",
                            "maxiter",maxiter,
                            "epsilon",epsilon,
                            "minibatch",minibatch);
                           
figure(2,"name","Trayectoria de minimización en el espacio paramétrico");
hold off;
plot3(thetas_stoch(:,1),thetas_stoch(:,2),thetas_stoch(:,3),"k-");
hold on;
plot3(thetas_stoch(:,1),thetas_stoch(:,2),thetas_stoch(:,3),"ob");
t0_str=cstrcat("Iniciado en ", mat2str(t0));
title({"Trayectoria de minimización en el espacio paramétrico";
              t0_str; "Método Stochastic Gradient Descent"},
              "fontsize", 20);
xlabel('{\theta_0}',"fontsize", 20);
ylabel('{\theta_1}',"fontsize", 20); 
zlabel('{\theta_2}',"fontsize", 20);
grid;

## Método Stochastic Gradient Descent with Momentum ##
t0 = [-1 -0.2 -0.3];
l_rate=0.003;
maxiter=500;
epsilon=0.001;
minibatch=20;

[thetas_mom,errors_mom]=descentpoly(@loss,@gradloss,t0,X,Y,l_rate,
                            "method","momentum",
                            "maxiter",maxiter,
                            "epsilon",epsilon,
                            "minibatch",minibatch);
                           
figure(3,"name","Trayectoria de minimización en el espacio paramétrico");
hold off;
plot3(thetas_mom(:,1),thetas_mom(:,2),thetas_mom(:,3),"k-");
hold on;
plot3(thetas_mom(:,1),thetas_mom(:,2),thetas_mom(:,3),"ob");
t0_str=cstrcat("Iniciado en ", mat2str(t0));
title({"Trayectoria de minimización en el espacio paramétrico";
              t0_str; "Método Stochastic Gradient Descent with Momentum"},
              "fontsize", 20);
xlabel('{\theta_0}',"fontsize", 20);
ylabel('{\theta_1}',"fontsize", 20); 
zlabel('{\theta_2}',"fontsize", 20);
grid;

## Método Stochastic Gradient Descent with RMSprop ##
t0 = [-1 -0.2 -0.3];
l_rate=0.01;
beta=0.9;
beta2=0.95;
maxiter=500;
epsilon=0.005;
minibatch=20; %%0.5*rows(Xo);

[thetas_rms,errors_rms]=descentpoly(@loss,@gradloss,t0,X,Y,l_rate,
                            "method","rmsprop",
                            "beta", beta,
                            "beta2", beta2,
                            "maxiter",maxiter,
                            "epsilon",epsilon,
                            "minibatch",minibatch);
                           
figure(4,"name","Trayectoria de minimización en el espacio paramétrico");
hold off;
plot3(thetas_rms(:,1),thetas_rms(:,2),thetas_rms(:,3),"k-");
hold on;
plot3(thetas_rms(:,1),thetas_rms(:,2),thetas_rms(:,3),"ob");
t0_str=cstrcat("Iniciado en ", mat2str(t0));
title({"Trayectoria de minimización en el espacio paramétrico";
              t0_str; "Método Stochastic Gradient Descent with RMSprop"},
              "fontsize", 20);
xlabel('{\theta_0}',"fontsize", 20);
ylabel('{\theta_1}',"fontsize", 20); 
zlabel('{\theta_2}',"fontsize", 20);
grid;

## Método Stochastic Gradient Descent with Adam ##
t0 = [-1 -0.2 -0.3];
l_rate=0.05;
beta=0.9;
maxiter=500;
epsilon=0.005;
minibatch=20; %%0.5*rows(Xo);

[thetas_adam,errors_adam]=descentpoly(@loss,@gradloss,t0,X,Y,l_rate,
                            "method","adam",
                            "beta", beta,
                            "maxiter",maxiter,
                            "epsilon",epsilon,
                            "minibatch",minibatch);
                           
figure(5,"name","Trayectoria de minimización en el espacio paramétrico");
hold off;
plot3(thetas_adam(:,1),thetas_adam(:,2),thetas_adam(:,3),"k-");
hold on;
plot3(thetas_adam(:,1),thetas_adam(:,2),thetas_adam(:,3),"ob");
t0_str=cstrcat("Iniciado en ", mat2str(t0));
title({"Trayectoria de minimización en el espacio paramétrico";
              t0_str; "Método Stochastic Gradient Descent with Adam"},
              "fontsize", 20);
xlabel('{\theta_0}',"fontsize", 20);
ylabel('{\theta_1}',"fontsize", 20); 
zlabel('{\theta_2}',"fontsize", 20);
grid;


########################################################################                         
## Apartado 4: Evolución de la hipótesis en cada caso para llegar     ##
## al mínimo.                                                         ##
########################################################################

#Tomado como referencia batch_grad_descent.m (brindado por el profesor)
minX = min(Xo);
maxX = max(Xo);
xxx=linspace(minX,maxX,length(Xo));
X1=nx.transform([ones(length(xxx),1) xxx']);

## Método Batch Gradient Descent ## 
figure(6,"name","Evolución de la hipótesis Batch Gradient Descent");
plot(nx.itransform(X),ny.itransform(Y),"*b",2);
hold on;

Y1=X1 * thetas_batch(1,1:2)';
Y11=ny.itransform(Y1);
plot(xxx,Y11,'k',"linewidth",2);

for (i=[2:rows(thetas_batch(:,1:2))])
  Y1s=X1 * thetas_batch(i,1:2)';
  Y11=ny.itransform(Y1s);    	
  plot(xxx,Y11,'c',"linewidth",0.5);
endfor;

plot(xxx,Y11,'g',"linewidth",3);



## Método Stochastic Gradient Descent ##
figure(7,"name","Evolución de la hipótesis Stochastic Gradient Descent");
plot(nx.itransform(X),ny.itransform(Y),"*b");
hold on;

Y1=X1 * thetas_stoch(1,1:2)';
Y11=ny.itransform(Y1);
plot(xxx,Y11,'k',"linewidth",2);

for (i=[2:rows(thetas_stoch(:,1:2))])
  Y1s=X1 * thetas_stoch(i,1:2)';
  Y11=ny.itransform(Y1s);    	
  plot(xxx,Y11,'c',"linewidth",0.5);
endfor;
  
plot(xxx,Y11,'g',"linewidth",3);



## Método Stochastic Gradient Descent with momentum ## 
figure(8,"name","Evolución de la hipótesis Stochastic Gradient Descent with Momentum");
plot(nx.itransform(X),ny.itransform(Y),"*b");
hold on;

Y1=X1 * thetas_mom(1,1:2)';
Y11=ny.itransform(Y1);
plot(xxx,Y11,'k',"linewidth",2);

for (i=[2:rows(thetas_mom(:,1:2))])
  Y1s=X1 * thetas_mom(i,1:2)';
  Y11=ny.itransform(Y1s);    	
  plot(xxx,Y11,'c',"linewidth",0.5);
endfor;

plot(xxx,Y11,'g',"linewidth",3);



## Método Stochastic Gradient Descent with RMSprop##
figure(9,"name","Evolución de la hipótesis Stochastic Gradient Descent with RMSprop");
plot(nx.itransform(X),ny.itransform(Y),"*b");
hold on;

Y1=X1 * thetas_mom(1,1:2)';
Y11=ny.itransform(Y1);
plot(xxx,Y11,'k',"linewidth",2);

for (i=[2:rows(thetas_rms(:,1:2))])
  Y1s=X1 * thetas_rms(i,1:2)';
  Y11=ny.itransform(Y1s);    	
  plot(xxx,Y11,'--c',"linewidth",0.3);
endfor;

plot(xxx,Y11,'g',"linewidth",3);



## Método Stochastic Gradient Descent with ADAM## 
figure(10,"name","Evolución de la hipótesis Stochastic Gradient Descent with ADAM");
plot(nx.itransform(X),ny.itransform(Y),"*b");
hold on;

Y1=X1 * thetas_adam(1,1:2)';
Y11=ny.itransform(Y1);
plot(xxx,Y11,'k',"linewidth",2);

for (i=[2:rows(thetas_adam(:,1:2))])
  Y1s=X1 * thetas_adam(i,1:2)';
  Y11=ny.itransform(Y1s);    	
  plot(xxx,Y11,'--c',"linewidth",0.3);
endfor;

plot(xxx,Y11,'g',"linewidth",3);
hold off;


########################################################################                         
## Apartado 5: Evolución del error J(?) en función del número de      ##
## iteración, para varios valores de la tasa de aprendizaje.          ##
########################################################################

t0 = [-1 -0.2 -0.3];
l_rate=0.01; #tiene que ser menor o igual a 0.01
maxiter=300;
epsilon=0.005;
method="batch";

[thetas_batch,errors_batch]=descentpoly(@loss,@gradloss,t0,X,Y,l_rate,
                            "method",method,
                            "epsilon",epsilon,
                            "maxiter",maxiter);
                           

figure(15,"name","error iteracion");

legend({'alpha=0.01','alpha=0.001','alpha=0.005','alpha=0.0043'}
                              ,"location","northeastoutside");
hold on;
iter_batch=[0:1:length(errors_batch)-1]';
plot(iter_batch,errors_batch,'b','linewidth',2);

l_rate=0.001; #tiene que ser menor o igual a 0.01
[thetas_batch,errors_batch]=descentpoly(@loss,@gradloss,t0,X,Y,l_rate,
                            "method",method,
                            "epsilon",epsilon,
                            "maxiter",maxiter);
iter_batch=[0:1:length(errors_batch)-1]';
plot(iter_batch,errors_batch,'r','linewidth',2);


l_rate=0.005; #tiene que ser menor o igual a 0.01
[thetas_batch,errors_batch]=descentpoly(@loss,@gradloss,t0,X,Y,l_rate,
                            "method",method,
                            "epsilon",epsilon,
                            "maxiter",maxiter);
iter_batch=[0:1:length(errors_batch)-1]';
plot(iter_batch,errors_batch,'g','linewidth',2);

l_rate=0.0043; #tiene que ser menor o igual a 0.01
[thetas_batch,errors_batch]=descentpoly(@loss,@gradloss,t0,X,Y,l_rate,
                            "method",method,
                            "epsilon",epsilon,
                            "maxiter",maxiter);
iter_batch=[0:1:length(errors_batch)-1]';
plot(iter_batch,errors_batch,'c','linewidth',2);
hold off;


##maxiter=300;
##figure(11,"name","Error vs iterations graph for batch");
##hold on;
##aux=1.5;
##n=6;
##errors_batch=[];
##errors_batch_prom=[];
##for i=(1:5)
##  [thetas_batch,errors_batchx]=descentpoly(@loss,@gradloss,t0,X,Y,aux/n,
##                            "method",method,
##                            "epsilon",epsilon,
##                            "maxiter",maxiter);
##   
##   errors_batch=[errors_batch,errors_batchx];                         
##endfor
##for i=1:length(errors_batch)
##  errors_batch_prom=[errors_batch_prom ; mean(errors_batch(i:40,:))];
##  i=i+40;
##endfor
##
##iter_batch=[0:1:length(errors_batch)-1]';
##plot(iter_batch,errors_batch_prom(:,1),'b','linewidth',2);
##plot(iter_batch,errors_batch_prom(:,2),'r','linewidth',2);
##plot(iter_batch,errors_batch_prom(:,3),'g','linewidth',2);
##plot(iter_batch,errors_batch_prom(:,4),'y','linewidth',2);
##plot(iter_batch,errors_batch_prom(:,5),'c','linewidth',2);
##axis([0 350 0 100]);
##hold off;
#iter_stoch=[0:1:length(errors_stoch)-1]';
#plot(iter_stoch,errors_stoch,'g','linewidth',1);

#iter_mom=[0:1:length(errors_mom)-1]';
#plot(iter_mom,errors_mom,'r','linewidth',1);

#iter_rms=[0:1:length(errors_rms)-1]';
#plot(iter_rms,errors_rms,'m','linewidth',1);

#iter_adam=[0:1:length(errors_adam)-1]';
#plot(iter_adam,errors_adam,'y','linewidth',1);


## Apartado 7: 

