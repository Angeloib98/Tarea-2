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


######################################################################                         
 Apartado 4: Evolución de la hipótesis en cada caso para llegar     ##
 al mínimo.                                                         ##
######################################################################

#Tomado como referencia batch_grad_descent.m (brindado por el profesor)
minX = min(Xo);
maxX = max(Xo);

areas=linspace(0,maxX,length(Xo));
nareas=nx.transform(areas);



## Método Batch Gradient Descent ## 
An=hypothesis(nareas',thetas_batch(1,:));
A=ny.itransform(An);

figure(6,"name","Evolución de la hipótesis Batch Gradient Descent");
  
plot(nx.itransform(X),ny.itransform(Y),"*b",2);
title({"Evolución de la hipótesis por el método:"," Batch Gradient Descent"},
              "fontsize", 20);
axis([0 maxX 0 1000]);
grid on;
hold on;

plot(areas,A,'k',"linewidth",2);

for (i=[2:rows(thetas_batch)])
  An=hypothesis(nareas',thetas_batch(i,:));
  A=ny.itransform(An);    	
  plot(areas,A,'c',"linewidth",0.5);
  
endfor;

plot(areas,A,'g',"linewidth",3);

################################################################################
################################################################################

## Método Stochastic Gradient Descent ##

An=hypothesis(nareas',thetas_stoch(1,:));
A=ny.itransform(An);

figure(7,"name","Evolución de la hipótesis Stochastic Gradient Descent");
  
plot(nx.itransform(X),ny.itransform(Y),"*b",2);
title({"Evolución de la hipótesis por el método:"," Stochastic Gradient Descent"},
              "fontsize", 20);
axis([0 maxX 0 1000]);
grid on;
hold on;

plot(areas,A,'k',"linewidth",2);

for (i=[2:rows(thetas_stoch)])
  An=hypothesis(nareas',thetas_stoch(i,:));
  A=ny.itransform(An);    	
  plot(areas,A,'c',"linewidth",0.5);
  
endfor;

plot(areas,A,'g',"linewidth",3);


## Método Stochastic Gradient Descent with momentum ## 

An=hypothesis(nareas',thetas_mom(1,:));
A=ny.itransform(An);

figure(8,"name","Evolución de la hipótesis Stochastic Gradient Descent");
  
plot(nx.itransform(X),ny.itransform(Y),"*b",2);
title({"Evolución de la hipótesis por el método:"," Stochastic Gradient Descent with momentum"},
              "fontsize", 20);
axis([0 maxX 0 1000]);
grid on;
hold on;

plot(areas,A,'k',"linewidth",2);

for (i=[2:rows(thetas_mom)])
  An=hypothesis(nareas',thetas_mom(i,:));
  A=ny.itransform(An);    	
  plot(areas,A,'c',"linewidth",0.5);
  
endfor;

plot(areas,A,'g',"linewidth",3);


## Método Stochastic Gradient Descent with RMSprop##

An=hypothesis(nareas',thetas_rms(1,:));
A=ny.itransform(An);

figure(9,"name","Evolución de la hipótesis Stochastic Gradient Descent");
  
plot(nx.itransform(X),ny.itransform(Y),"*b",2);
title({"Evolución de la hipótesis por el método:"," Stochastic Gradient Descent with RMSprop"},
              "fontsize", 20);
axis([0 maxX 0 1000]);
grid on;
hold on;

plot(areas,A,'k',"linewidth",2);

for (i=[2:rows(thetas_rms)])
  An=hypothesis(nareas',thetas_rms(i,:));
  A=ny.itransform(An);    	
  plot(areas,A,'c',"linewidth",0.5);
  
endfor;

plot(areas,A,'g',"linewidth",3);


## Método Stochastic Gradient Descent with ADAM## 

An=hypothesis(nareas',thetas_adam(1,:));
A=ny.itransform(An);

figure(10,"name","Evolución de la hipótesis Stochastic Gradient Descent");
  
plot(nx.itransform(X),ny.itransform(Y),"*b",2);
title({"Evolución de la hipótesis por el método:"," Stochastic Gradient Descent with ADAM"},
              "fontsize", 20);
axis([0 maxX 0 1000]);
grid on;
hold on;

plot(areas,A,'k',"linewidth",2);

for (i=[2:rows(thetas_adam)])
  An=hypothesis(nareas',thetas_adam(i,:));
  A=ny.itransform(An);    	
  plot(areas,A,'c',"linewidth",0.5);
  
endfor;

plot(areas,A,'g',"linewidth",3);


##########################################################################                         
#### Apartado 5: Evolución del error J(?) en función del número de      ##
#### iteración, para varios valores de la tasa de aprendizaje.          ##
##########################################################################

#Método batch gradient descent 

t0 = [-1 -0.2 -0.3];
l_rate=0.01; #tiene que ser menor o igual a 0.01
maxiter=300;
epsilon=0.005;
method="batch";

[thetas_batch,errors_batch]=descentpoly(@loss,@gradloss,t0,X,Y,l_rate,
                            "method",method,
                            "epsilon",epsilon,
                            "maxiter",maxiter);
                           

figure(11,"name","error vs iteracion");


hold on;
iter_batch=[0:1:length(errors_batch)-1]';
plot(iter_batch,errors_batch,'b','linewidth',2);


l_rate=0.005; #tiene que ser menor o igual a 0.01
[thetas_batch,errors_batch]=descentpoly(@loss,@gradloss,t0,X,Y,l_rate,
                            "method",method,
                            "epsilon",epsilon,
                            "maxiter",maxiter);
iter_batch=[0:1:length(errors_batch)-1]';
plot(iter_batch,errors_batch,'r','linewidth',2);


l_rate=0.001; #tiene que ser menor o igual a 0.01
[thetas_batch,errors_batch]=descentpoly(@loss,@gradloss,t0,X,Y,l_rate,
                            "method",method,
                            "epsilon",epsilon,
                            "maxiter",maxiter);
iter_batch=[0:1:length(errors_batch)-1]';
plot(iter_batch,errors_batch,'g','linewidth',2);

l_rate=0.0008; #tiene que ser menor o igual a 0.01
[thetas_batch,errors_batch]=descentpoly(@loss,@gradloss,t0,X,Y,l_rate,
                            "method",method,
                            "epsilon",epsilon,
                            "maxiter",maxiter);
iter_batch=[0:1:length(errors_batch)-1]';
plot(iter_batch,errors_batch,'c','linewidth',2);
legend({'alpha=0.01','alpha=0.005','alpha=0.001','alpha=0.0008'}
                              ,"location","northeastoutside");

title({"Evolución del error J(theta) con distintos ''lr'' por el método:"," Batch Gradient Descent "},
              "fontsize", 15);                              
xlabel('Iterations',"fontsize", 10);
ylabel('Error',"fontsize", 10); 
xlim([0 250]);
ylim([0 12]);
grid on;
hold off;

## Método Stochastic Gradient Descent ##

t0 = [-1 -0.2 -0.3];
l_rate=0.009; #debe ser menor a 0.04
maxiter=500;
epsilon=0.001;
minibatch=20; %%0.5*rows(Xo);

[thetas_stoch,errors_stoch]=descentpoly(@loss,@gradloss,t0,X,Y,l_rate,
                            "method","stochastic",
                            "maxiter",maxiter,
                            "epsilon",epsilon,
                            "minibatch",minibatch);
                  

figure(12,"name","error vs iteracion");
hold on;
iter_stoch=[0:1:length(errors_stoch)-1]';
plot(iter_stoch,errors_stoch,'b','linewidth',2);

l_rate=0.005; 

[thetas_stoch,errors_stoch]=descentpoly(@loss,@gradloss,t0,X,Y,l_rate,
                            "method","stochastic",
                            "maxiter",maxiter,
                            "epsilon",epsilon,
                            "minibatch",minibatch);
iter_stoch=[0:1:length(errors_stoch)-1]';
plot(iter_stoch,errors_stoch,'r','linewidth',2);         
                   
l_rate=0.001; 

[thetas_stoch,errors_stoch]=descentpoly(@loss,@gradloss,t0,X,Y,l_rate,
                            "method","stochastic",
                            "maxiter",maxiter,
                            "epsilon",epsilon,
                            "minibatch",minibatch);
iter_stoch=[0:1:length(errors_stoch)-1]';
plot(iter_stoch,errors_stoch,'g','linewidth',2);    

l_rate=0.0008; 

[thetas_stoch,errors_stoch]=descentpoly(@loss,@gradloss,t0,X,Y,l_rate,
                            "method","stochastic",
                            "maxiter",maxiter,
                            "epsilon",epsilon,
                            "minibatch",minibatch);
iter_stoch=[0:1:length(errors_stoch)-1]';
plot(iter_stoch,errors_stoch,'c','linewidth',2);    


legend({'alpha=0.009','alpha=0.005','alpha=0.001','alpha=0.0008'}
                              ,"location","northeastoutside");

title({"Evolución del error J(theta) con distintos ''lr'' por el método:"," Stochastic Gradient Descent "},
              "fontsize", 15);                              
xlabel('Iterations',"fontsize", 10);
ylabel('Error',"fontsize", 10); 
xlim([0 250]);
ylim([0 40]);
grid on;
hold off;

## Método Stochastic Gradient Descent with momentum ## 

t0 = [-1 -0.2 -0.3];
l_rate=0.009;
maxiter=500;
epsilon=0.001;
minibatch=20;

[thetas_mom,errors_mom]=descentpoly(@loss,@gradloss,t0,X,Y,l_rate,
                            "method","momentum",
                            "maxiter",maxiter,
                            "epsilon",epsilon,
                            "minibatch",minibatch);
                         
figure(13,"name","error vs iteracion");
hold on;
iter_mom=[0:1:length(errors_mom)-1]';
plot(iter_mom,errors_mom,'b','linewidth',2);                          

l_rate=0.005;
[thetas_mom,errors_mom]=descentpoly(@loss,@gradloss,t0,X,Y,l_rate,
                            "method","momentum",
                            "maxiter",maxiter,
                            "epsilon",epsilon,
                            "minibatch",minibatch);
                         
iter_mom=[0:1:length(errors_mom)-1]';
plot(iter_mom,errors_mom,'r','linewidth',2);    


l_rate=0.001;
[thetas_mom,errors_mom]=descentpoly(@loss,@gradloss,t0,X,Y,l_rate,
                            "method","momentum",
                            "maxiter",maxiter,
                            "epsilon",epsilon,
                            "minibatch",minibatch);
                         
iter_mom=[0:1:length(errors_mom)-1]';
plot(iter_mom,errors_mom,'g','linewidth',2);   

 
l_rate=0.0008;
[thetas_mom,errors_mom]=descentpoly(@loss,@gradloss,t0,X,Y,l_rate,
                            "method","momentum",
                            "maxiter",maxiter,
                            "epsilon",epsilon,
                            "minibatch",minibatch);
                         
iter_mom=[0:1:length(errors_mom)-1]';
plot(iter_mom,errors_mom,'c','linewidth',2);    

legend({'alpha=0.009','alpha=0.005','alpha=0.001','alpha=0.0008'}
                              ,"location","northeastoutside");

title({"Evolución del error J(theta) con distintos ''lr'' por el método:",
              " Stochastic Gradient Descent with momentum "},
              "fontsize", 15);                              
xlabel('Iterations',"fontsize", 10);
ylabel('Error',"fontsize", 10); 
xlim([0 250]);
ylim([0 40]);
grid on;
hold off;

## Método Stochastic Gradient Descent with RMSprop ##


t0 = [-1 -0.2 -0.3];
l_rate=0.05;
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
figure(14,"name","error vs iteracion");
hold on;
iter_rms=[0:1:length(errors_rms)-1]';
plot(iter_rms,errors_rms,'b','linewidth',2);                            
                            
l_rate=0.01;   

[thetas_rms,errors_rms]=descentpoly(@loss,@gradloss,t0,X,Y,l_rate,
                            "method","rmsprop",
                            "beta", beta,
                            "beta2", beta2,
                            "maxiter",maxiter,
                            "epsilon",epsilon,
                            "minibatch",minibatch);        
           
iter_rms=[0:1:length(errors_rms)-1]';
plot(iter_rms,errors_rms,'r','linewidth',2);                            
                            
l_rate=0.008;              
[thetas_rms,errors_rms]=descentpoly(@loss,@gradloss,t0,X,Y,l_rate,
                            "method","rmsprop",
                            "beta", beta,
                            "beta2", beta2,
                            "maxiter",maxiter,
                            "epsilon",epsilon,
                            "minibatch",minibatch);        
           
iter_rms=[0:1:length(errors_rms)-1]';
plot(iter_rms,errors_rms,'g','linewidth',2);                            
                            
l_rate=0.004;     

[thetas_rms,errors_rms]=descentpoly(@loss,@gradloss,t0,X,Y,l_rate,
                            "method","rmsprop",
                            "beta", beta,
                            "beta2", beta2,
                            "maxiter",maxiter,
                            "epsilon",epsilon,
                            "minibatch",minibatch);        
           
iter_rms=[0:1:length(errors_rms)-1]';
plot(iter_rms,errors_rms,'c','linewidth',2);                            
                            
 legend({'alpha=0.05','alpha=0.01','alpha=0.008','alpha=0.004'}
                              ,"location","northeastoutside");

title({"Evolución del error J(theta) con distintos ''lr'' por el método:",
              " Stochastic Gradient Descent with RMSprop "},
              "fontsize", 15);                              
xlabel('Iterations',"fontsize", 10);
ylabel('Error',"fontsize", 10); 
xlim([0 200]);
ylim([0 65]);
grid on;
hold off;


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
figure(15,"name","error vs iteracion");
hold on;
iter_adam=[0:1:length(errors_adam)-1]';
plot(iter_adam,errors_adam,'b','linewidth',2);                            
                            
l_rate=0.03;   

[thetas_adam,errors_adam]=descentpoly(@loss,@gradloss,t0,X,Y,l_rate,
                            "method","adam",
                            "beta", beta,
                            "maxiter",maxiter,
                            "epsilon",epsilon,
                            "minibatch",minibatch);                            
iter_adam=[0:1:length(errors_adam)-1]';
plot(iter_adam,errors_adam,'r','linewidth',2);                            
                            
l_rate=0.01;     

[thetas_adam,errors_adam]=descentpoly(@loss,@gradloss,t0,X,Y,l_rate,
                            "method","adam",
                            "beta", beta,
                            "maxiter",maxiter,
                            "epsilon",epsilon,
                            "minibatch",minibatch);                            
iter_adam=[0:1:length(errors_adam)-1]';
plot(iter_adam,errors_adam,'g','linewidth',2);                            
                            
l_rate=0.009;    

[thetas_adam,errors_adam]=descentpoly(@loss,@gradloss,t0,X,Y,l_rate,
                            "method","adam",
                            "beta", beta,
                            "maxiter",maxiter,
                            "epsilon",epsilon,
                            "minibatch",minibatch);                            
iter_adam=[0:1:length(errors_adam)-1]';
plot(iter_adam,errors_adam,'c','linewidth',2);                            
                            
 legend({'alpha=0.05','alpha=0.03','alpha=0.01','alpha=0.009'}
                              ,"location","northeastoutside");

title({"Evolución del error J(theta) con distintos ''lr'' por el método:",
              " Stochastic Gradient Descent with ADAM "},
              "fontsize", 15);                              
xlabel('Iterations',"fontsize", 10);
ylabel('Error',"fontsize", 10); 
xlim([0 100]);
ylim([0 65]);
grid on;
hold off;                            
#### Apartado 7: 

