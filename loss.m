%% Loss function
%% Objective function of the parameters theta using the data and labels
function res=loss(theta,X,Y)
  orden=columns(theta)-1;
  X_poly=bsxfun(@power,X,(0:orden));
  Residuo = (X_poly*theta' - Y*ones(1,rows(theta)));
  res = 0.5*sum(Residuo.*Residuo,1)';
endfunction;
