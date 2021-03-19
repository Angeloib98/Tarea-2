%% Gradient of the loss function
function res=gradloss(theta,X,Y)
  orden = columns(theta)-1;
  X_poly=bsxfun(@power,X,(0:orden));
  res = (X_poly'*(X_poly*theta'-Y*ones(1,rows(theta))))';
endfunction;
