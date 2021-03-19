% Evaluate the hypothesis with all given x
function y=hypothesis(x,theta)
  
  ## Put your code in here
  x_poly = bsxfun(@power,x,0:length(theta)-1);
  y = x_poly*theta;
endfunction;
