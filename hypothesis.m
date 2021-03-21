% Evaluate the hypothesis with all given x
function y=hypothesis(x,theta)
  orden=columns(theta)-1;
  x_poly = bsxfun(@power,x,(0:orden));
  y = x_poly*theta;
endfunction;
