function [V]=vif(X)

R0 = corrcoef(X);
V = diag(inv(R0))';