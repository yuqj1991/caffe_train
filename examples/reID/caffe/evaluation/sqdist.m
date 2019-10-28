function m = sqdist(p, q)
% SQDIST      Squared Euclidean or Mahalanobis distance.
% SQDIST(p,q)   returns m(i,j) = (p(:,i) - q(:,j))'*(p(:,i) - q(:,j)).
% The Lightspeed Matlab toolbox
% Written by Tom Minka

[d, pn] = size(p);
[d, qn] = size(q);

if pn == 0 || qn == 0
  m = zeros(pn,qn);
  return
end

pmag = sum(p .* p);
qmag = sum(q .* q);
m = repmat(qmag, pn, 1) + repmat(pmag', 1, qn) - 2*p'*q;
end
