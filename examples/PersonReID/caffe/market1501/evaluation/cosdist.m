function m = cosdist(p, q)
% COSTDIST      cosine distance
% The Lightspeed Matlab toolbox
% Written by Tom Minka

[dim, pn] = size(p);
[dim, qn] = size(q);
assert (size(p,1) == size(q,1));

if pn == 0 || qn == 0
  m = zeros(pn,qn);
  return
end

  
p = p ./ repmat(sqrt(sum(p.*p)), dim, 1);
q = q ./ repmat(sqrt(sum(q.*q)), dim, 1);
%m = p' * q;
m = sqdist(p, q);
end
