function m = cal_dis(p, q, str)
% SQDIST      Squared Euclidean or Mahalanobis distance.
% SQDIST(p,q)   returns m(i,j) = (p(:,i) - q(:,j))'*(p(:,i) - q(:,j)).
% SQDIST(p,q,A) returns m(i,j) = (p(:,i) - q(:,j))'*A*(p(:,i) - q(:,j)).
% The Lightspeed Matlab toolbox
% Written by Tom Minka

[d, pn] = size(p);
[d, qn] = size(q);
assert (pn ~= 0 && qn ~= 0);

pmag = sum(p .* p, 1);
qmag = sum(q .* q, 1);
if (strcmp(str, 'Euc') == 1)
    m = repmat(qmag, pn, 1) + repmat(pmag', 1, qn) - 2*p'*q;
elseif (strcmp(str, 'Cos') == 1)
    dis = p' * q;
    pmag = sqrt(pmag);
    qmag = sqrt(qmag);
    m = dis ./ repmat(qmag, pn, 1) ./ repmat(pmag', 1, qn);
else
    error(['Unknow distance type ', str]);
end
