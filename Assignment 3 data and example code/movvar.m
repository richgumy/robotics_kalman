function [var_] = movvar(A,k)
% A is sample data, k must be an even number
    if mod(k,2)
        error('Parameter, k, for movvar(A,k) must be even')
    end
    var_ = zeros(size(A));
    for i=1:length(A-k)
        lower_lim = i-k/2;
        upper_lim = i+k/2;
        if lower_lim <= 0
            lower_lim = 1;
        end
        if upper_lim >= length(A)
            upper_lim = length(A);
        end
        strip = A(lower_lim:upper_lim);
        for j=1:k
            var_(i) = var(strip);
        end
    end
end