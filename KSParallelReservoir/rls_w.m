function [w_out, Pn, Rn] = rls_w(wout,Pn_,Rn_, yn , x, output)
Rn=Rn_;

Num = Rn_+x'*Pn_*x;
% Pn = 1/Rn_*(( Pn_ - (Pn_ * x * (x)' * Pn_.') /Num) );
Px = (Pn_*x);
% Px = (Pn_*x) / Num;
En = mean(yn - output, 2);
w_out = wout + En * Px.';

Pn = 1/Rn_*(( Pn_ - (Pn_ * x * (x)' * Pn_.') /Num) );
% e_before = z - target;
% % display(e_before(1));
% Pr = P * rt;
% dw = beta * Pr .* e_before.';
% wreadout = wreadout - dw;
% 
% P = P - (Pr*Pr.') / (1 + rt.' * Pr);
% 
% e_after = wreadout.' * rt - target;
end