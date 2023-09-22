function [theta, Pn, Rn] = rls(theta_,Pn_,Rn_,Yn ,Zn)
Rn=Rn_;
 
Num = Rn_+Zn'*Pn_*Zn;
Pn = 1/Rn_*(( Pn_ - (Pn_ * Zn * (Zn)' * Pn_) /Num) );
Ln = (Pn_*Zn) /Num;
En = Yn - Zn' * theta_;
 
theta = theta_ + Ln * En;
 
end