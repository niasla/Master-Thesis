function [A, B, pi] = HMMtrain(A_hat, B_hat, pi_hat, seq, alphabet ,iter_t)

 % a->1 b->2 c->3 d->4   
seq_idx=symbols_to_idx(seq,alphabet);

A=A_hat;
B=B_hat;
pi=pi_hat;

%sequence long.
T = size(seq_idx,2);
N = size(A,2);
K = size(B,2);

epsilon=zeros(N,N,T-1);


for i=1:iter_t
   [~ , alpha] = forward(A, B, pi, seq, alphabet);
   [~ , beta] = backward(A, B, pi, seq, alphabet);
   
      
   for t=1:T-1
    epsilon(:,:,t) = repmat(alpha(:,t),1,N).*A.*...
                     repmat(B(:,seq_idx(t+1))',N,1).*...
                     repmat(beta(:,t+1)',N,1);
    %normalize
    epsilon(:,:,t) = epsilon(:,:,t)/sum(sum(epsilon(:,:,t),2),1);  
   end
   
   %calc Gamma
   gamma = alpha.*beta;
   gamma = gamma./repmat(sum(gamma,1),N,1);
   
   
   
   A=sum(epsilon,3)./repmat(sum(gamma(:,1:end-1),2),1,N);
   
   pi=gamma(:,1)';
   
   s=sum(gamma,2);
   for k=1:K
    mask=(seq_idx==k);   
    B(:,k)=sum(repmat(mask,N,1).*gamma,2)./s;
   end
   

   
end



