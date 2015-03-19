function [p , path] = HMMviterbi(A, B, pi, seq, alphabet)

seq_idx=symbols_to_idx(seq,alphabet);

%sequence long.
T = size(seq_idx,2);
N=size(B,1);

delta = zeros(N,T);
path = zeros(1,T) ;

%init
delta(:,1)=pi'.*B(:,seq_idx(1));
[~,path(1)]=max(delta(:,1));
for t=2:T
     trans_mat = repmat(delta(:,t-1),1,N).*A;
     [max_p, ~] = max(trans_mat,[],1);
     delta(:,t)=max_p'.*B(:,seq_idx(t));
     [~,path(t)]=max(delta(:,t));
end

p=max(delta(:,T));