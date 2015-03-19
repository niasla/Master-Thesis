function [p , alpha] = HMMdecode(A, B, pi, seq, alphabet)
    
    
    seq_idx=symbols_to_idx(seq,alphabet);
    
    %sequence long.
    T = size(seq_idx,2);
    
    %number of states and number of observations
    N=size(A,1);
    alpha=zeros(T,N);
    
    %init
    alpha(1,:)=pi.*B(:,seq_idx(1))';
    
    %induction step
    for t=2:T
        alpha(t,:) = (alpha(t-1,:)*A)'.*B(:,seq_idx(t));
    end
    
    %sum over all possible end states at T
    p=sum(alpha(T,:),2);
    alpha=alpha';%[pi' alpha'];

end