function seq_idx=symbols_to_idx(seq,alphabet)

T=size(seq,2);

seq_idx=zeros(1,T);

for t=1:T
    [~,seq_idx(t)]=ismember(seq{t},alphabet);
end