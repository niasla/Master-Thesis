function [utterances_lbl] = label_frames(HMM_Models,utterances,utterances_info)

n_utr = size(utterances,1);
utterances_lbl = cell(n_utr,1);

for utr_idx=1:n_utr
    utr = utterances{utr_idx};
    frame_lbls = zeros(size(utr,2),1);
    utr_info = utterances_info{utr_idx};
    n_seqcs = size(utr_info,1);
    for seq_idx=1:n_seqcs
        seq_start = utr_info(seq_idx,1);
        seq_end   = utr_info(seq_idx,2);
        seq_lbl_idx   = utr_info(seq_idx,3);
       
        if seq_start ~= 0 && seq_end ~=0
            seq = utr(:,seq_start:seq_end);
            HMM_Model = HMM_Models{seq_lbl_idx};
            if ~isempty(HMM_Model)
                B = mixgauss_prob(seq, HMM_Model.mu, HMM_Model.sigma, HMM_Model.mixmat);
                path = viterbi_path(HMM_Model.pi, HMM_Model.A, B);
                frame_lbls(seq_start:seq_end) = 3*(seq_lbl_idx-1)+path;
            end
        end
       
    end
    utterances_lbl{utr_idx} = frame_lbls;
    
    
end

end