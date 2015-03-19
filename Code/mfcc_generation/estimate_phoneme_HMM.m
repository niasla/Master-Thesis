function HMM_Models = estimate_phoneme_HMM(sorted_frames,n_states,n_mixtures)
%Generates HMM Models for each type of phoneme to be used for labeling 
%the 3-state phonemes by forced alignment 

phonemes_vec ={'h#','sh','ix','hv','eh','dcl','jh','ih','d','ah',...
               'kcl','k','s','ux','q','en','gcl','g','r','w','ao',...
               'epi','dx','axr','l','y','uh','n','ae','m','oy','ax',...
               'dh','tcl','iy','v','f','t','pcl','ow','hh','ch','bcl',...
               'b','aa','em','ng','ay','th','ax-h','ey','p','aw','er',...
               'nx','z','el','uw','pau','zh','eng'};

n_phonemes = size(phonemes_vec,2);

HMM_Models = cell(n_phonemes,1);
d= 13;
pi0 = [1 0 0];
A0 =  [0.4 0.6 0;
       0 0.4 0.6;
       0 0 1];


for phn_idx=1:n_phonemes
    if sorted_frames.isKey(phonemes_vec{phn_idx})
        phn_seqs = sorted_frames(phonemes_vec{phn_idx});
        %n_seqs = size(phn_seqs,2);
        
        data = cell2mat(phn_seqs);
        
                
       
        [mu0, Sigma0] = mixgauss_init(n_states*n_mixtures, data, 'diag');
        
        mu0 = reshape(mu0, [d n_states n_mixtures]);
        Sigma0 = reshape(Sigma0, [d d n_states n_mixtures]);
        mixmat0 = mk_stochastic(rand(n_states,n_mixtures));
        
        [LL, prior1, transmat1, mu1, Sigma1, mixmat1] = ...
    mhmm_em(phn_seqs, pi0, A0, mu0, Sigma0, mixmat0, 'max_iter', 100);
        %transmat1
        
        hmm_params.pi =prior1;
        hmm_params.A  =transmat1;
        hmm_params.mu = mu1;
        hmm_params.sigma = Sigma1;
        hmm_params.mixmat=mixmat1;
        
        HMM_Models{phn_idx} = hmm_params;
        
        
%         HMM_Models{phn_idx} = hmmFit(phn_seqs{1}, n_states, 'mixGaussTied', ...
%             'verbose', true, 'maxiter', 5, ...
%             'nmix', n_mixtures,'pi0',pi0,'trans0',A0,'piPrior',pi0);%,...
%             %'transPrior',A0);%,'emissionPrior',0); 
%     end
%     for seq_idx=1:n_seqs
%         seq = phn_seqs{seq_idx};
%     
%     
    end 
end
           
           
           
end