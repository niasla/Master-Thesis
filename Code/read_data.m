function [audio,phonemes,prior_prob]=read_data(sound_files,phonemes_files,n_data,n_phonemes)
% Reads the data from the speech and phonemes paths.

rnd_idx=randperm(size(sound_files,1));
indices=rnd_idx(1:n_data);
audio=cell(n_data,1);
phonemes=cell(n_data,1);
prior_prob=zeros(n_phonemes,1);


for i=1:n_data
    %Read phonemes data
    phonemes_vec=dlmread(phonemes_files{indices(i)},' ');
    phonemes{i}=phonemes_vec;
    
    % Read the wav audio file
    [y,Fs]=audioread(sound_files{indices(i)});
    %Get the MFCC
    [cepstra,~,~] = melfcc(y);
    
    spx_data.audio=y;
    spx_data.Fs=Fs;
    spx_data.mfcc=cepstra';
    n_cepstra=size(cepstra,2);
    m=mean(spx_data.mfcc,1);
    variance=var(spx_data.mfcc);
    spx_data.norm_mfcc=(spx_data.mfcc-repmat(m,n_cepstra,1))./repmat(variance,n_cepstra,1);
    
    
    lbls=ones(n_cepstra,1);
    n_segments=size(phonemes_vec,1);
    for j=1:n_cepstra
        sample_idx=repmat((j-1)*0.01*Fs,n_segments,1);
        idx=find(sample_idx >= phonemes_vec(:,1),1,'last');
        if ~isempty(idx)
            lbls(j)=phonemes_vec(idx,3)+1;
        end
%         if(lbls(j)>60)
%                j
%            sample_idx
%            idx
%            phonemes_vec
%            pause;
%         end
        prior_prob(lbls(j))=prior_prob(lbls(j))+1;
    end
    spx_data.labels=lbls;
    audio{i}=spx_data;
end
prior_prob=prior_prob/sum(prior_prob,1);
