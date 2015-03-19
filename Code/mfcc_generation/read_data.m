function [utterances,utterances_phn_tags,sorted_frames]=read_data(wav_files,phn_files)
% Reads the data from the speech and phonemes paths.
n_utr = size(wav_files,1);

%Sorted frames by phonemes
sorted_frames = containers.Map;
utterances=cell(n_utr,1);
utterances_phn_tags=cell(n_utr,1);
phn_idx_dict = get_phn_idx_dict();



%phonemes=cell(n_utr,1);
%sorted_utr_by_phn=cell(n_phn,1);




for i=1:n_utr
    %Read phonemes data from .phn
    %phonemes_info=dlmread(phn_files{i},' ');
    f = fopen(phn_files{i});
    phonemes_info = textscan(f,'%d %d %s','Delimiter',' ');
    fclose(f);
    
    start_info  = phonemes_info{1};
    end_info    = phonemes_info{2};
    labels_info = phonemes_info{3};
    
    
    % Read the wav audio file
    [y,~]=audioread(wav_files{i});
    
    %Get the MFCC
    %To nearly replicate htk mfcc
    %     'lifterexp': -22
    %        'nbands': 20
    %       'maxfreq': 8000
    %      'sumpower': 0
    %        'fbtype': 'htkmel'
    %       'dcttype': 3
    
    sr=16000;
    utr_i = melfcc(y,sr,'lifterexp',-22,'nbands',20,'maxfreq',8000,...
                        'sumpower',0,'fbtype','htkmel','dcttype',3);
    utterances{i}=utr_i;
    n_frames=size(utr_i,2);
    
      
    n_segments=size(labels_info,1);
    
    last_label_idx = 0;
    utr_info=zeros(size(labels_info));
    
    for j=1:n_frames
        displacement = start_info(1);
        % 0.01 * 16Khz number of utterances in samples
        % reference is the initial frame time
        sample=repmat(displacement+(j-1)*160,n_segments,1);
        label_idx=find(sample >= start_info,1,'last');
        
        phn_lbl_cell = labels_info(label_idx);
        phn_lbl = phn_lbl_cell{1};
        
        %map the phn label string to phn idx
        phn_idx_lbl = phn_idx_dict(phn_lbl);
        
        if ( last_label_idx ~= label_idx )
            if j~=1
                utr_info(last_label_idx,2)=j-1;
            end
            utr_info(label_idx,1) = j;
            utr_info(label_idx,3) = phn_idx_lbl;
            
            if sorted_frames.isKey(phn_lbl)
                sqces_phn_cell = sorted_frames(phn_lbl);
                sqces_phn_cell=[sqces_phn_cell utr_i(:,j)];
                sorted_frames(phn_lbl)=sqces_phn_cell;
                
%                 stacked_phn_cell = stacked_sorted_frames(phn_lbl);
%                 stacked_sorted_frames(phn_lbl)={[stacked_phn_cell{1};utr_i(j,:)]};
%                 
            else
                sorted_frames(phn_lbl)={utr_i(:,j)};
%                 stacked_sorted_frames(phn_lbl)={utr_i(j,:)};
            end
           
            
        else
            if j == n_frames
                utr_info(last_label_idx,2)=j;
            end
            
            sqces_phn_cell = sorted_frames(phn_lbl);
            sqces_phn_cell{end}=[sqces_phn_cell{end} utr_i(:,j)];
            sorted_frames(phn_lbl)=sqces_phn_cell;
            
%             %stacked_sorted_frames(phn_lbl)={utr_i(j,:)};
%             stacked_phn_cell = stacked_sorted_frames(phn_lbl);
%             stacked_sorted_frames(phn_lbl)={[stacked_phn_cell{1};utr_i(j,:)]};
        end
                
        last_label_idx=label_idx;
    end
    utterances_phn_tags{i}=utr_info;
end

