%Put here the path to the directory to the folder containing both folders
% test and train in timit
warning('off','all');
restoredefaultpath;

root_path = getenv('TIMIT');

tst_path = [root_path '/test'];
train_path = [root_path '/train'];


addpath( genpath([getenv('MT_ROOT') '/ToolBox/rastamat']) );

mat_files_path = [getenv('MT_ROOT') '/Code/mfcc_generation/mat_files'];


disp('Reading files paths...')

%Intentionally processing in splitted way
[wav_files_tr,phn_files_tr] = read_files(train_path);
[wav_files_tst,phn_files_tst] = read_files(tst_path);

all_wav_files = [wav_files_tr;wav_files_tst];
all_phn_files = [phn_files_tr;phn_files_tst];

if ~exist(mat_files_path,'dir')
    mkdir(mat_files_path);   
    
    disp('Processing  wav files...')
    [utterances,utterances_phn_tags,sorted_frames]=read_data(all_wav_files,all_phn_files);
    %[utterances_tst,utterances_phn_tags_tst,sorted_frames_tst]=read_data(wav_files_tst(1:50),phn_files_tst(1:50));
    
    save([mat_files_path '/sorted_frames.mat'],'sorted_frames');
    save([mat_files_path '/utterances.mat'],'utterances');
    save([mat_files_path '/utterances_phn_tags.mat'],'utterances_phn_tags');
    
%     save([mat_files_path '/sorted_frames_tst.mat'],'sorted_frames_tst');
%     save([mat_files_path '/utterances_tst.mat'],'utterances_tst');
%     save([mat_files_path '/utterances_phn_tags_tst.mat'],'utterances_phn_tags_tst');
else
    load([mat_files_path '/sorted_frames.mat']);
    load([mat_files_path '/utterances.mat']);
    load([mat_files_path '/utterances_phn_tags.mat']);
    
%     load([mat_files_path '/sorted_frames_tst.mat']);
%     load([mat_files_path '/utterances_tst.mat']);
%     load([mat_files_path '/utterances_phn_tags_tst.mat']);
end

%Train 3 state HMM Model for phoneme alignment
addpath( genpath([getenv('MT_ROOT') '/ToolBox/HMMall']) );
mixtures_vec=[16 32 64 128];
mixtures_vec_sz=size(mixtures_vec,2);
n_states = 3;

for mix_idx=1:mixtures_vec_sz
    n_mixtures=mixtures_vec(mix_idx);
    if ~exist([mat_files_path sprintf('/HMM_Models_%d_mixtures.mat',n_mixtures)],'file')
        fprintf('Training HMMs with %d mixtures per state...\n',n_mixtures);
        HMM_Models = estimate_phoneme_HMM(sorted_frames,n_states,n_mixtures);
        fprintf('Saving HMM model with %d mixtures per state...\n',n_mixtures);
        save([mat_files_path sprintf('/HMM_Models_%d_mixtures.mat',n_mixtures)],'HMM_Models');
    else
        fprintf('Loading saved HMMs with %d mixtures per state...\n',n_mixtures);
        load([mat_files_path sprintf('/HMM_Models_%d_mixtures.mat',n_mixtures)]);
    end
  
    
    if ~exist([mat_files_path sprintf('/utterances_lbl_%d_mixtures.mat',n_mixtures)],'file')
        fprintf('Labeling frames using HMM model with %d mixtures per state...\n',n_mixtures);
        [utterances_lbl] = label_frames(HMM_Models,utterances,utterances_phn_tags);
        save([mat_files_path sprintf('/utterances_lbl_%d_mixtures.mat',n_mixtures)],'utterances_lbl');
    else
        fprintf('Loading saved utterances labels with %d mixtures per state...\n',n_mixtures);
        load([mat_files_path sprintf('/utterances_lbl_%d_mixtures.mat',n_mixtures)]);
    end
        fprintf('Writing mfcc files corresponding to %d mixtures per state...\n',n_mixtures);
        write_mfcc_files(all_wav_files, utterances ,utterances_lbl,n_mixtures);
    
end


% write_mfcc_files(wav_files_tr, utterances_tr ,sorted_frames_tr);
% write_mfcc_files(wav_files_tst, utterances_tst ,sorted_frames_tst);

rmpath( genpath([getenv('MT_ROOT') '/ToolBox/HMMall']) );
rmpath( genpath([getenv('MT_ROOT') '/ToolBox/rastamat']) );
