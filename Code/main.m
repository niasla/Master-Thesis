clear all;
training_path='/home/nizar/timit/timit/timit/train';
test_path='/home/nizar/timit/timit/timit/test';
[spx_f_training,phn_f_training] = read_files(training_path);
[spx_f_test,phn_f_test] = read_files(test_path);
n_phonemes=61;
%Conext Independent (Temporary)
n_states=n_phonemes;
phonemes_labels_vec=eye(n_phonemes);
%% Setting/Creating the training and test data
training_sz=120;
test_sz=25;

tic;
[spx_training,phn_training,phoneme_prior]=...
            read_data(spx_f_training,phn_f_training,training_sz,n_phonemes);

%GPU
%spx_training = gpuArray(spx_tr);
%phn_training = gpuArray(phn_tr)
%phoneme_prior = gpuArray(phoneme_prior)


[spx_test,phn_test,~]=read_data(spx_f_test,phn_f_test,training_sz,n_phonemes);
t1=toc;
fprintf('Elapse Time Reading Data(s): %u\n',t1);

%pause
%% Create and set a DBN-DNN
n_units=2048;
layers=[n_units n_units n_units];
%DBN setup
dbn.sizes = layers;
opts.numepochs =   1;
opts.momentum  =   0;
opts.alpha     =   1;
%specify the use of GRBM as first layer
opts.grbm = 1;
dbn = dbnsetup(dbn, spx_training{1}.norm_mfcc, opts);

%% DBN Train

% TODO , CHECK GRBM!!!!
% TODO , Multiple Frames as input
tic;
for i=1:training_sz
    opts.batchsize =   size(spx_training{i}.mfcc,1);
    dbn = dbntrain(dbn, spx_training{i}.mfcc, opts);   
end
t2=toc;
fprintf('Elapse Time Pre-Training (s): %u\n',t2);


%% Unfold DBN to DNN and fine-tune the DNN
tic;
nn = dbnunfoldtonn(dbn, n_states);
nn.activation_function = 'sigm'; %or sigm also
nn.output='sigm';

%train nn
opts.numepochs =  3;
for i=1:training_sz
    opts.batchsize = size(spx_training{i}.mfcc,1);
    nn = nntrain(nn, spx_training{i}.mfcc,...
        phonemes_labels_vec(:,spx_training{i}.labels)', opts);
end
%[er, bad] = nntest(nn, test_x, test_y);
t3=toc;
fprintf('Elapse Time Fine-Tuning (s): %u\n',t3);
%% Create and train HMM to Hybrid with DBN-DNN
