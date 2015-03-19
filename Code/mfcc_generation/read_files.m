function [wav_files,phn_files] = read_files(path)
wav_files={};
phn_files={};
 
dr_list=dir(path);
%removing . and  ..
dr_list=dr_list(3:end);
dr_size=size(dr_list,1);

for dr=1:dr_size
    %female and males folders
    temp_path=[path '/' dr_list(dr).name];
    sex_list=dir(temp_path);
    sex_list=sex_list(3:end);
    sex_size=size(sex_list,1);
    
    for sx=1:sex_size
        local_path=[temp_path '/' sex_list(sx).name '/'];
        wav_files_list=dir([local_path '*.wav']);
        phn_files_list=dir([local_path '*.phn']);
        
        %It's the same size of both types
        files_n=size(wav_files_list,1);
        local_wav=cell(files_n,1);
        local_lbl=cell(files_n,1);
        for j=1:files_n
            local_wav{j}=[local_path wav_files_list(j).name];
            local_lbl{j}=[local_path phn_files_list(j).name];
        end
        wav_files=[wav_files;local_wav];
        phn_files=[phn_files;local_lbl];
    end
end

