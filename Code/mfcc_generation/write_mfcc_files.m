function write_mfcc_files(files_path, frames, phonemes,n_mixtures)

n_files = size(files_path,1);


%Intentionally writing line by line because of formats (double and int)
for i=1:n_files
    frame_i = frames{i}';
    phoneme_i = phonemes{i};
    nlines = size(frame_i,1);
    str = sprintf('_mix_%d.mfcc',n_mixtures);
    f = fopen(char(strrep(files_path(i), '.wav', str)),'w');
    for j=1:nlines
        fprintf(f,'%f %f %f %f %f %f %f %f %f %f %f %f %f %d\n',...
              frame_i(j,1),frame_i(j,2),frame_i(j,3),frame_i(j,4),...
              frame_i(j,5),frame_i(j,6),frame_i(j,7),frame_i(j,8),...
              frame_i(j,9),frame_i(j,10),frame_i(j,11),frame_i(j,12),...
              frame_i(j,13),phoneme_i(j));
    end
    fclose(f);
end