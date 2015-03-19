function dict=get_phn_idx_dict()

dict = containers.Map; 
phonemes_vec ={'h#','sh','ix','hv','eh','dcl','jh','ih','d','ah',...
               'kcl','k','s','ux','q','en','gcl','g','r','w','ao',...
               'epi','dx','axr','l','y','uh','n','ae','m','oy','ax',...
               'dh','tcl','iy','v','f','t','pcl','ow','hh','ch','bcl',...
               'b','aa','em','ng','ay','th','ax-h','ey','p','aw','er',...
               'nx','z','el','uw','pau','zh','eng'};

n_phonemes = size(phonemes_vec,2);

for i=1:n_phonemes
   dict(phonemes_vec{i})=i;
   
end


end