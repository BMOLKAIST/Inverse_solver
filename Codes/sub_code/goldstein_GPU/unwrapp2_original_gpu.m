function unwrapped_phase = unwrapp2_gpu(wrapped_phase)
residue_map = get_residue_gpu(wrapped_phase);
[residue_map,step1_unwrapp,step2_unwrapp]=remove_dipoles_gpu(residue_map,int32(find(residue_map(:)>0))); %also need the position of positive residues
[step1_unwrapp,step2_unwrapp]=goldstein_branch_cut_original(residue_map,step1_unwrapp,step2_unwrapp);
unwrapped_phase=unwrapp_phase_goldstein_gpu(wrapped_phase,step1_unwrapp,step2_unwrapp);
end