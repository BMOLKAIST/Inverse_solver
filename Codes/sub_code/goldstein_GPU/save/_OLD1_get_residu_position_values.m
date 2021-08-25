function [residue_idx,residue_value] = get_residu_position_values(residue_map)

residue_idx=find(residue_map);

residue_value=residue_map(residue_idx);

end