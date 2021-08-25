function [residue_idx,residue_value,residue_map] = get_residue(wrapped_phase)

%{
error('use the coda code get_residue_gpu which has the same output but is 4 time faster');

dx=wrapped_phase(1:size(wrapped_phase,1)-1,1:size(wrapped_phase,2),:)...
    -wrapped_phase(2:size(wrapped_phase,1),1:size(wrapped_phase,2),:);
dy=wrapped_phase(1:size(wrapped_phase,1),1:size(wrapped_phase,2)-1,:)...
    -wrapped_phase(1:size(wrapped_phase,1),2:size(wrapped_phase,2),:);

dx=mod(dx+pi,2*pi)-pi;
dy=mod(dy+pi,2*pi)-pi;

residue_map=dx(1:size(dx,1),1:size(dx,2)-1,:)...
    -dx(1:size(dx,1),2:size(dx,2),:)...
    -dy(1:size(dy,1)-1,1:size(dy,2),:)...
    +dy(2:size(dy,1),1:size(dy,2),:);

residue_map=residue_map./(2*pi);

residue_map=round(residue_map);
%}
residue_map=get_residue_gpu(wrapped_phase);

residue_idx=find(residue_map);

residue_value=residue_map(residue_idx);

end