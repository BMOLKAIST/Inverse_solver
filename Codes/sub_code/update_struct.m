function updated_struct=update_struct(base_struct,update)

updated_struct=base_struct;

fields=fieldnames(update);

for ii=1:length(fields)
    field=fields{ii};
    if isfield(base_struct,field)
        updated_struct.(field)=update.(field);
    else
        warning(['Unexpected field : "' field '" was set ?! Verify the field name of your parameter structures.']);
        updated_struct.(field)=update.(field);
    end
end
end