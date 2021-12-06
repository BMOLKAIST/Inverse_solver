#include"static_parameter.h"

// original code inspired from : https://www.mathworks.com/matlabcentral/fileexchange/38034-mexthread

#include <mex.h>
#include <vector>
#include <string>
#include <map>


//! Converts an mxArray to a unisgned int value
void exact_convert(const mxArray* ma, uint32_t& scalar)
{
    mxClassID classid = mxGetClassID(ma);
    switch (classid)
    {
    case mxUINT32_CLASS:
        scalar = (uint32_t)((uint32_t*)mxGetData(ma))[0];
        break;
    default:
        mexErrMsgTxt("Not a supported mxClassID");
    }
}


void convert(const std::map<std::string, mxArray*>& map_out, mxArray*& ma) {
    //step 1 retrieve the field
    int sz = map_out.size();
    const char** fieldnames = new const char*[sz];
    size_t curr_pos = 0;
    for (auto it = map_out.begin(); it != map_out.end(); it++)
    {
        fieldnames[curr_pos] = it->first.c_str();
        curr_pos++;
    }
    mwSize dims = 1;
    //step 2 create the array
    ma = mxCreateStructArray(
        1, &dims, map_out.size(), fieldnames);
    curr_pos = 0;
    for (auto it = map_out.begin(); it != map_out.end(); it++)
    {
        mxSetFieldByNumber(ma, 0, curr_pos, it->second);
        curr_pos++;
    }
    delete[] fieldnames;
}