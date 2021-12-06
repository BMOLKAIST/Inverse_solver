#ifndef MEX_UTIL_H
#define MEX_UTIL_H

#include"static_parameter.h"
#include "static_parameter.h"
#include <mex.h>
#include <vector>
#include <string>
#include <map>

//convert methods
template <typename T> void convert(const mxArray* ma, T& scalar);
template <typename T> void convert(const T& scalar, mxArray*& ma);
void convert(const std::map<std::string, mxArray*>& map_out, mxArray*& ma);
//exacy convert
void exact_convert(const mxArray* ma, uint32_t& scalar);
//create methods

//search and set with error throwing for structures
template<typename T> void search_and_set(const std::map<std::string,const mxArray*>& map_in,const std::string index,T& output) {
    auto search = map_in.find(index);
    if (search != map_in.end()) {
        convert(search->second, output);
    }
    else {
        std::string msg_err = std::string("Parameter ") + index + std::string(" not found");
        mexPrintf(msg_err.c_str());
        throw(std::runtime_error(msg_err));
    }
}

//template definitions
template <typename T> mxClassID get_mex_class(){
    mxClassID classid = mxUNKNOWN_CLASS;
    if (std::is_same<T, unsigned char>::value) classid = mxUINT8_CLASS;
    else if (std::is_same<T, double>::value) classid = mxDOUBLE_CLASS;
    else if (std::is_same<T, float>::value) classid = mxSINGLE_CLASS;
    else if (std::is_same<T, int32_t>::value) classid = mxINT32_CLASS;
    else if (std::is_same<T, int16_t>::value) classid = mxINT16_CLASS;
    else if (std::is_same<T, uint32_t>::value) classid = mxUINT32_CLASS;
    else if (std::is_same<T, uint16_t>::value) classid = mxUINT16_CLASS;
    else if (std::is_same<T, bool>::value) classid = mxLOGICAL_CLASS;
    return classid;
}
template <typename T> void convert(const T& scalar, mxArray*& ma) {
    mxClassID classid = get_mex_class<T>();
    if (classid == mxUNKNOWN_CLASS) {
        mexErrMsgTxt("Not a supported mxClassID");
    }
    ma = mxCreateNumericMatrix((mwSize)1, (mwSize)1, classid, mxREAL);
    ((T*)mxGetPr(ma))[0] = scalar;
}
template <typename T2, std::size_t N> void convert(const std::array<T2, N>& scalar, mxArray*& ma) {
    mxClassID classid = get_mex_class<T2>();
    if (classid == mxUNKNOWN_CLASS) {
        mexErrMsgTxt("Not a supported mxClassID");
    }
    ma = mxCreateNumericMatrix((mwSize)N, (mwSize)1, classid, mxREAL);
    std::copy(scalar.begin(), scalar.end(), ((T2*)mxGetPr(ma)));
}
template <typename T> void convert(const mxArray* ma, T& scalar)
{
    mxClassID classid = mxGetClassID(ma);
    switch (classid)
    {
    case mxUINT8_CLASS:
        scalar = (T)((unsigned char*)mxGetUint8s(ma))[0];
        break;
    case mxDOUBLE_CLASS:
        scalar = (T)((double*)mxGetDoubles(ma))[0];
        break;
    case mxSINGLE_CLASS:
        scalar = (T)((float*)mxGetSingles(ma))[0];
        break;
    case mxINT32_CLASS:
        scalar = (T)((int32_t*)mxGetInt32s(ma))[0];
        break;
    case mxINT16_CLASS:
        scalar = (T)((int16_t*)mxGetInt16s(ma))[0];
        break;
    case mxUINT32_CLASS:
        scalar = (T)((uint32_t*)mxGetUint32s(ma))[0];
        break;
    case mxUINT16_CLASS:
        scalar = (T)((uint16_t*)mxGetUint16s(ma))[0];
        break;
    case mxLOGICAL_CLASS:
        scalar = (T)*mxGetLogicals(ma);
        break;
    default:
        mexErrMsgTxt("Not a supported mxClassID");
    }
}
template <typename T> void convert(const mxArray* ma, std::vector<T>& array)
{
    if (display_debug) mexPrintf("convert to vect");
    size_t ellement_num = mxGetNumberOfElements(ma);
    
    if (display_debug) mexPrintf("el : %d",ellement_num);
    array=std::vector<T>((size_t)ellement_num);

    mxClassID classid = mxGetClassID(ma);
    switch (classid)
    {
    case mxUINT8_CLASS:
    {
        auto ptr = ((unsigned char*)mxGetUint8s(ma));
        for (size_t i = 0; i < ellement_num; i++)
        {
            array[i]=ptr[i];
        }
        break;
    }
    case mxDOUBLE_CLASS:
    {
        auto ptr = ((double*)mxGetDoubles(ma));
        for (size_t i = 0; i < ellement_num; i++)
        {
            array[i] = ptr[i];
        }
        break;
    }
    case mxSINGLE_CLASS:
    {
        auto ptr = ((float*)mxGetSingles(ma));
        for (size_t i = 0; i < ellement_num; i++)
        {
            array[i] = ptr[i];
        }
        break;
    }
    case mxINT32_CLASS:
    {
        auto ptr = ((int32_t*)mxGetInt32s(ma));
        for (size_t i = 0; i < ellement_num; i++)
        {
            array[i] = ptr[i];
        }
        break;
    }
    case mxINT16_CLASS:
    {
        auto ptr = ((int16_t*)mxGetData(ma));
        for (size_t i = 0; i < ellement_num; i++)
        {
            array[i] = ptr[i];
        }
        break;
    }
    case mxUINT32_CLASS:
    {
        auto ptr = ((uint32_t*)mxGetData(ma));
        for (size_t i = 0; i < ellement_num; i++)
        {
            array[i] = ptr[i];
        }
        break;
    }
    case mxUINT16_CLASS:
    {
        auto ptr = ((uint16_t*)mxGetData(ma));
        for (size_t i = 0; i < ellement_num; i++)
        {
            array[i] = ptr[i];
        }
        break;
    }
    case mxLOGICAL_CLASS:
    {
        auto ptr = (mxGetLogicals(ma));
        for (size_t i = 0; i < ellement_num; i++)
        {
            array[i] = ptr[i];
        }
        break;
    }
    default:
        mexErrMsgTxt("Not a supported mxClassID");
    }
}
template <typename T2, std::size_t N> void convert(const mxArray* ma, std::array<T2,N>& array)
{
    size_t ellement_num = mxGetNumberOfElements(ma);

    if (ellement_num != N) {
        mexErrMsgTxt("The calss has not the exact number of element predicted use vector istead of array");
    }

    mxClassID classid = mxGetClassID(ma);
    switch (classid)
    {
    case mxUINT8_CLASS:
    {
        auto ptr = ((unsigned char*)mxGetUint8s(ma));
        for (size_t i = 0; i < ellement_num; i++)
        {
            array[i]=ptr[i];
        }
        break;
    }
    case mxDOUBLE_CLASS:
    {
        auto ptr = ((double*)mxGetDoubles(ma));
        for (size_t i = 0; i < ellement_num; i++)
        {
            array[i] = ptr[i];
        }
        break;
    }
    case mxSINGLE_CLASS:
    {
        auto ptr = ((float*)mxGetSingles(ma));
        for (size_t i = 0; i < ellement_num; i++)
        {
            array[i] = ptr[i];
        }
        break;
    }
    case mxINT32_CLASS:
    {
        auto ptr = ((int32_t*)mxGetInt32s(ma));
        for (size_t i = 0; i < ellement_num; i++)
        {
            array[i] = ptr[i];
        }
        break;
    }
    case mxINT16_CLASS:
    {
        auto ptr = ((int16_t*)mxGetData(ma));
        for (size_t i = 0; i < ellement_num; i++)
        {
            array[i] = ptr[i];
        }
        break;
    }
    case mxUINT32_CLASS:
    {
        auto ptr = ((uint32_t*)mxGetData(ma));
        for (size_t i = 0; i < ellement_num; i++)
        {
            array[i] = ptr[i];
        }
        break;
    }
    case mxUINT16_CLASS:
    {
        auto ptr = ((uint16_t*)mxGetData(ma));
        for (size_t i = 0; i < ellement_num; i++)
        {
            array[i] = ptr[i];
        }
        break;
    }
    case mxLOGICAL_CLASS:
    {
        auto ptr = (mxGetLogicals(ma));
        for (size_t i = 0; i < ellement_num; i++)
        {
            array[i] = ptr[i];
        }
        break;
    }
    default:
        mexErrMsgTxt("Not a supported mxClassID");
    }
}
template <> void convert(const mxArray* ma, std::string& sString)
{
    if (!mxIsChar(ma))
        mexErrMsgTxt("convert<const mxArray*, std::string&>: !mxIsChar(ma)");
    char* str = mxArrayToString(ma);
    sString = std::string(str);
}
template <> void convert(const mxArray* ma, std::map<std::string, const mxArray*>& map_out)
{
    map_out.clear();
    if (!mxIsStruct(ma)) {
        mexErrMsgTxt("The input is not a structure --> failed to convert");
    }
    int num_field = mxGetNumberOfFields(ma);
    for (size_t i = 0; i < num_field; i++)
    {
        const char* str = mxGetFieldNameByNumber(ma, i);
        std::string name = std::string(str);
        const mxArray* field = mxGetFieldByNumber(ma, 0, i);
        if (field == NULL) {
            mexErrMsgTxt("The input is not a structure or is empty");
        }
        map_out[name] = field;
    }
}


#endif