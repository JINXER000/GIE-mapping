#ifndef PNTCLD_PARAM_H
#define PNTCLD_PARAM_H

struct PntcldParam
{
    int cld_sz;
    int valid_pnt_count;
    PntcldParam():
        cld_sz(0),
        valid_pnt_count(0)
    {}

    PntcldParam(int cld_sz_):
        cld_sz(cld_sz_),
        valid_pnt_count(0)
    {}
};

#endif // PNTCLD_PARAM_H
