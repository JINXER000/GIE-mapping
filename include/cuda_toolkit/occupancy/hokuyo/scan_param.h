#ifndef SCAN_PARAM_H
#define SCAN_PARAM_H

struct ScanParam
{
    float max_r, theta_inc, theta_min;
    int scan_num;
    ScanParam(int scan_num_,
                float max_r_, float theta_inc_,
                float theta_min_):
        scan_num(scan_num_),
        max_r(max_r_),
        theta_inc(theta_inc_),
        theta_min(theta_min_)
    {}
    ScanParam()
    {}
};

#endif // SCAN_PARAM_H
