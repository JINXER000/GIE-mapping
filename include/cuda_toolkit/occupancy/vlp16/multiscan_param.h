

#ifndef SRC_MULTISCAN_PARAM_H
#define SRC_MULTISCAN_PARAM_H

struct MulScanParam
{
    float max_r, theta_inc, theta_min, phi_inc, phi_min;
    int scan_num;
    int ring_num;

    MulScanParam(int scan_num_, int ring_num_,
                  float max_r_, float theta_inc_,
                  float theta_min_, float phi_inc_,
                  float phi_min_):
            scan_num(scan_num_),
            ring_num(ring_num_),
            max_r(max_r_),
            theta_inc(theta_inc_),
            theta_min(theta_min_),
            phi_inc(phi_inc_),
            phi_min(phi_min_)
    {}
    MulScanParam()
    {}
};

#endif //SRC_MULTISCAN_PARAM_H
