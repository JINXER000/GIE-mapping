#ifndef CAMERA_PARAM_H
#define CAMERA_PARAM_H

struct CamParam
{
    int rows, cols;
    float cx, cy, fx, fy;
    bool valid_NaN;
    CamParam():
        rows(0),cols(0),
        cx(0),cy(0),
        fx(0),fy(0),
        valid_NaN(0)
    {}

    CamParam(int rows_, int cols_,
                  float cx_,float cy_,
                  float fx_, float fy_,
                  bool valid_NaN_):
        rows(rows_),cols(cols_),
        cx(cx_),cy(cy_),
        fx(fx_),fy(fy_),
        valid_NaN(valid_NaN_)
    {}
};

#endif // CAMERA_PARAM_H
