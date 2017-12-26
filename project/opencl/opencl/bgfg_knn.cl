#if CN==1

#define T_MEAN float
#define F_ZERO (0.0f)
#define cnMode 1

#define frameToMean(a, b) (b) = *(a);
#if FL==0
#define meanToFrame(a, b) *b = convert_uchar_sat(a);
#else
#define meanToFrame(a, b) *b = (float)a;
#endif

#else

#define T_MEAN float4
#define F_ZERO (0.0f, 0.0f, 0.0f, 0.0f)
#define cnMode 4

#if FL == 0
#define meanToFrame(a, b)\
    b[0] = convert_uchar_sat(a.x); \
    b[1] = convert_uchar_sat(a.y); \
    b[2] = convert_uchar_sat(a.z);
#else
#define meanToFrame(a, b)\
    b[0] = a.x; \
    b[1] = a.y; \
    b[2] = a.z;
#endif

#define frameToMean(a, b)\
    b.x = a[0]; \
    b.y = a[1]; \
    b.z = a[2]; \
    b.w = 0.0f;

#endif

__kernel void knn_kernel(__global const uchar* frame, int frame_step, int frame_offset, int frame_row, int frame_col,
                          __global uchar* nNextLongUpdate,
                          __global uchar* nNextMidUpdate,
                          __global uchar* nNextShortUpdate,
                          __global uchar* aModelIndexLong,
                          __global uchar* aModelIndexMid,
                          __global uchar* aModelIndexShort,
                          __global uchar* flag,
                          __global uchar* sample,
                          __global uchar* fgmask, int fgmask_step, int fgmask_offset,
                          int nLongCounter, int nMidCounter, int nShortCounter,
                          int nLongUpdate, int nMidUpdate, int nShortUpdate,
                          float c_Tb, int c_nkNN, float c_tau,
                          int randLong, int randMid, int randShort
#ifdef SHADOW_DETECT
                          , uchar c_shadowVal
#endif
                          )
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if( x < frame_col && y < frame_row)
    {
        #if FL==0
        __global const uchar* _frame = (frame + mad24(y, frame_step, mad24(x, CN, frame_offset)));
        #else
        __global const float* _frame = ((__global const float*)( frame + mad24(y, frame_step, frame_offset)) + mad24(x, CN, 0));
        #endif
        T_MEAN pix;
        frameToMean(_frame, pix);

        uchar foreground = 255; // 0 - the pixel classified as background

        int Pbf = 0;
        int Pb = 0;
        uchar include = 0;

        int pt_idx =  mad24(y, frame_col, x);
        int idx_step = frame_row * frame_col;

        __global T_MEAN* _sample = (__global T_MEAN*)(sample);

        uchar n = 0;
        for (; n < (NSAMPLES) * 3 ; ++n)
        {
            int n_idx = mad24(n, idx_step, pt_idx);

            T_MEAN c_mean = _sample[n_idx];
            uchar c_flag = flag[n_idx];

            T_MEAN diff = c_mean - pix;
            float dist2 = dot(diff, diff);

            if (dist2 < c_Tb)
            {
                Pbf++;
                if (c_flag)
                {
                    Pb++;
                    if (Pb >= c_nkNN)
                    {
                        include = 1;
                        foreground = 0;
                        break;
                    }
                }
            }
        }
        if (Pbf >= c_nkNN)
        {
            include = 1;
        }

#ifdef SHADOW_DETECT
        if (foreground)
        {
            int Ps = 0;
            for (uchar n = 0; n < (NSAMPLES) * 3 ; ++n)
            {
                int n_idx = mad24(n, idx_step, pt_idx);
                uchar c_flag = flag[n_idx];

                if (c_flag)
                {
                    T_MEAN c_mean = _sample[n_idx];
                    float numerator = dot(pix, c_mean);
                    float denominator = dot(c_mean, c_mean);

                    if (denominator == 0)
                        break;

                    if (numerator <= denominator && numerator >= c_tau * denominator)
                    {
                        float a = numerator / denominator;

                        T_MEAN dD = mad(a, c_mean, -pix);

                        if (dot(dD, dD) < c_Tb * a * a)
                        {
                            Ps++;
                            if (Ps >= c_nkNN)
                            {
                                foreground = c_shadowVal;
                                break;
                            }
                        }
                    }
                }
            }
        }
#endif
        __global uchar* _fgmask = fgmask + mad24(y, fgmask_step, x + fgmask_offset);
        *_fgmask = (uchar)foreground;

        __global uchar* _nNextLongUpdate = nNextLongUpdate + pt_idx;
        __global uchar* _nNextMidUpdate = nNextMidUpdate + pt_idx;
        __global uchar* _nNextShortUpdate = nNextShortUpdate + pt_idx;
        __global uchar* _aModelIndexLong = aModelIndexLong + pt_idx;
        __global uchar* _aModelIndexMid = aModelIndexMid + pt_idx;
        __global uchar* _aModelIndexShort = aModelIndexShort + pt_idx;

        uchar nextLongUpdate = _nNextLongUpdate[0];
        uchar nextMidUpdate = _nNextMidUpdate[0];
        uchar nextShortUpdate = _nNextShortUpdate[0];
        uchar modelIndexLong = _aModelIndexLong[0];
        uchar modelIndexMid = _aModelIndexMid[0];
        uchar modelIndexShort = _aModelIndexShort[0];
        int offsetLong = mad24(mad24(2, (NSAMPLES), modelIndexLong), idx_step, pt_idx);
        int offsetMid = mad24((NSAMPLES)+modelIndexMid, idx_step, pt_idx);
        int offsetShort = mad24(modelIndexShort, idx_step, pt_idx);
        if (nextLongUpdate == nLongCounter)
        {
            _sample[offsetLong] = _sample[offsetMid];
            flag[offsetLong] = flag[offsetMid];
            _aModelIndexLong[0] = (modelIndexLong >= ((NSAMPLES)-1)) ? 0 : (modelIndexLong + 1);
        }

        if (nLongCounter == (nLongUpdate-1))
            _nNextLongUpdate[0] = (uchar)( randLong % nLongUpdate );

        if (nextMidUpdate == nMidCounter)
        {
            _sample[offsetMid] = _sample[offsetShort];
            flag[offsetMid] = flag[offsetShort];
            _aModelIndexMid[0] = (modelIndexMid >= ((NSAMPLES)-1)) ? 0 : (modelIndexMid + 1);
        }

        if (nMidCounter == (nMidUpdate-1))
            _nNextMidUpdate[0] = (uchar)( randMid % nMidUpdate );

        if (nextShortUpdate == nShortCounter)
        {
            _sample[offsetShort] = pix;
            flag[offsetShort] = include;
            _aModelIndexShort[0] = (modelIndexShort >= ((NSAMPLES)-1)) ? 0 : (modelIndexShort + 1);
        }

        if (nShortCounter == (nShortUpdate-1))
            _nNextShortUpdate[0] = (uchar)( randShort % nShortUpdate );
    }
}
