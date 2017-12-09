#include <opencv2/opencv.hpp>
#include <stdlib.h>
#include <iostream>
#include <thread>
#include "MMESKNN.hpp"
using namespace std;


//{ to do - paralelization ...
//struct KNNInvoker....
static inline void
_cvUpdatePixelBackgroundNP(
    const uchar *currPixel,
    int channels,
    int nSample,
    bool *flag,
    uchar *Model,
    uchar *NextLongUpdate,
    uchar *NextMidUpdate,
    uchar *NextShortUpdate,
    uchar *ModelIndexLong,
    uchar *ModelIndexMid,
    uchar *ModelIndexShort,
    int LongCounter,
    int MidCounter,
    int ShortCounter,
    int LongUpdate,
    int MidUpdate,
    int ShortUpdate,
    bool include
)
{
    // hold the offset
    long flagoffsetShort = *ModelIndexShort;
    long flagoffsetMid   = *ModelIndexMid  + nSample * 1;
    long flagoffsetLong  = *ModelIndexLong + nSample * 2;
    long offsetShort = channels * ( *ModelIndexShort );
    long offsetMid   = channels * ( *ModelIndexMid  + nSample * 1 );
    long offsetLong  = channels * ( *ModelIndexLong + nSample * 2 );
    uint seed = time( NULL );

    // Long update?
    if ( *NextLongUpdate == LongCounter )
    {
        // add the oldest pixel from Mid to the list of values (for each color)
        Model[offsetLong]     = Model[offsetMid];
        Model[offsetLong + 1] = Model[offsetMid + 1];
        Model[offsetLong + 2] = Model[offsetMid + 2];
        flag[flagoffsetLong]  = flag[flagoffsetMid];
        // increase the index
        *ModelIndexLong = ( *ModelIndexLong >= ( nSample - 1 ) ) ? 0 : ( *ModelIndexLong + 1 );
    }
    if ( LongCounter == ( LongUpdate - 1 ) )
    {
        *NextLongUpdate = ( uchar )( rand_r( &seed ) % LongUpdate ); //0,...LongUpdate-1;
    }

    // Mid update?
    if ( *NextMidUpdate == MidCounter )
    {
        // add this pixel to the list of values (for each color)
        Model[offsetMid]     = Model[offsetShort];
        Model[offsetMid + 1] = Model[offsetShort + 1];
        Model[offsetMid + 2] = Model[offsetShort + 2];
        flag[flagoffsetMid]  = flag[flagoffsetShort];
        // increase the index
        *ModelIndexMid = ( *ModelIndexMid >= ( nSample - 1 ) ) ? 0 : ( *ModelIndexMid + 1 );
    }
    if ( MidCounter == ( MidUpdate - 1 ) )
    {
        *NextMidUpdate = ( uchar )( rand_r( &seed ) % MidUpdate );
    }

    // Short update?
    if ( *NextShortUpdate == ShortCounter )
    {
        // add this pixel to the list of values (for each color)
        Model[offsetShort]     = currPixel[0];
        Model[offsetShort + 1] = currPixel[1];
        Model[offsetShort + 2] = currPixel[2];
        flag[flagoffsetShort]  = include;
        // increase the index
        *ModelIndexShort = ( *ModelIndexShort >= ( nSample - 1 ) ) ? 0 : ( *ModelIndexShort + 1 );
    }
    if ( ShortCounter == ( ShortUpdate - 1 ) )
    {
        *NextShortUpdate = ( uchar )( rand_r( &seed ) % ShortUpdate );
    }
}

static inline int
_cvCheckPixelBackgroundNP(
    const uchar *currPixel,
    int channels,
    int nSample,
    bool *flag,
    uchar *Model,
    float Tb,
    int kNN,
    float tau,
    bool ShadowDetection,
    bool *include
)
{
    int Pbf = 0; // the total probability that this pixel is background
    int Pb = 0; //background model probability

    *include = false; //do we include this pixel into background model?

    /* long posPixel = pixel * ndata * nSample * 3; */
    // now increase the probability for each pixel
    for ( int n = 0; n < nSample * 3; n++ )
    {
        //calculate difference and distance
        int d0 = Model[n * channels] - currPixel[0];
        int d1 = Model[n * channels + 1] - currPixel[1];
        int d2 = Model[n * channels + 2] - currPixel[2];
        int dist2 = d0 * d0 + d1 * d1 + d2 * d2;

        if ( dist2 < Tb )
        {
            Pbf++;//all
            //background only
            if ( flag[n] ) //indicator
            {
                Pb++;
                if ( Pb >= kNN ) //Tb
                {
                    *include = true; //include
                    return 1;//background ->exit
                }
            }
        }
    }

    //include?
    if ( Pbf >= kNN ) //Tbf)
    {
        *include = true;
    }

    // Detected as moving object, perform shadow detection
    if ( ShadowDetection )
    {
        int Ps = 0; // the total probability that this pixel is background shadow
        for ( int n = 0; n < nSample * 3; n++ )
        {
            uchar *mean_m = Model + n * channels;

            if ( flag[n] ) //check only background
            {
                float numerator = 0.0f;
                float denominator = 0.0f;
                for ( int c = 0; c < channels; c++ )
                {
                    numerator   += currPixel[c] * mean_m[c];
                    denominator += mean_m[c] * mean_m[c];
                }

                // no division by zero allowed
                if ( denominator == 0 )
                {
                    return 0;
                }

                // if tau < a < 1 then also check the color distortion
                if ( numerator <= denominator && numerator >= tau * denominator )
                {
                    float a = numerator / denominator;
                    float dist2a = 0.0f;

                    for ( int c = 0; c < channels; c++ )
                    {
                        float dD = a * mean_m[c] - currPixel[c];
                        dist2a += dD * dD;
                    }

                    if ( dist2a < Tb * a * a )
                    {
                        Ps++;
                        if ( Ps >= kNN ) //shadow
                        {
                            return 2;
                        }
                    }
                }
            }
        }
    }
    return 0;
}

void runner
(
    int nrows,
    int ncols,
    const uchar *srcData,
    uchar *dst,
    int channels,
    int nSample,
    bool *flag,
    uchar *Model,
    float Tb,
    int kNN,
    bool ShadowDetection,
    uchar *NextLongUpdate,
    uchar *NextMidUpdate,
    uchar *NextShortUpdate,
    uchar *ModelIndexLong,
    uchar *ModelIndexMid,
    uchar *ModelIndexShort,
    int LongCounter,
    int MidCounter,
    int ShortCounter,
    int	ShortUpdate,
    int MidUpdate,
    int LongUpdate,
    float AlphaT,
    float Tau,
    uchar ShadowValue
)
{
    //go through the image
    for ( long y = 0; y < nrows; y++ )
    {
        for ( long x = 0; x < ncols; x++ )
        {
            int posPixel = ncols * y + x;
            /* start addr of current pixel */
            const uchar *currPixel = srcData + posPixel * channels;

            //update model+ background subtract
            bool include = 0;
            int result = _cvCheckPixelBackgroundNP(
                             currPixel,
                             channels,
                             nSample,
                             flag + posPixel * nSample * 3,
                             Model + posPixel * channels * nSample * 3,
                             // pass Model's start address of pixel
                             Tb,
                             kNN,
                             Tau,
                             ShadowDetection,
                             &include
                         );

            _cvUpdatePixelBackgroundNP(
                currPixel,
                channels,
                nSample,
                flag + posPixel * nSample * 3,
                Model + posPixel * channels * nSample * 3,
                NextLongUpdate + posPixel,
                NextMidUpdate + posPixel,
                NextShortUpdate + posPixel,
                ModelIndexLong + posPixel,
                ModelIndexMid + posPixel,
                ModelIndexShort + posPixel,
                LongCounter,
                MidCounter,
                ShortCounter,
                LongUpdate,
                MidUpdate,
                ShortUpdate,
                include
            );
            switch ( result )
            {
            case 0:
                //foreground
                dst[posPixel] = 255;
                break;
            case 1:
                //background
                dst[posPixel] = 0;
                break;
            case 2:
                //shadow
                dst[posPixel] = ShadowValue;
                break;
            }
        }
    }
}

static inline void
icvUpdatePixelBackgroundNP(
    int channels,
    int nrows,
    int ncols,
    uchar *srcData,
    uchar *dst,
    bool *flag,
    uchar *Model,
    uchar *NextLongUpdate,
    uchar *NextMidUpdate,
    uchar *NextShortUpdate,
    uchar *ModelIndexLong,
    uchar *ModelIndexMid,
    uchar *ModelIndexShort,
    int &LongCounter,
    int &MidCounter,
    int &ShortCounter,
    int nSample,
    float AlphaT,
    float Tb,
    int kNN,
    float Tau,
    bool ShadowDetection,
    uchar ShadowValue
)
{
    //recalculate update rates - in case alpha is changed
    // calculate update parameters (using alpha)
    int Kshort, Kmid, Klong;
    //approximate exponential learning curve
    Kshort = ( int )( log( 0.7 ) / log( 1 - AlphaT ) ) + 1; //Kshort
    Kmid   = ( int )( log( 0.4 ) / log( 1 - AlphaT ) ) - Kshort + 1; //Kmid
    Klong  = ( int )( log( 0.1 ) / log( 1 - AlphaT ) ) - Kshort - Kmid + 1; //Klong

    //refresh rates
    int	ShortUpdate = ( Kshort / nSample ) + 1;
    int MidUpdate   = ( Kmid   / nSample ) + 1;
    int LongUpdate  = ( Klong  / nSample ) + 1;
    int cpus = 4;
    thread *ts = new thread[cpus];
    int chunks = ( cpus - 1 + nrows ) / cpus;
    int offset = 0;
    int remainRows = nrows;
    for ( int i = 0; i < cpus; i++ )
    {
        chunks = i == cpus - 1 ? remainRows : chunks;
        ts[i] = thread(
                    runner,
                    /* parameters */
                    chunks,
                    ncols,
                    srcData + offset * ncols * channels,
                    dst + offset * ncols,
                    channels,
                    nSample,
                    flag + offset * ncols * nSample * 3,
                    Model + offset * ncols * channels * nSample * 3,
                    Tb,
                    kNN,
                    ShadowDetection,
                    NextLongUpdate + offset * ncols,
                    NextMidUpdate + offset * ncols,
                    NextShortUpdate + offset * ncols,
                    ModelIndexLong + offset * ncols,
                    ModelIndexMid + offset * ncols,
                    ModelIndexShort + offset * ncols,
                    LongCounter,
                    MidCounter,
                    ShortCounter,
                    ShortUpdate,
                    MidUpdate,
                    LongUpdate,
                    AlphaT,
                    Tau,
                    ShadowValue
                );
        remainRows -= chunks;
        offset += chunks;
    }
    for ( int i = 0; i < cpus; i++ )
    {
        ts[i].join();
    }
    delete[] ts;

    //update counters for the refresh rate

    //0,1,...,ShortUpdate-1
    if ( ++ShortCounter >= ShortUpdate )
    {
        ShortCounter = 0;
    }
    if ( ++MidCounter >= MidUpdate )
    {
        MidCounter = 0;
    }
    if ( ++LongCounter >= LongUpdate )
    {
        LongCounter = 0;
    }
}



void MMESKNN::apply( cv::Mat &image, cv::Mat &fgmask, double learningRate )
{
    bool needToInitialize = nframes == 0 || learningRate >= 1 || image.size() != frameSize || image.type() != frameType;

    if ( needToInitialize )
    {
        initialize( image.size(), image.type() );
    }

    fgmask.create( image.size(), CV_8U );

    ++nframes;
    learningRate = learningRate >= 0 && nframes > 1 ? learningRate : 1. / std::min( 2 * nframes, history );
    CV_Assert( learningRate >= 0 );

    icvUpdatePixelBackgroundNP(
        image.channels(),
        image.rows,
        image.cols,
        image.ptr(),
        fgmask.ptr(),
        flag,
        bgmodel,
        nNextLongUpdate,
        nNextMidUpdate,
        nNextShortUpdate,
        aModelIndexLong,
        aModelIndexMid,
        aModelIndexShort,
        nLongCounter,
        nMidCounter,
        nShortCounter,
        nSample,
        ( float )learningRate,
        fTb,
        nkNN,
        fTau,
        ShadowDetection,
        ShadowValue
    );
}
