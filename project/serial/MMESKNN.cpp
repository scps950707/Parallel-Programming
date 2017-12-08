#include <opencv2/opencv.hpp>
#include <stdlib.h>
#include <iostream>
#include "MMESKNN.hpp"
using namespace std;


//{ to do - paralelization ...
//struct KNNInvoker....
static inline void
_cvUpdatePixelBackgroundNP(
    const uchar *currPixel,
    int channels,
    int nSample,
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
    int ndata        = 1 + channels;
    long offsetLong  = ndata * ( *ModelIndexLong + nSample * 2 );
    long offsetMid   = ndata * ( *ModelIndexMid  + nSample * 1 );
    long offsetShort = ndata * ( *ModelIndexShort );
    uint seed = time( NULL );

    // Long update?
    if ( *NextLongUpdate == LongCounter )
    {
        // add the oldest pixel from Mid to the list of values (for each color)
        Model[offsetLong]     = Model[offsetMid];
        Model[offsetLong + 1] = Model[offsetMid + 1];
        Model[offsetLong + 2] = Model[offsetMid + 2];
        Model[offsetLong + 3] = Model[offsetMid + 3];
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
        Model[offsetMid + 3] = Model[offsetShort + 3];
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
        //set the include flag
        Model[offsetShort + channels] = ( uchar )include;
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
    uchar *Model,
    float Tb,
    int kNN,
    float tau,
    bool ShadowDetection,
    bool &include
)
{
    int Pbf = 0; // the total probability that this pixel is background
    int Pb = 0; //background model probability

    include = false; //do we include this pixel into background model?

    int ndata = channels + 1;
    /* long posPixel = pixel * ndata * nSample * 3; */
    // now increase the probability for each pixel
    for ( int n = 0; n < nSample * 3; n++ )
    {
        //calculate difference and distance
        int d0 = Model[n * ndata] - currPixel[0];
        int d1 = Model[n * ndata + 1] - currPixel[1];
        int d2 = Model[n * ndata + 2] - currPixel[2];
        int dist2 = d0 * d0 + d1 * d1 + d2 * d2;

        if ( dist2 < Tb )
        {
            Pbf++;//all
            //background only
            if ( Model[n * ndata + 3] ) //indicator
            {
                Pb++;
                if ( Pb >= kNN ) //Tb
                {
                    include = true; //include
                    return 1;//background ->exit
                }
            }
        }
    }

    //include?
    if ( Pbf >= kNN ) //Tbf)
    {
        include = true;
    }

    // Detected as moving object, perform shadow detection
    if ( ShadowDetection )
    {
        int Ps = 0; // the total probability that this pixel is background shadow
        for ( int n = 0; n < nSample * 3; n++ )
        {
            uchar *mean_m = &Model[n * ndata];

            if ( mean_m[channels] ) //check only background
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

static inline void
icvUpdatePixelBackgroundNP(
    int channels,
    int nrows,
    int ncols,
    uchar *srcData,
    uchar *dst,
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
                             Model + posPixel * ( channels + 1 ) * nSample * 3,
                             // pass Model's start address of pixel
                             Tb,
                             kNN,
                             Tau,
                             ShadowDetection,
                             include
                         );

            _cvUpdatePixelBackgroundNP(
                currPixel,
                channels,
                nSample,
                Model + posPixel * ( channels + 1 ) * nSample * 3,
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
