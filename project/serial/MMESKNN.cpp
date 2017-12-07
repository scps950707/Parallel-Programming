#include <opencv2/opencv.hpp>
#include <stdlib.h>
#include <iostream>
#include "MMESKNN.hpp"
using namespace std;


//{ to do - paralelization ...
//struct KNNInvoker....
static inline void
_cvUpdatePixelBackgroundNP(
    long pixel,
    const uchar *data,
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
    int ndata = 1 + channels;
    long offsetLong =  ndata * ( pixel * nSample * 3 + ModelIndexLong[pixel] + nSample * 2 );
    long offsetMid =   ndata * ( pixel * nSample * 3 + ModelIndexMid[pixel]  + nSample * 1 );
    long offsetShort = ndata * ( pixel * nSample * 3 + ModelIndexShort[pixel] );

    // Long update?
    if ( NextLongUpdate[pixel] == LongCounter )
    {
        // add the oldest pixel from Mid to the list of values (for each color)
        memcpy( &Model[offsetLong], &Model[offsetMid], ndata * sizeof( uchar ) );
        // increase the index
        ModelIndexLong[pixel] = ( ModelIndexLong[pixel] >= ( nSample - 1 ) ) ? 0 : ( ModelIndexLong[pixel] + 1 );
    };
    if ( LongCounter == ( LongUpdate - 1 ) )
    {
        //NextLongUpdate[pixel] = (uchar)(((LongUpdate)*(rand()-1))/RAND_MAX);//0,...LongUpdate-1;
        NextLongUpdate[pixel] = ( uchar )( rand() % LongUpdate ); //0,...LongUpdate-1;
    };

    // Mid update?
    if ( NextMidUpdate[pixel] == MidCounter )
    {
        // add this pixel to the list of values (for each color)
        memcpy( &Model[offsetMid], &Model[offsetShort], ndata * sizeof( uchar ) );
        // increase the index
        ModelIndexMid[pixel] = ( ModelIndexMid[pixel] >= ( nSample - 1 ) ) ? 0 : ( ModelIndexMid[pixel] + 1 );
    };
    if ( MidCounter == ( MidUpdate - 1 ) )
    {
        NextMidUpdate[pixel] = ( uchar )( rand() % MidUpdate );
    };

    // Short update?
    if ( NextShortUpdate[pixel] == ShortCounter )
    {
        // add this pixel to the list of values (for each color)
        memcpy( &Model[offsetShort], data, ndata * sizeof( uchar ) );
        //set the include flag
        Model[offsetShort + channels] = ( uchar )include;
        // increase the index
        ModelIndexShort[pixel] = ( ModelIndexShort[pixel] >= ( nSample - 1 ) ) ? 0 : ( ModelIndexShort[pixel] + 1 );
    };
    if ( ShortCounter == ( ShortUpdate - 1 ) )
    {
        NextShortUpdate[pixel] = ( uchar )( rand() % ShortUpdate );
    };
}

static inline int
_cvCheckPixelBackgroundNP(
    long pixel,
    const uchar *data,
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

    //uchar& include=data[channels];
    include = false; //do we include this pixel into background model?

    int ndata = channels + 1;
    long posPixel = pixel * ndata * nSample * 3;
    // now increase the probability for each pixel
    for ( int n = 0; n < nSample * 3; n++ )
    {
        uchar *mean_m = &Model[posPixel + n * ndata];

        //calculate difference and distance
        float d0 = ( float )mean_m[0] - data[0];
        float d1 = ( float )mean_m[1] - data[1];
        float d2 = ( float )mean_m[2] - data[2];
        float dist2 = d0 * d0 + d1 * d1 + d2 * d2;

        if ( dist2 < Tb )
        {
            Pbf++;//all
            //background only
            //if(Model[subPosPixel + channels])//indicator
            if ( mean_m[channels] ) //indicator
            {
                Pb++;
                if ( Pb >= kNN ) //Tb
                {
                    include = true; //include
                    return 1;//background ->exit
                };
            }
        };
    };

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
            //long subPosPixel = posPixel + n*ndata;
            uchar *mean_m = &Model[posPixel + n * ndata];

            if ( mean_m[channels] ) //check only background
            {
                float numerator = 0.0f;
                float denominator = 0.0f;
                for ( int c = 0; c < channels; c++ )
                {
                    numerator   += ( float )data[c] * mean_m[c];
                    denominator += ( float )mean_m[c] * mean_m[c];
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
                        float dD = a * mean_m[c] - data[c];
                        dist2a += dD * dD;
                    }

                    if ( dist2a < Tb * a * a )
                    {
                        Ps++;
                        if ( Ps >= kNN ) //shadow
                        {
                            return 2;
                        }
                    };
                };
            };
        };
    }
    return 0;
}

static inline void
icvUpdatePixelBackgroundNP(
    const cv::Mat &src,
    cv::Mat &dst,
    uchar *Model,
    uchar *NextLongUpdate,
    uchar *NextMidUpdate,
    uchar *NextShortUpdate,
    uchar *ModelIndexLong,
    uchar *ModelIndexMid,
    uchar *ModelIndexShort,
    int &_nLongCounter,
    int &_nMidCounter,
    int &_nShortCounter,
    int nSample,
    float AlphaT,
    float Tb,
    int kNN,
    float Tau,
    bool ShadowDetection,
    uchar ShadowValue
)
{
    int channels = CV_MAT_CN( src.type() );

    //recalculate update rates - in case alpha is changed
    // calculate update parameters (using alpha)
    int Kshort, Kmid, Klong;
    //approximate exponential learning curve
    Kshort = ( int )( log( 0.7 ) / log( 1 - AlphaT ) ) + 1; //Kshort
    Kmid = ( int )( log( 0.4 ) / log( 1 - AlphaT ) ) - Kshort + 1; //Kmid
    Klong = ( int )( log( 0.1 ) / log( 1 - AlphaT ) ) - Kshort - Kmid + 1; //Klong

    //refresh rates
    int	ShortUpdate = ( Kshort / nSample ) + 1;
    int MidUpdate = ( Kmid / nSample ) + 1;
    int LongUpdate = ( Klong / nSample ) + 1;

    //update counters for the refresh rate
    int LongCounter = _nLongCounter;
    int MidCounter = _nMidCounter;
    int ShortCounter = _nShortCounter;

    _nShortCounter++;//0,1,...,ShortUpdate-1
    _nMidCounter++;
    _nLongCounter++;
    if ( _nShortCounter >= ShortUpdate )
    {
        _nShortCounter = 0;
    }
    if ( _nMidCounter >= MidUpdate )
    {
        _nMidCounter = 0;
    }
    if ( _nLongCounter >= LongUpdate )
    {
        _nLongCounter = 0;
    }

    //go through the image
    long i = 0;
    for ( long y = 0; y < src.rows; y++ )
    {
        for ( long x = 0; x < src.cols; x++ )
        {
            const uchar *data = src.ptr( ( int )y, ( int )x );

            //update model+ background subtract
            bool include = 0;
            int result = _cvCheckPixelBackgroundNP(
                             i,
                             data,
                             channels,
                             nSample,
                             Model,
                             Tb,
                             kNN,
                             Tau,
                             ShadowDetection,
                             include
                         );

            _cvUpdatePixelBackgroundNP(
                i,
                data,
                channels,
                nSample,
                Model,
                NextLongUpdate,
                NextMidUpdate,
                NextShortUpdate,
                ModelIndexLong,
                ModelIndexMid,
                ModelIndexShort,
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
                *dst.ptr( ( int )y, ( int )x ) = 255;
                break;
            case 1:
                //background
                *dst.ptr( ( int )y, ( int )x ) = 0;
                break;
            case 2:
                //shadow
                *dst.ptr( ( int )y, ( int )x ) = ShadowValue;
                break;
            }
            i++;
        }
    }
}



void MMESKNN::apply( cv::Mat &image, cv::Mat &fgmask, double learningRate )
{
    /* CV_INSTRUMENT_REGION() */

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
        image,
        fgmask,
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
