#include <opencv2/opencv.hpp>
#include <stdlib.h>
#include <iostream>
#include "MMESKNN.hpp"
using namespace std;


//{ to do - paralelization ...
//struct KNNInvoker....
static inline void
_cvUpdatePixelBackgroundNP( long pixel, const uchar *data, int nchannels, int m_nN,
                            uchar *m_aModel,
                            uchar *m_nNextLongUpdate,
                            uchar *m_nNextMidUpdate,
                            uchar *m_nNextShortUpdate,
                            uchar *m_aModelIndexLong,
                            uchar *m_aModelIndexMid,
                            uchar *m_aModelIndexShort,
                            int m_nLongCounter,
                            int m_nMidCounter,
                            int m_nShortCounter,
                            int m_nLongUpdate,
                            int m_nMidUpdate,
                            int m_nShortUpdate,
                            uchar include
                          )
{
    // hold the offset
    int ndata = 1 + nchannels;
    long offsetLong =  ndata * ( pixel * m_nN * 3 + m_aModelIndexLong[pixel] + m_nN * 2 );
    long offsetMid =   ndata * ( pixel * m_nN * 3 + m_aModelIndexMid[pixel]  + m_nN * 1 );
    long offsetShort = ndata * ( pixel * m_nN * 3 + m_aModelIndexShort[pixel] );

    // Long update?
    if ( m_nNextLongUpdate[pixel] == m_nLongCounter )
    {
        // add the oldest pixel from Mid to the list of values (for each color)
        memcpy( &m_aModel[offsetLong], &m_aModel[offsetMid], ndata * sizeof( unsigned char ) );
        // increase the index
        m_aModelIndexLong[pixel] = ( m_aModelIndexLong[pixel] >= ( m_nN - 1 ) ) ? 0 : ( m_aModelIndexLong[pixel] + 1 );
    };
    if ( m_nLongCounter == ( m_nLongUpdate - 1 ) )
    {
        //m_nNextLongUpdate[pixel] = (uchar)(((m_nLongUpdate)*(rand()-1))/RAND_MAX);//0,...m_nLongUpdate-1;
        m_nNextLongUpdate[pixel] = ( uchar )( rand() % m_nLongUpdate ); //0,...m_nLongUpdate-1;
    };

    // Mid update?
    if ( m_nNextMidUpdate[pixel] == m_nMidCounter )
    {
        // add this pixel to the list of values (for each color)
        memcpy( &m_aModel[offsetMid], &m_aModel[offsetShort], ndata * sizeof( unsigned char ) );
        // increase the index
        m_aModelIndexMid[pixel] = ( m_aModelIndexMid[pixel] >= ( m_nN - 1 ) ) ? 0 : ( m_aModelIndexMid[pixel] + 1 );
    };
    if ( m_nMidCounter == ( m_nMidUpdate - 1 ) )
    {
        m_nNextMidUpdate[pixel] = ( uchar )( rand() % m_nMidUpdate );
    };

    // Short update?
    if ( m_nNextShortUpdate[pixel] == m_nShortCounter )
    {
        // add this pixel to the list of values (for each color)
        memcpy( &m_aModel[offsetShort], data, ndata * sizeof( unsigned char ) );
        //set the include flag
        m_aModel[offsetShort + nchannels] = include;
        // increase the index
        m_aModelIndexShort[pixel] = ( m_aModelIndexShort[pixel] >= ( m_nN - 1 ) ) ? 0 : ( m_aModelIndexShort[pixel] + 1 );
    };
    if ( m_nShortCounter == ( m_nShortUpdate - 1 ) )
    {
        m_nNextShortUpdate[pixel] = ( uchar )( rand() % m_nShortUpdate );
    };
}

static inline int
_cvCheckPixelBackgroundNP( long pixel,
                           const uchar *data, int nchannels,
                           int m_nN,
                           uchar *m_aModel,
                           float m_fTb,
                           int m_nkNN,
                           float tau,
                           int m_nShadowDetection,
                           uchar &include )
{
    int Pbf = 0; // the total probability that this pixel is background
    int Pb = 0; //background model probability
    float dData[CV_CN_MAX];

    //uchar& include=data[nchannels];
    include = 0; //do we include this pixel into background model?

    int ndata = nchannels + 1;
    long posPixel = pixel * ndata * m_nN * 3;
//	float k;
    // now increase the probability for each pixel
    for ( int n = 0; n < m_nN * 3; n++ )
    {
        uchar *mean_m = &m_aModel[posPixel + n * ndata];

        //calculate difference and distance
        float dist2;

        if ( nchannels == 3 )
        {
            dData[0] = ( float )mean_m[0] - data[0];
            dData[1] = ( float )mean_m[1] - data[1];
            dData[2] = ( float )mean_m[2] - data[2];
            dist2 = dData[0] * dData[0] + dData[1] * dData[1] + dData[2] * dData[2];
        }
        else
        {
            dist2 = 0.f;
            for ( int c = 0; c < nchannels; c++ )
            {
                dData[c] = ( float )mean_m[c] - data[c];
                dist2 += dData[c] * dData[c];
            }
        }

        if ( dist2 < m_fTb )
        {
            Pbf++;//all
            //background only
            //if(m_aModel[subPosPixel + nchannels])//indicator
            if ( mean_m[nchannels] ) //indicator
            {
                Pb++;
                if ( Pb >= m_nkNN ) //Tb
                {
                    include = 1; //include
                    return 1;//background ->exit
                };
            }
        };
    };

    //include?
    if ( Pbf >= m_nkNN ) //m_nTbf)
    {
        include = 1;
    }

    // Detected as moving object, perform shadow detection
    if ( m_nShadowDetection )
    {
        int Ps = 0; // the total probability that this pixel is background shadow
        for ( int n = 0; n < m_nN * 3; n++ )
        {
            //long subPosPixel = posPixel + n*ndata;
            uchar *mean_m = &m_aModel[posPixel + n * ndata];

            if ( mean_m[nchannels] ) //check only background
            {
                float numerator = 0.0f;
                float denominator = 0.0f;
                for ( int c = 0; c < nchannels; c++ )
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

                    for ( int c = 0; c < nchannels; c++ )
                    {
                        float dD = a * mean_m[c] - data[c];
                        dist2a += dD * dD;
                    }

                    if ( dist2a < m_fTb * a * a )
                    {
                        Ps++;
                        if ( Ps >= m_nkNN ) //shadow
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
icvUpdatePixelBackgroundNP( const cv::Mat &_src, cv::Mat &_dst,
                            cv::Mat &_bgmodel,
                            cv::Mat &_nNextLongUpdate,
                            cv::Mat &_nNextMidUpdate,
                            cv::Mat &_nNextShortUpdate,
                            cv::Mat &_aModelIndexLong,
                            cv::Mat &_aModelIndexMid,
                            cv::Mat &_aModelIndexShort,
                            int &_nLongCounter,
                            int &_nMidCounter,
                            int &_nShortCounter,
                            int _nN,
                            float _fAlphaT,
                            float _fTb,
                            int _nkNN,
                            float _fTau,
                            int _bShadowDetection,
                            uchar nShadowDetection
                          )
{
    int nchannels = CV_MAT_CN( _src.type() );

    //model
    uchar *m_aModel = _bgmodel.ptr( 0 );
    uchar *m_nNextLongUpdate = _nNextLongUpdate.ptr( 0 );
    uchar *m_nNextMidUpdate = _nNextMidUpdate.ptr( 0 );
    uchar *m_nNextShortUpdate = _nNextShortUpdate.ptr( 0 );
    uchar *m_aModelIndexLong = _aModelIndexLong.ptr( 0 );
    uchar *m_aModelIndexMid = _aModelIndexMid.ptr( 0 );
    uchar *m_aModelIndexShort = _aModelIndexShort.ptr( 0 );

    //some constants
    int m_nN = _nN;
    float m_fAlphaT = _fAlphaT;
    float m_fTb = _fTb; //Tb - threshold on the distance
    float m_fTau = _fTau;
    int m_nkNN = _nkNN;
    int m_bShadowDetection = _bShadowDetection;

    //recalculate update rates - in case alpha is changed
    // calculate update parameters (using alpha)
    int Kshort, Kmid, Klong;
    //approximate exponential learning curve
    Kshort = ( int )( log( 0.7 ) / log( 1 - m_fAlphaT ) ) + 1; //Kshort
    Kmid = ( int )( log( 0.4 ) / log( 1 - m_fAlphaT ) ) - Kshort + 1; //Kmid
    Klong = ( int )( log( 0.1 ) / log( 1 - m_fAlphaT ) ) - Kshort - Kmid + 1; //Klong

    //refresh rates
    int	m_nShortUpdate = ( Kshort / m_nN ) + 1;
    int m_nMidUpdate = ( Kmid / m_nN ) + 1;
    int m_nLongUpdate = ( Klong / m_nN ) + 1;

    //int	m_nShortUpdate = MAX((Kshort/m_nN),m_nN);
    //int m_nMidUpdate = MAX((Kmid/m_nN),m_nN);
    //int m_nLongUpdate = MAX((Klong/m_nN),m_nN);

    //update counters for the refresh rate
    int m_nLongCounter = _nLongCounter;
    int m_nMidCounter = _nMidCounter;
    int m_nShortCounter = _nShortCounter;

    _nShortCounter++;//0,1,...,m_nShortUpdate-1
    _nMidCounter++;
    _nLongCounter++;
    if ( _nShortCounter >= m_nShortUpdate )
    {
        _nShortCounter = 0;
    }
    if ( _nMidCounter >= m_nMidUpdate )
    {
        _nMidCounter = 0;
    }
    if ( _nLongCounter >= m_nLongUpdate )
    {
        _nLongCounter = 0;
    }

    //go through the image
    long i = 0;
    for ( long y = 0; y < _src.rows; y++ )
    {
        for ( long x = 0; x < _src.cols; x++ )
        {
            const uchar *data = _src.ptr( ( int )y, ( int )x );

            //update model+ background subtract
            uchar include = 0;
            int result = _cvCheckPixelBackgroundNP( i, data, nchannels,
                                                    m_nN, m_aModel, m_fTb, m_nkNN, m_fTau, m_bShadowDetection, include );

            _cvUpdatePixelBackgroundNP( i, data, nchannels,
                                        m_nN, m_aModel,
                                        m_nNextLongUpdate,
                                        m_nNextMidUpdate,
                                        m_nNextShortUpdate,
                                        m_aModelIndexLong,
                                        m_aModelIndexMid,
                                        m_aModelIndexShort,
                                        m_nLongCounter,
                                        m_nMidCounter,
                                        m_nShortCounter,
                                        m_nLongUpdate,
                                        m_nMidUpdate,
                                        m_nShortUpdate,
                                        include
                                      );
            switch ( result )
            {
            case 0:
                //foreground
                *_dst.ptr( ( int )y, ( int )x ) = 255;
                break;
            case 1:
                //background
                *_dst.ptr( ( int )y, ( int )x ) = 0;
                break;
            case 2:
                //shadow
                *_dst.ptr( ( int )y, ( int )x ) = nShadowDetection;
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

    icvUpdatePixelBackgroundNP( image, fgmask,
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
                                nN,
                                ( float )learningRate,
                                fTb,
                                nkNN,
                                fTau,
                                bShadowDetection,
                                nShadowDetection
                              );
}

void MMESKNN::getBackgroundImage( cv::Mat backgroundImage ) const
{
    /* CV_INSTRUMENT_REGION() */

    int nchannels = CV_MAT_CN( frameType );
    //CV_Assert( nchannels == 3 );
    cv::Mat meanBackground( frameSize, CV_8UC3, cv::Scalar::all( 0 ) );

    int ndata = nchannels + 1;
    int modelstep = ( ndata * nN * 3 );

    const uchar *pbgmodel = bgmodel.ptr( 0 );
    for ( int row = 0; row < meanBackground.rows; row++ )
    {
        for ( int col = 0; col < meanBackground.cols; col++ )
        {
            for ( int n = 0; n < nN * 3; n++ )
            {
                const uchar *mean_m = &pbgmodel[n * ndata];
                if ( mean_m[nchannels] )
                {
                    meanBackground.at<cv::Vec3b>( row, col ) = cv::Vec3b( mean_m );
                    break;
                }
            }
            pbgmodel = pbgmodel + modelstep;
        }
    }

    switch ( CV_MAT_CN( frameType ) )
    {
    case 1:
    {
        std::vector<cv::Mat> channels;
        split( meanBackground, channels );
        channels[0].copyTo( backgroundImage );
        break;
    }
    case 3:
    {
        meanBackground.copyTo( backgroundImage );
        break;
    }
    default:
        CV_Error( cv::Error::StsUnsupportedFormat, "" );
    }
}
