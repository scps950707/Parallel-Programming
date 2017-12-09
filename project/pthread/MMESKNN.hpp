#ifndef __MMESKNN_HPP__
#define __MMESKNN_HPP__

#include <opencv2/opencv.hpp>

// default parameters of gaussian background detection algorithm
static const int defaultHistory = 500; // Learning rate; alpha = 1/defaultHistory
static const int defaultNsamples = 7; // number of samples saved in memory
static const float defaultDist2Threshold = 20.0f * 20.0f; //threshold on distance from the sample

class MMESKNN
{
public:
    //! the default constructor
    MMESKNN() : MMESKNN( defaultHistory, defaultDist2Threshold, true )
    {
    }
    //! the full constructor that takes the length of the history,
    // the number of gaussian mixtures, the background ratio parameter and the noise strength
    MMESKNN( int _history,  float _dist2Threshold, bool ShadowDetection = true ) :
        ShadowDetection( ShadowDetection )
    {
        frameSize = cv::Size( 0, 0 );
        frameType = 0;
        nframes = 0;
        history = _history > 0 ? _history : defaultHistory;

        //set parameters
        // N - the number of samples stored in memory per model
        nSample = defaultNsamples;
        //kNN - k nearest neighbour - number on NN for detcting background - default K=[0.1*nSample]
        nkNN = MAX( 1, cvRound( 0.1 * nSample * 3 + 0.40 ) );

        //Tb - Threshold Tb*kernelwidth
        fTb = _dist2Threshold > 0 ? _dist2Threshold : defaultDist2Threshold;

        ShadowValue = ( uchar )127; // value to use in the segmentation mask for shadows, set 0 not to do shadow detection
        fTau = 0.5; // Tau - shadow threshold, see the paper for explanation
        nLongCounter = 0;
        nMidCounter = 0;
        nShortCounter = 0;
        bgmodel = nullptr;
        flag = nullptr;
        aModelIndexShort = nullptr;
        aModelIndexMid = nullptr;
        aModelIndexLong = nullptr;
        nNextShortUpdate = nullptr;
        nNextMidUpdate = nullptr;
        nNextLongUpdate = nullptr;
    }
    //! the destructor
    ~MMESKNN()
    {
        delete[] bgmodel;
        delete[] flag;
        delete[] aModelIndexShort;
        delete[] aModelIndexMid;
        delete[] aModelIndexLong;
        delete[] nNextShortUpdate;
        delete[] nNextMidUpdate;
        delete[] nNextLongUpdate;
    }
    //! the update operator
    void apply( cv::Mat &image, cv::Mat &fgmask, double learningRate = -1 );

    //! re-initialization method
    void initialize( cv::Size frameSize, int frameType )
    {
        this->frameSize = frameSize;
        this->frameType = frameType;
        nframes = 0;

        int nchannels = CV_MAT_CN( frameType );
        CV_Assert( nchannels <= CV_CN_MAX );

        // Reserve memory for the model
        int totalPixels = frameSize.height * frameSize.width;
        // for each sample of 3 speed pixel models each pixel bg model we store ...
        bgmodel = new uchar[nSample * 3 * nchannels * totalPixels];
        std::fill( bgmodel, bgmodel + nSample * 3 * nchannels * totalPixels, 0 );
        flag = new bool[nSample * 3 * totalPixels];
        std::fill( flag, flag + nSample * 3 * totalPixels, false );

        //index through the three circular lists
        aModelIndexShort = new uchar[totalPixels];
        aModelIndexMid = new uchar[totalPixels];
        aModelIndexLong = new uchar[totalPixels];
        //when to update next
        nNextShortUpdate = new uchar[totalPixels];
        nNextMidUpdate = new uchar[totalPixels];
        nNextLongUpdate = new uchar[totalPixels];

        //Reset counters
        nShortCounter = 0;
        nMidCounter = 0;
        nLongCounter = 0;

        std::fill( aModelIndexShort, aModelIndexShort + totalPixels, 0 );
        std::fill( aModelIndexMid, aModelIndexMid + totalPixels, 0 );
        std::fill( aModelIndexLong, aModelIndexLong + totalPixels, 0 );
        std::fill( nNextShortUpdate, nNextShortUpdate + totalPixels, 0 );
        std::fill( nNextMidUpdate, nNextMidUpdate + totalPixels, 0 );
        std::fill( nNextLongUpdate, nNextLongUpdate + totalPixels, 0 );
    }

    int getHistory() const
    {
        return history;
    }
    void setHistory( int _nframes )
    {
        history = _nframes;
    }

    int getNSamples() const
    {
        return nSample;
    }
    void setNSamples( int _nN )
    {
        nSample = _nN;    //needs reinitialization!
    }

    int getkNNSamples() const
    {
        return nkNN;
    }
    void setkNNSamples( int _nkNN )
    {
        nkNN = _nkNN;
    }

    double getDist2Threshold() const
    {
        return fTb;
    }
    void setDist2Threshold( double _dist2Threshold )
    {
        fTb = ( float )_dist2Threshold;
    }

    bool getDetectShadows() const
    {
        return ShadowDetection;
    }
    void setDetectShadows( bool detectshadows )
    {
        ShadowDetection = detectshadows;
    }

    int getShadowValue() const
    {
        return ShadowValue;
    }
    void setShadowValue( int value )
    {
        ShadowValue = ( uchar )value;
    }

    double getShadowThreshold() const
    {
        return fTau;
    }
    void setShadowThreshold( double value )
    {
        fTau = ( float )value;
    }

protected:
    cv::Size frameSize;
    int frameType;
    int nframes;
    /////////////////////////
    //very important parameters - things you will change
    ////////////////////////
    int history;
    //alpha=1/history - speed of update - if the time interval you want to average over is T
    //set alpha=1/history. It is also usefull at start to make T slowly increase
    //from 1 until the desired T
    float fTb;
    //Tb - threshold on the squared distance from the sample used to decide if it is well described
    //by the background model or not. A typical value could be 2 sigma
    //and that is Tb=2*2*10*10 =400; where we take typical pixel level sigma=10

    /////////////////////////
    //less important parameters - things you might change but be carefull
    ////////////////////////
    int nSample;//totlal number of samples
    int nkNN;//number on NN for detcting background - default K=[0.1*nSample]

    //shadow detection parameters
    bool ShadowDetection;//default 1 - do shadow detection
    uchar ShadowValue;//do shadow detection - insert this value as the detection result - 127 default value
    float fTau;
    // Tau - shadow threshold. The shadow is detected if the pixel is darker
    //version of the background. Tau is a threshold on how much darker the shadow can be.
    //Tau= 0.5 means that if pixel is more than 2 times darker then it is not shadow
    //See: Prati,Mikic,Trivedi,Cucchiara,"Detecting Moving Shadows...",IEEE PAMI,2003.

    //model data
    int nLongCounter;//circular counter
    int nMidCounter;
    int nShortCounter;
    uchar *bgmodel; // model data pixel values
    bool *flag; // pixel is included in current model
    uchar *aModelIndexShort;// index into the models
    uchar *aModelIndexMid;
    uchar *aModelIndexLong;
    uchar *nNextShortUpdate;//random update points per model
    uchar *nNextMidUpdate;
    uchar *nNextLongUpdate;
};

#endif
