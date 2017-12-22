#ifndef __MMESKNN_HPP__
#define __MMESKNN_HPP__

#include <opencv2/opencv.hpp>

// default parameters of gaussian background detection algorithm
static const int defaultHistory2 = 500; // Learning rate; alpha = 1/defaultHistory2
static const int defaultNsamples = 7; // number of samples saved in memory
static const float defaultDist2Threshold = 20.0f * 20.0f; //threshold on distance from the sample

// additional parameters
static const unsigned char defaultnShadowDetection2 = ( unsigned char )127; // value to use in the segmentation mask for shadows, set 0 not to do shadow detection
static const float defaultfTau = 0.5f; // Tau - shadow threshold, see the paper for explanation

class MMESKNN
{
public:
    //! the default constructor
    MMESKNN()
    {
        frameSize = cv::Size( 0, 0 );
        frameType = 0;
        nframes = 0;
        history = defaultHistory2;

        //set parameters
        // N - the number of samples stored in memory per model
        nN = defaultNsamples;

        //kNN - k nearest neighbour - number on NN for detecting background - default K=[0.1*nN]
        nkNN = MAX( 1, cvRound( 0.1 * nN * 3 + 0.40 ) );

        //Tb - Threshold Tb*kernelwidth
        fTb = defaultDist2Threshold;

        // Shadow detection
        bShadowDetection = 1;//turn on
        nShadowDetection =  defaultnShadowDetection2;
        fTau = defaultfTau;// Tau - shadow threshold
        nLongCounter = 0;
        nMidCounter = 0;
        nShortCounter = 0;
    }
    //! the full constructor that takes the length of the history,
    // the number of gaussian mixtures, the background ratio parameter and the noise strength
    MMESKNN( int _history,  float _dist2Threshold, bool _bShadowDetection = true )
    {
        frameSize = cv::Size( 0, 0 );
        frameType = 0;

        nframes = 0;
        history = _history > 0 ? _history : defaultHistory2;

        //set parameters
        // N - the number of samples stored in memory per model
        nN = defaultNsamples;
        //kNN - k nearest neighbour - number on NN for detcting background - default K=[0.1*nN]
        nkNN = MAX( 1, cvRound( 0.1 * nN * 3 + 0.40 ) );

        //Tb - Threshold Tb*kernelwidth
        fTb = _dist2Threshold > 0 ? _dist2Threshold : defaultDist2Threshold;

        bShadowDetection = _bShadowDetection;
        nShadowDetection =  defaultnShadowDetection2;
        fTau = defaultfTau;
        nLongCounter = 0;
        nMidCounter = 0;
        nShortCounter = 0;
    }
    //! the destructor
    ~MMESKNN() {}
    //! the update operator
    void apply( cv::Mat &image, cv::Mat &fgmask, double learningRate = -1 );

    //! computes a background image which are the mean of all background gaussians
    void getBackgroundImage( cv::Mat backgroundImage ) const;

    //! re-initialization method
    void initialize( cv::Size _frameSize, int _frameType )
    {
        frameSize = _frameSize;
        frameType = _frameType;
        nframes = 0;

        int nchannels = CV_MAT_CN( frameType );
        CV_Assert( nchannels <= CV_CN_MAX );

        // Reserve memory for the model
        int size = frameSize.height * frameSize.width;
        // for each sample of 3 speed pixel models each pixel bg model we store ...
        // values + flag (nchannels+1 values)
        bgmodel.create( 1, ( nN * 3 ) * ( nchannels + 1 )* size, CV_8U );
        bgmodel = cv::Scalar::all( 0 );

        //index through the three circular lists
        aModelIndexShort.create( 1, size, CV_8U );
        aModelIndexMid.create( 1, size, CV_8U );
        aModelIndexLong.create( 1, size, CV_8U );
        //when to update next
        nNextShortUpdate.create( 1, size, CV_8U );
        nNextMidUpdate.create( 1, size, CV_8U );
        nNextLongUpdate.create( 1, size, CV_8U );

        //Reset counters
        nShortCounter = 0;
        nMidCounter = 0;
        nLongCounter = 0;

        aModelIndexShort = cv::Scalar::all( 0 ); //random? //((m_nN)*rand())/(RAND_MAX+1);//0...m_nN-1
        aModelIndexMid = cv::Scalar::all( 0 );
        aModelIndexLong = cv::Scalar::all( 0 );
        nNextShortUpdate = cv::Scalar::all( 0 );
        nNextMidUpdate = cv::Scalar::all( 0 );
        nNextLongUpdate = cv::Scalar::all( 0 );
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
        return nN;
    }
    void setNSamples( int _nN )
    {
        nN = _nN;    //needs reinitialization!
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
        return bShadowDetection;
    }
    void setDetectShadows( bool detectshadows )
    {
        bShadowDetection = detectshadows;
    }

    int getShadowValue() const
    {
        return nShadowDetection;
    }
    void setShadowValue( int value )
    {
        nShadowDetection = ( uchar )value;
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
    int nN;//totlal number of samples
    int nkNN;//number on NN for detcting background - default K=[0.1*nN]

    //shadow detection parameters
    bool bShadowDetection;//default 1 - do shadow detection
    unsigned char nShadowDetection;//do shadow detection - insert this value as the detection result - 127 default value
    float fTau;
    // Tau - shadow threshold. The shadow is detected if the pixel is darker
    //version of the background. Tau is a threshold on how much darker the shadow can be.
    //Tau= 0.5 means that if pixel is more than 2 times darker then it is not shadow
    //See: Prati,Mikic,Trivedi,Cucchiara,"Detecting Moving Shadows...",IEEE PAMI,2003.

    //model data
    int nLongCounter;//circular counter
    int nMidCounter;
    int nShortCounter;
    cv::Mat bgmodel; // model data pixel values
    cv::Mat aModelIndexShort;// index into the models
    cv::Mat aModelIndexMid;
    cv::Mat aModelIndexLong;
    cv::Mat nNextShortUpdate;//random update points per model
    cv::Mat nNextMidUpdate;
    cv::Mat nNextLongUpdate;
};

#endif
