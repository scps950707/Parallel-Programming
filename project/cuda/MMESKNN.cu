#include <opencv2/opencv.hpp>
#include <opencv2/video/background_segm.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <stdlib.h>
#include <iostream>
#include <chrono>
#include "MMESKNN.hpp"
using namespace std;
using namespace cv;

//{ to do - paralelization ...
//struct KNNInvoker....
__device__ void _cvUpdatePixelBackgroundNP(
    const uchar *currPixel,
    int channels,
    int nSample,
    bool  *flag,
    uchar *Model,
    uchar *ModelIndexLong,
    uchar *ModelIndexMid,
    uchar *ModelIndexShort,
    bool LongUpdate,
    bool MidUpdate,
    bool ShortUpdate,
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

    // Long update? --------------------------------------------------------------------------------------
    if ( LongUpdate )
    {
        // add the oldest pixel from Mid to the list of values (for each color)
        Model[offsetLong]     = Model[offsetMid];
        Model[offsetLong + 1] = Model[offsetMid + 1];
        Model[offsetLong + 2] = Model[offsetMid + 2];
        flag[flagoffsetLong]  = flag[flagoffsetMid];
        // increase the index
        *ModelIndexLong = ( *ModelIndexLong >= ( nSample - 1 ) ) ? 0 : ( *ModelIndexLong + 1 );
    }

    // Mid update? --------------------------------------------------------------------------------------
    if ( MidUpdate )
    {
        // add this pixel to the list of values (for each color)
        Model[offsetMid]     = Model[offsetShort];
        Model[offsetMid + 1] = Model[offsetShort + 1];
        Model[offsetMid + 2] = Model[offsetShort + 2];
        flag[flagoffsetMid]  = flag[flagoffsetShort];
        // increase the index
        *ModelIndexMid = ( *ModelIndexMid >= ( nSample - 1 ) ) ? 0 : ( *ModelIndexMid + 1 );
    }

    // Short update? --------------------------------------------------------------------------------------
    if ( ShortUpdate )
    {
        // add this pixel to the list of values (for each color)
        Model[offsetShort]     = currPixel[0];
        Model[offsetShort + 1] = currPixel[1];
        Model[offsetShort + 2] = currPixel[2];
        flag[flagoffsetShort]  = include;
        // increase the index
        *ModelIndexShort = ( *ModelIndexShort >= ( nSample - 1 ) ) ? 0 : ( *ModelIndexShort + 1 );
    }
}

__device__ int _cvCheckPixelBackgroundNP(
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
                    return 1;//background ->exit , dst[pixel]=0
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
            if ( flag[n] ) //check only background
            {
                float numerator = Model[n * channels] * currPixel[0]
                                  + Model[n * channels + 1] * currPixel[1]
                                  + Model[n * channels + 2] * currPixel[2];
                float denominator = Model[n * channels] * Model[n * channels]
                                    + Model[n * channels + 1] * Model[n * channels + 1]
                                    + Model[n * channels + 2] * Model[n * channels + 2];

                // no division by zero allowed
                if ( denominator == 0 )
                {
                    return 0;
                }

                // if tau < a < 1 then also check the color distortion
                if ( numerator <= denominator && numerator >= tau * denominator )
                {
                    float a = numerator / denominator;
                    float dist2a = 0.0f, dD;

                    dD = a * Model[n * channels] - currPixel[0];
                    dist2a += dD * dD;
                    dD = a * Model[n * channels + 1] - currPixel[1];
                    dist2a += dD * dD;
                    dD = a * Model[n * channels + 2] - currPixel[2];
                    dist2a += dD * dD;

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
    return 0; //dst[pixel]=255
}

__global__ void icvUpdatePixelBackgroundNP(
    int cols,
    int rows,
    int channels,
    int totalPixels,
    uchar *srcData,
    uchar *dst,
    bool *flag,
    uchar *Model,
    uchar *ModelIndexLong,
    uchar *ModelIndexMid,
    uchar *ModelIndexShort,
    int NextLongUpdate,
    int NextMidUpdate,
    int NextShortUpdate,
    int LongCounter,
    int MidCounter,
    int ShortCounter,
    int nSample,
    float Tb,
    int kNN,
    float Tau,
    bool ShadowDetection,
    uchar ShadowValue
)
{
    /* 2D */
    int posCol = blockIdx.x * blockDim.x + threadIdx.x;
    int posRow = blockIdx.y * blockDim.y + threadIdx.y;
    int posPixel = cols * ( posRow - 1 ) + posCol;

    /* 1D */
    /* int posPixel = blockIdx.x * blockDim.x + threadIdx.x; */

    uchar *currPixel = srcData + posPixel * channels;

    /* 2D */
    if ( posPixel < totalPixels && posCol < cols && posRow < rows )

        /* 1D */
        /* if ( posPixel < totalPixels) */
    {
        // int posPixel = ncols * y + x;
        /* start addr of current pixel */

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
            ModelIndexLong + posPixel,
            ModelIndexMid + posPixel,
            ModelIndexShort + posPixel,
            NextLongUpdate == LongCounter,
            NextMidUpdate == MidCounter,
            NextShortUpdate == ShortCounter,
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



void MMESKNN::apply( cv::Mat &image, cv::Mat &dst, double learningRate )
{
    bool needToInitialize = nframes == 0 || learningRate >= 1 || image.size() != frameSize || image.type() != frameType;
    if ( needToInitialize )
    {
        initialize( image.size(), image.type() );
    }

    dst.create( image.size(), CV_8UC1 );

    ++nframes;
    learningRate = learningRate >= 0 && nframes > 1 ? learningRate : 1. / std::min( 2 * nframes, history );
    CV_Assert( learningRate >= 0 );

    // recalculate update rates - in case alpha is changed
    // calculate update parameters (using alpha)
    int Kshort, Kmid, Klong;
    //approximate exponential learning curve
    Kshort = ( int )( log( 0.7 ) / log( 1 - learningRate ) ) + 1; //Kshort
    Kmid   = ( int )( log( 0.4 ) / log( 1 - learningRate ) ) - Kshort + 1; //Kmid
    Klong  = ( int )( log( 0.1 ) / log( 1 - learningRate ) ) - Kshort - Kmid + 1; //Klong

    //refresh rates
    int ShortUpdate = ( Kshort / nSample ) + 1;
    int MidUpdate   = ( Kmid   / nSample ) + 1;
    int LongUpdate  = ( Klong  / nSample ) + 1;

    if ( nLongCounter == ( LongUpdate - 1 ) )
    {
        nNextLongUpdate = rand() % LongUpdate;
    }
    if ( nMidCounter == ( MidUpdate - 1 ) )
    {
        nNextMidUpdate = rand() % MidUpdate;
    }
    if ( nShortCounter == ( ShortUpdate - 1 ) )
    {
        nNextShortUpdate = rand() % ShortUpdate;
    }
    // cuda
    //rows:size().height ; cols:size().width
    int totalPixels = image.rows * image.cols;
    cudaMemcpy( imageData, image.ptr(), sizeof( uchar ) * totalPixels * image.channels(), cudaMemcpyHostToDevice );

    /* 1D */
    /* icvUpdatePixelBackgroundNP <<< ( totalPixels + 255 ) / 256, 256 >>> ( */
    /* icvUpdatePixelBackgroundNP <<< ( totalPixels + 1023) / 1024, 1024 >>> ( */

    /* 2D */
    /* dim3 threadsPerBlock( 32, 32 ); */
    /* our video resolution 16:9 */
    dim3 threadsPerBlock( 32, 8 );
    dim3 numBlocks( image.cols + threadsPerBlock.x - 1 / threadsPerBlock.x, image.rows + threadsPerBlock.y - 1 / threadsPerBlock.y );
    icvUpdatePixelBackgroundNP <<<numBlocks, threadsPerBlock>>> (

        image.cols,
        image.rows,
        image.channels(),
        totalPixels,
        imageData,
        dstData,
        flag,
        bgmodel,
        aModelIndexLong,
        aModelIndexMid,
        aModelIndexShort,
        nNextLongUpdate,
        nNextMidUpdate,
        nNextShortUpdate,
        nLongCounter,
        nMidCounter,
        nShortCounter,
        nSample,
        fTb,
        nkNN,
        fTau,
        ShadowDetection, // 1: do ShadowDetection
        ShadowValue // default = (uchar) 127
    );

    cudaMemcpy( dst.ptr(),   dstData,   sizeof( uchar ) * totalPixels, cudaMemcpyDeviceToHost );

    //update counters for the refresh rate
    //0,1,...,ShortUpdate-1
    if ( ++nShortCounter >= ShortUpdate )
    {
        nShortCounter = 0;
    }
    if ( ++nMidCounter >= MidUpdate )
    {
        nMidCounter = 0;
    }
    if ( ++nLongCounter >= LongUpdate )
    {
        nLongCounter = 0;
    }
}

int main( int argc, char *argv[] )
{

    Mat frame;
    Mat output;
    Mat writeout;

    auto start = std::chrono::system_clock::now();
    MMESKNN *BG = new MMESKNN();
    /* Ptr<BackgroundSubtractor> BG = createBackgroundSubtractorMOG2(); */
    /* Ptr<BackgroundSubtractor> BG = createBackgroundSubtractorKNN(); */

    VideoCapture input( argv[1] );
    if ( argc == 3 )
    {
        input.set( CV_CAP_PROP_POS_FRAMES, atoi( argv[2] ) * 30 );
    }
    VideoWriter writer;
    input.read(frame);
    writer.open( "./output.avi", cv::VideoWriter::fourcc( 'M', 'J', 'P', 'G' ), 30.0, frame.size() );
    std::chrono::duration<double> BGtime = std::chrono::duration<double>::zero();
    while ( true )
    {
        if ( !( input.read( frame ) ) ) //get one frame form video
        {
            break;
        }
        auto t1 = std::chrono::system_clock::now();
        BG->apply( frame, output );
        auto t2 = std::chrono::system_clock::now();
        BGtime += t2 - t1;
        /* imshow( "Origin", frame ); */
        /* imshow( "KNN", output ); */
        /* if ( waitKey( 30 ) >= 0 ) */
        /* { */
        /*     break; */
        /* } */
        cv::cvtColor( output, writeout, CV_GRAY2BGR );
        writer << writeout;
    }
    std::chrono::duration<double> totalTime = std::chrono::system_clock::now() - start;
    cout << "BG time: " << BGtime.count() << "s\n";
    cout << "total time: " << totalTime.count() << "s\n";
    delete BG;
}
