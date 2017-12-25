#include <opencv2/video/background_segm.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <chrono>
using namespace std;
using namespace cv;

int main( int argc, char *argv[] )
{

    UMat frame;
    UMat output;
    Mat writeout;

    auto start = std::chrono::system_clock::now();
    //Ptr<BackgroundSubtractor> BG = createBackgroundSubtractorMOG2();
    Ptr<BackgroundSubtractor> BG = createBackgroundSubtractorKNN();

    VideoCapture input( argv[1] );
    input >> frame;
    if ( argc == 3 )
    {
        input.set( CV_CAP_PROP_POS_FRAMES, atoi( argv[2] ) * 30 );
    }
    VideoWriter writer;
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
        //BG->getBackgroundImage( output );
        BGtime += t2 - t1;
        //imshow( "Origin", frame );
        //imshow( "KNN", output );
        cv::cvtColor( output, writeout, CV_GRAY2BGR );
        writer << writeout;
    }
    std::chrono::duration<double> totalTime = std::chrono::system_clock::now() - start;
    cout << "BG time: " << BGtime.count() << "s\n";
    cout << "total time: " << totalTime.count() << "s\n";
}
