#include <opencv2/video/background_segm.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "MMESKNN.hpp"
using namespace std;
using namespace cv;

int main( int argc, char *argv[] )
{

    Mat frame;
    Mat output;

    MMESKNN *BG = new MMESKNN();
    /* Ptr<BackgroundSubtractor> BG = createBackgroundSubtractorMOG2(); */
    /* Ptr<BackgroundSubtractor> BG = createBackgroundSubtractorKNN(); */

    VideoCapture input( argv[1] );
    while ( true )
    {
        Mat cameraFrame;
        if ( !( input.read( frame ) ) ) //get one frame form video
        {
            break;
        }
        BG->apply( frame, output );
        imshow( "Origin", frame );
        imshow( "KNN", output );
        if ( waitKey( 30 ) >= 0 )
        {
            break;
        }
    }
    delete BG;
}
