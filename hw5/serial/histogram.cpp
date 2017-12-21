#include "stdio.h"
#include "stdlib.h"
#include "string.h"
#include <fstream>
#include <iostream>
#include <chrono>
using namespace std;

unsigned int *histogram( unsigned int *image_data, unsigned int _size )
{

    unsigned int *img = image_data;
    unsigned int *ref_histogram_results;
    unsigned int *ptr;

    ref_histogram_results = ( unsigned int * )malloc( 256 * 3 * sizeof( unsigned int ) );
    ptr = ref_histogram_results;
    memset ( ref_histogram_results, 0x0, 256 * 3 * sizeof( unsigned int ) );

    // histogram of R
    for ( unsigned int i = 0; i < _size; i += 3 )
    {
        unsigned int index = img[i];
        ptr[index]++;
    }

    // histogram of G
    ptr += 256;
    for ( unsigned int i = 1; i < _size; i += 3 )
    {
        unsigned int index = img[i];
        ptr[index]++;
    }

    // histogram of B
    ptr += 256;
    for ( unsigned int i = 2; i < _size; i += 3 )
    {
        unsigned int index = img[i];
        ptr[index]++;
    }

    return ref_histogram_results;
}

int main( int argc, char const *argv[] )
{

    unsigned int *histogram_results;
    unsigned int i = 0, a, input_size;
    std::fstream inFile( "input", std::ios_base::in );
    std::ofstream outFile( "0656017.out", std::ios_base::out );

    inFile >> input_size;
    unsigned int *image = new unsigned int[input_size];
    while ( inFile >> a )
    {
        image[i++] = a;
    }

#ifdef __DEBUG__
    cout << "timer start" << '\n';
    auto t1 = chrono::high_resolution_clock::now();
#endif
    histogram_results = histogram( image, input_size );
#ifdef __DEBUG__
    auto t2 = chrono::high_resolution_clock::now();
    auto dur = chrono::duration_cast<chrono::milliseconds>( t2 - t1 );
    cout << " Time:" << dur.count() << "msec" << endl;
#endif
    for ( unsigned int i = 0; i < 256 * 3; ++i )
    {
        if ( i % 256 == 0 && i != 0 )
        {
            outFile << std::endl;
        }
        outFile << histogram_results[i] << ' ';
    }

    inFile.close();
    outFile.close();

    return 0;
}
