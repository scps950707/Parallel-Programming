/*
 * Author:         scps950707
 * Email:          scps950707@gmail.com
 * Created:        2017-10-23 17:19
 * Last Modified:  2017-10-24 17:40
 * Filename:       pi.cpp
 */

/*
 * After some experiments -O2 opimization make program too fast so that
 * I can't find speedup with threads, but without O2 flag the speed has
 * positive correlation with num of threads.
 */

#include<iostream>
#include<cstdlib>
#include<random>
#include<cstdint>
#include<pthread.h>
#include<chrono>
using namespace std;
random_device rd;
mt19937 gen( rd() );
uniform_real_distribution<double> dis( -1.0, 1.0 );

uint64_t number_in_circle = 0;
pthread_mutex_t mutexNCircle;

void *getPI( void *n )
{
    uint64_t number_of_tosses = *( static_cast<uint64_t *>( n ) );
    uint64_t localNumberInCircle = 0;
    while ( number_of_tosses-- )
    {
        double x = dis( gen );
        double y = dis( gen );
        if ( x * x + y * y <= 1.0 )
        {
            localNumberInCircle++;
        }
    }
    pthread_mutex_lock( &mutexNCircle );
    number_in_circle += localNumberInCircle;
    pthread_mutex_unlock( &mutexNCircle );
    return NULL;
}
int main( int argc, char *argv[] )
{
    if ( argc != 3 )
    {
        cout << "./pi [nCPU] [nTimes]" << endl;
        return EXIT_FAILURE;
    }
    int32_t nCPU = atoi( argv[1] );
    uint64_t nTosses = strtoull( argv[2], NULL, 10 );
    uint64_t nTossesPerThread = nTosses / nCPU;

    pthread_t *threadPools = ( pthread_t * )malloc( sizeof( pthread_t ) * nCPU );
    pthread_mutex_init( &mutexNCircle, NULL );
#ifdef __DEBUG__
    auto t1 = chrono::high_resolution_clock::now();
#endif
    for ( int32_t i = 0; i < nCPU; ++i )
    {
        pthread_create( &threadPools[i], NULL, getPI, &nTossesPerThread );
    }
    for ( int32_t i = 0; i < nCPU; ++i )
    {
        pthread_join( threadPools[i], NULL );
    }
#ifdef __DEBUG__
    auto t2 = chrono::high_resolution_clock::now();
    auto dur = chrono::duration_cast<chrono::milliseconds>( t2 - t1 );
    cout << "Compute PI with " << nCPU << " CPUs Time:" << dur.count() << "msec" << endl;
#endif
    pthread_mutex_destroy( &mutexNCircle );
    free( threadPools );

    cout << ( number_in_circle << 2 ) / static_cast<double>( nTosses ) << endl;
    return EXIT_SUCCESS;
}
