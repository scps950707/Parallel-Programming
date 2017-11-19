/*
 * Author:         scps950707
 * Email:          scps950707@gmail.com
 * Created:        2017-11-18 17:20
 * Last Modified:  2017-11-19 17:22
 * Filename:       integrate.c
 */
#include <stdio.h>
#include <math.h>
#include <mpi.h>

#define PI 3.1415926535

int main( int argc, char **argv )
{
    long long i, num_intervals;
    int rank = 0, size = 0;
    double rect_width = 0, area = 0, sum = 0, x_middle = 0, lsum = 0;
    MPI_Init( &argc, &argv );
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );
    MPI_Comm_size( MPI_COMM_WORLD, &size );
#ifdef __DEBUG__
    double begin = MPI_Wtime();
#endif
    sscanf( argv[1], "%llu", &num_intervals );

    rect_width = PI / num_intervals;

    int start = 1 + rank;
    for ( i = start; i < num_intervals + 1; i += size )
    {
        /* find the middle of the interval on the X-axis. */
        x_middle = ( i - 0.5 ) * rect_width;
        area = sin( x_middle ) * rect_width;
        lsum += area;
    }
    MPI_Reduce( &lsum, &sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD );
    if ( rank == 0 )
    {
        printf( "The total area is: %f\n", ( float )sum );
#ifdef __DEBUG__
        printf( "Time:%.3fs\n", MPI_Wtime() - begin );
#endif
    }
    MPI_Finalize();
    return 0;
}
