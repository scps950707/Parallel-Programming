/*
 * Author:         scps950707
 * Email:          scps950707@gmail.com
 * Created:        2017-11-18 17:20
 * Last Modified:  2017-11-22 17:04
 * Filename:       prime.c
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>

int isprime( long long int n )
{
    long long int i, squareroot;
    if ( n > 10 )
    {
        squareroot = ( long long int ) sqrt( n );
        for ( i = 3; i <= squareroot; i = i + 2 )
            if ( ( n % i ) == 0 )
            {
                return 0;
            }
        return 1;
    }
    else
    {
        return 0;
    }
}

int main( int argc, char *argv[] )
{
    long long int pc = 0,     /* prime counter */
                  lpc = 0,
                  foundone = 0, /* most recent prime found */
                  lfoundone = 0;
    int rank = 0, size = 0;
    long long int n, limit;
    MPI_Init( &argc, &argv );
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );
    MPI_Comm_size( MPI_COMM_WORLD, &size );
#ifdef __DEBUG__
    double begin = MPI_Wtime();
#endif
    sscanf( argv[1], "%llu", &limit );
    if ( rank == 0 )
    {
        printf( "Starting. Numbers to be scanned= %lld\n", limit );
    }
    int start = 11 + 2 * rank, step = 2 * size;
    for ( n = start; n <= limit; n += step )
    {
        if ( isprime( n ) )
        {
            ++lpc;
            lfoundone = n;
        }
    }
    MPI_Reduce( &lpc, &pc, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD );
    MPI_Reduce( &lfoundone, &foundone, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD );
    if ( rank == 0 )
    {
        pc += 4;   /* Assume (2,3,5,7) are counted here */
        printf( "Done. Largest prime is %lld Total primes %lld\n", foundone, pc );
#ifdef __DEBUG__
        printf( "Time:%.3fs\n", MPI_Wtime() - begin );
#endif
    }
    MPI_Finalize();
    return 0;
}
