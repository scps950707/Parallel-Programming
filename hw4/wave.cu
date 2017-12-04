/*
 * Author:         scps950707
 * Email:          scps950707@gmail.com
 * Created:        2017-12-04 12:09
 * Last Modified:  2017-12-04 18:42
 * Filename:       wave.cu
 * description: Serial Concurrent Wave Equation - C Version
 *              This program implements the concurrent wave equation
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define MAXPOINTS 1000000
#define MAXSTEPS 1000000
#define MINPOINTS 20
#define PI 3.14159265

static void HandleError( cudaError_t err, const char *file, int line )
{
    if ( err != cudaSuccess )
    {
        printf( "%s in %s at line %d\n", cudaGetErrorString( err ), file, line );
        exit( EXIT_FAILURE );
    }
}
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

int nsteps;
/* number of time steps */
int tpoints;
/* total points along string */
float H_currVal[MAXPOINTS + 2];
/* values at time t */


/**********************************************************************
 *	Checks input values from parameters
 *********************************************************************/
void check_param( void )
{
    char tchar[20];

    /* check number of points, number of iterations */
    while ( ( tpoints < MINPOINTS ) || ( tpoints > MAXPOINTS ) )
    {
        printf( "Enter number of points along vibrating string [%d-%d]: "
                , MINPOINTS, MAXPOINTS );
        scanf( "%s", tchar );
        tpoints = atoi( tchar );
        if ( ( tpoints < MINPOINTS ) || ( tpoints > MAXPOINTS ) )
            printf( "Invalid. Please enter value between %d and %d\n",
                    MINPOINTS, MAXPOINTS );
    }
    while ( ( nsteps < 1 ) || ( nsteps > MAXSTEPS ) )
    {
        printf( "Enter number of time steps [1-%d]: ", MAXSTEPS );
        scanf( "%s", tchar );
        nsteps = atoi( tchar );
        if ( ( nsteps < 1 ) || ( nsteps > MAXSTEPS ) )
        {
            printf( "Invalid. Please enter value between 1 and %d\n", MAXSTEPS );
        }
    }

    printf( "Using points = %d, steps = %d\n", tpoints, nsteps );

}
/**********************************************************************
 *     initialize points on line
 *     Update all values along line a specified number of times
 *********************************************************************/
__global__ void initAndUpdate( float *D_oldVal, float *D_currVal, int tpoints, int nsteps )
{
    int j = blockDim.x * blockIdx.x + threadIdx.x;
    if ( j < tpoints )
    {
        j += 1;
        /* Calculate initial values based on sine curve */
        /* Initialize old values array */
        float x = ( float )( j - 1 ) / ( tpoints - 1 );
        D_oldVal[j] = D_currVal[j] = sin ( 6.2831853f * x );
        int i;
        /* global endpoints */
        if ( ( j == 1 ) || ( j  == tpoints ) )
        {
            D_currVal[j] = 0.0;
        }
        else
        {
            /* Update values for each time step */
            for ( i = 1; i <= nsteps; i++ )
            {
                /* Update old values with new values */
                float newVal = ( 2.0 * D_currVal[j] ) - D_oldVal[j] + ( 0.09f * ( -2.0 ) * D_currVal[j] );
                D_oldVal[j] = D_currVal[j];
                D_currVal[j] = newVal;
            }
        }
    }
}

/**********************************************************************
 *     Print final results
 *********************************************************************/
void printfinal()
{
    int i;
    for ( i = 1; i <= tpoints; i++ )
    {
        printf( "%6.4f ", H_currVal[i] );
        if ( i % 10 == 0 )
        {
            printf( "\n" );
        }
    }
}

/**********************************************************************
 *	Main program
 *********************************************************************/
int main( int argc, char *argv[] )
{
    /* Error code to check return values for CUDA calls */

    sscanf( argv[1], "%d", &tpoints );
    sscanf( argv[2], "%d", &nsteps );
    check_param();

    int threadsPerBlock = 256;
    int blocksPerGrid = ( tpoints + threadsPerBlock - 1 ) / threadsPerBlock;
    float *D_currVal, *D_oldVal;

    HANDLE_ERROR( cudaMalloc( ( void ** )&D_currVal, sizeof( float ) * ( tpoints + 2 ) ) );
    HANDLE_ERROR( cudaMalloc( ( void ** )&D_oldVal, sizeof( float ) * ( tpoints + 2 ) ) );

    printf( "Initializing points on the line...\n" );
    printf( "Updating all points for all time steps...\n" );
#if __DEBUG__
    clock_t t = clock();
#endif
    initAndUpdate <<<blocksPerGrid, threadsPerBlock>>>( D_oldVal, D_currVal, tpoints, nsteps );
    HANDLE_ERROR( cudaMemcpy( H_currVal, D_currVal, sizeof( float ) * ( tpoints + 2 ), cudaMemcpyDeviceToHost ) );
#if __DEBUG__
    t = clock() - t;
#endif
    printf( "Printing final results...\n" );
    printfinal();
    printf( "\nDone.\n\n" );
#if __DEBUG__
    printf( "time:%f\n", ( float )t / CLOCKS_PER_SEC );
#endif

    HANDLE_ERROR( cudaFree( D_currVal ) );
    HANDLE_ERROR( cudaFree( D_oldVal ) );

    return EXIT_SUCCESS;
}
