__kernel void histogram(
    __global unsigned int *image_data,
    __global unsigned int *ref_histogram_results,
    unsigned int jobs,
    unsigned int jobsPerWork )
{
    int c = get_global_id( 0 );
    size_t w = get_global_size( 0 );
    unsigned int idx;
    if ( c < 256 * 3 )
    {
        ref_histogram_results[c] = 0;
    }
    unsigned int i, j;
    for ( i = 0; i < jobsPerWork; i++ )
    {
        if ( ( c + ( i * w ) ) < jobs )
        {
            idx = image_data[( c + ( i * w ) ) * 3];
            atomic_inc( ref_histogram_results + idx );
            idx = image_data[( c + ( i * w ) ) * 3 + 1];
            atomic_inc( ref_histogram_results + idx + 256 );
            idx = image_data[( c + ( i * w ) ) * 3 + 2];
            atomic_inc( ref_histogram_results + idx + 512 );
        }
    }
}
