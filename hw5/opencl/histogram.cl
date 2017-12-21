__kernel void histogram(
    __global unsigned int *image_data,
    __global unsigned int *ref_histogram_results,
    unsigned int bound,
    unsigned int eachsize )
{
    int col = get_global_id( 0 );
    unsigned int idx;
    size_t width = get_global_size( 0 );
    if ( col < 256 * 3 )
    {
        ref_histogram_results[col] = 0;
    }
    unsigned int i, j;
    for ( i = 0; i < eachsize; i++ )
    {
        if ( ( col + ( i * width ) ) < bound )
        {
            idx = image_data[( col + ( i * width ) ) * 3];
            atomic_inc( ref_histogram_results + idx );
            idx = image_data[( col + ( i * width ) ) * 3 + 1];
            atomic_inc( ref_histogram_results + idx + 256 );
            idx = image_data[( col + ( i * width ) ) * 3 + 2];
            atomic_inc( ref_histogram_results + idx + 512 );
        }
    }
}
