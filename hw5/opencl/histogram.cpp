/*
 * Author:         scps950707
 * Email:          scps950707@gmail.com
 * Created:        2017-12-21 16:30
 * Last Modified:  2017-12-22 01:22
 * Filename:       histogram.cpp
 */

#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <chrono>
#include <CL/cl.h>

using namespace std;

/* {{{static const char *getErrorString( cl_int error ) */
static const char *getErrorString( cl_int error )
{
    switch ( error )
    {
    // run-time and JIT compiler errors
    case 0:
        return "CL_SUCCESS";
    case -1:
        return "CL_DEVICE_NOT_FOUND";
    case -2:
        return "CL_DEVICE_NOT_AVAILABLE";
    case -3:
        return "CL_COMPILER_NOT_AVAILABLE";
    case -4:
        return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
    case -5:
        return "CL_OUT_OF_RESOURCES";
    case -6:
        return "CL_OUT_OF_HOST_MEMORY";
    case -7:
        return "CL_PROFILING_INFO_NOT_AVAILABLE";
    case -8:
        return "CL_MEM_COPY_OVERLAP";
    case -9:
        return "CL_IMAGE_FORMAT_MISMATCH";
    case -10:
        return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
    case -11:
        return "CL_BUILD_PROGRAM_FAILURE";
    case -12:
        return "CL_MAP_FAILURE";
    case -13:
        return "CL_MISALIGNED_SUB_BUFFER_OFFSET";
    case -14:
        return "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";
    case -15:
        return "CL_COMPILE_PROGRAM_FAILURE";
    case -16:
        return "CL_LINKER_NOT_AVAILABLE";
    case -17:
        return "CL_LINK_PROGRAM_FAILURE";
    case -18:
        return "CL_DEVICE_PARTITION_FAILED";
    case -19:
        return "CL_KERNEL_ARG_INFO_NOT_AVAILABLE";

    // compile-time errors
    case -30:
        return "CL_INVALID_VALUE";
    case -31:
        return "CL_INVALID_DEVICE_TYPE";
    case -32:
        return "CL_INVALID_PLATFORM";
    case -33:
        return "CL_INVALID_DEVICE";
    case -34:
        return "CL_INVALID_CONTEXT";
    case -35:
        return "CL_INVALID_QUEUE_PROPERTIES";
    case -36:
        return "CL_INVALID_COMMAND_QUEUE";
    case -37:
        return "CL_INVALID_HOST_PTR";
    case -38:
        return "CL_INVALID_MEM_OBJECT";
    case -39:
        return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
    case -40:
        return "CL_INVALID_IMAGE_SIZE";
    case -41:
        return "CL_INVALID_SAMPLER";
    case -42:
        return "CL_INVALID_BINARY";
    case -43:
        return "CL_INVALID_BUILD_OPTIONS";
    case -44:
        return "CL_INVALID_PROGRAM";
    case -45:
        return "CL_INVALID_PROGRAM_EXECUTABLE";
    case -46:
        return "CL_INVALID_KERNEL_NAME";
    case -47:
        return "CL_INVALID_KERNEL_DEFINITION";
    case -48:
        return "CL_INVALID_KERNEL";
    case -49:
        return "CL_INVALID_ARG_INDEX";
    case -50:
        return "CL_INVALID_ARG_VALUE";
    case -51:
        return "CL_INVALID_ARG_SIZE";
    case -52:
        return "CL_INVALID_KERNEL_ARGS";
    case -53:
        return "CL_INVALID_WORK_DIMENSION";
    case -54:
        return "CL_INVALID_WORK_GROUP_SIZE";
    case -55:
        return "CL_INVALID_WORK_ITEM_SIZE";
    case -56:
        return "CL_INVALID_GLOBAL_OFFSET";
    case -57:
        return "CL_INVALID_EVENT_WAIT_LIST";
    case -58:
        return "CL_INVALID_EVENT";
    case -59:
        return "CL_INVALID_OPERATION";
    case -60:
        return "CL_INVALID_GL_OBJECT";
    case -61:
        return "CL_INVALID_BUFFER_SIZE";
    case -62:
        return "CL_INVALID_MIP_LEVEL";
    case -63:
        return "CL_INVALID_GLOBAL_WORK_SIZE";
    case -64:
        return "CL_INVALID_PROPERTY";
    case -65:
        return "CL_INVALID_IMAGE_DESCRIPTOR";
    case -66:
        return "CL_INVALID_COMPILER_OPTIONS";
    case -67:
        return "CL_INVALID_LINKER_OPTIONS";
    case -68:
        return "CL_INVALID_DEVICE_PARTITION_COUNT";

    // extension errors
    case -1000:
        return "CL_INVALID_GL_SHAREGROUP_REFERENCE_KHR";
    case -1001:
        return "CL_PLATFORM_NOT_FOUND_KHR";
    case -1002:
        return "CL_INVALID_D3D10_DEVICE_KHR";
    case -1003:
        return "CL_INVALID_D3D10_RESOURCE_KHR";
    case -1004:
        return "CL_D3D10_RESOURCE_ALREADY_ACQUIRED_KHR";
    case -1005:
        return "CL_D3D10_RESOURCE_NOT_ACQUIRED_KHR";
    default:
        return "Unknown OpenCL error";
    }
}
/* }}} */

static void HandleError( cl_int err, string file, int line )
{
    if ( err != CL_SUCCESS )
    {
        cerr << getErrorString( err ) << " in " << file << " at " << line << endl;
        exit( EXIT_FAILURE );
    }
}
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

int main( void )
{
    cl_int ret;
    size_t global;
    size_t local;
    size_t workDim[3];
    size_t workMax;

    cl_device_id deviceId;
    cl_context context;
    cl_command_queue cmds;
    cl_program program;
    cl_kernel kernel;
    cl_platform_id platform;
    cl_mem input, output;

    unsigned int *histogram_results = new unsigned int[256 * 3];
    unsigned int i = 0, a, input_size;
    unsigned int jobs, jobsPerWork;
    fstream kernelSource( "histogram.cl", ios_base::in );
    fstream inFile( "input", ios_base::in );
    ofstream outFile( "0656017.out", ios_base::out );
    stringstream ss;
    ss << kernelSource.rdbuf();
    string codeStr( ss.str() );
    const char *codeCStr = codeStr.c_str();
    size_t codeLen = codeStr.length();

    std::fill( histogram_results, histogram_results + 256 * 3, 0 );

    inFile >> input_size;
    jobs = input_size / 3;
    unsigned int *image = new unsigned int[input_size];
    while ( inFile >> a )
    {
        image[i++] = a;
    }
#ifdef __DEBUG__
    auto t1 = chrono::high_resolution_clock::now();
    cout << "timer start" << '\n';
#endif
    HANDLE_ERROR( clGetPlatformIDs( 1, &platform, NULL ) );
    HANDLE_ERROR( clGetDeviceIDs( platform, CL_DEVICE_TYPE_GPU, 1, &deviceId, NULL ) );
    /*
     * Maximum number of work-items that can be specified
     * in each dimension of the work-group to clEnqueueNDRangeKernel
     */
    HANDLE_ERROR( clGetDeviceInfo( deviceId, CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof( workDim ), &workDim, NULL ) );

    workMax = workDim[0] * workDim[1] * workDim[2];
    if ( jobs > workMax )
    {
        jobsPerWork = jobs / workMax;
        if ( jobs % workMax != 0 )
        {
            jobsPerWork++;
        }
    }
    else
    {
        jobsPerWork = 1;
    }

    context = clCreateContext( 0, 1, &deviceId, NULL, NULL, &ret );
    HANDLE_ERROR( ret );

    /* cmds = clCreateCommandQueueWithProperties( context, deviceId, NULL, &ret ); */
    cmds = clCreateCommandQueue( context, deviceId, 0, &ret );
    HANDLE_ERROR( ret );

    program = clCreateProgramWithSource( context, 1, ( const char ** ) & ( codeCStr ), &codeLen, &ret );
    HANDLE_ERROR( ret );

    ret = clBuildProgram( program, 0, NULL, NULL, NULL, NULL );
    if ( ret != CL_SUCCESS )
    {
        char buildLog[16384];
        clGetProgramBuildInfo( program, deviceId, CL_PROGRAM_BUILD_LOG, sizeof( buildLog ), buildLog, NULL );
        cerr << buildLog << endl;
        clReleaseProgram( program );
        exit( EXIT_FAILURE );
    }

    kernel = clCreateKernel( program, "histogram", &ret );
    HANDLE_ERROR( ret );

    input = clCreateBuffer( context,  CL_MEM_READ_ONLY,  sizeof( unsigned ) * input_size, NULL, &ret );
    HANDLE_ERROR( ret );
    output = clCreateBuffer( context, CL_MEM_WRITE_ONLY, sizeof( unsigned ) * 256 * 3, NULL, &ret );
    HANDLE_ERROR( ret );

    HANDLE_ERROR( clEnqueueWriteBuffer( cmds, input, CL_TRUE, 0, sizeof( unsigned ) * input_size, image, 0, NULL, NULL ) );

    HANDLE_ERROR( clSetKernelArg( kernel, 0, sizeof( cl_mem ), &input ) );
    HANDLE_ERROR( clSetKernelArg( kernel, 1, sizeof( cl_mem ), &output ) );
    HANDLE_ERROR( clSetKernelArg( kernel, 2, sizeof( unsigned int ), &jobs ) );
    HANDLE_ERROR( clSetKernelArg( kernel, 3, sizeof( unsigned int ), &jobsPerWork ) );

    HANDLE_ERROR( clGetKernelWorkGroupInfo( kernel, deviceId, CL_KERNEL_WORK_GROUP_SIZE, sizeof( local ), &local, NULL ) );

    global = workMax;
#ifdef __DEBUG__
    cout << "glo:" << global << " loc:" << local << endl;
#endif
    HANDLE_ERROR( clEnqueueNDRangeKernel( cmds, kernel, 1, NULL, &global, &local, 0, NULL, NULL ) );
    HANDLE_ERROR( clFinish( cmds ) );
    HANDLE_ERROR( clEnqueueReadBuffer( cmds, output, CL_TRUE, 0, sizeof( unsigned ) * 256 * 3, histogram_results, 0, NULL, NULL ) );

#ifdef __DEBUG__
    auto t2 = chrono::high_resolution_clock::now();
    auto dur = chrono::duration_cast<chrono::milliseconds>( t2 - t1 );
    cout << "Time:" << dur.count() << "msec" << endl;
#endif

    for ( unsigned int i = 0; i < 256 * 3; ++i )
    {
        if ( i % 256 == 0 && i != 0 )
        {
            outFile << endl;
        }
        outFile << histogram_results[i] << ' ';
    }
    delete [] histogram_results;
    delete [] image;
    inFile.close();
    outFile.close();

    HANDLE_ERROR( clReleaseMemObject( input ) );
    HANDLE_ERROR( clReleaseMemObject( output ) );
    HANDLE_ERROR( clReleaseProgram( program ) );
    HANDLE_ERROR( clReleaseKernel( kernel ) );
    HANDLE_ERROR( clReleaseCommandQueue( cmds ) );
    HANDLE_ERROR( clReleaseContext( context ) );

    return EXIT_SUCCESS;
}
