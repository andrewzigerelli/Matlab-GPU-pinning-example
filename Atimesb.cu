#include "mex.h"
#include "cuda.h"
#include "cublas_v2.h"

static int isPinned = 0;
static double *pinnedA = NULL;
static mwSize dim;
void cleanup(void) {
    cudaFree(pinnedA);
    mexPrintf("Removing pinned matrix A.\n");
}
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {

    const mwSize *dims;
    const mwSize *bdims;
    mwSize ndim;
    double *A, *b, *d_b, *d_y;
    double *y; 

    double *alpha;
    alpha = (double *) malloc(sizeof(double));
    *alpha = 1;
    double *beta;
    beta = (double *) malloc(sizeof(double));
    *beta = 0;

    /* setup cublas */
    cublasStatus_t stat;
    cublasHandle_t handle;

    stat = cublasCreate(&handle);
    if ( stat != CUBLAS_STATUS_SUCCESS ) {
        mexErrMsgIdAndTxt(" MATLAB:Atimesb:cudaFailure", 
                "cublas initialization failed");
    }

    /* input should be matrix A and vector b, if first run
        else, input may just be vector b */
    if(nrhs == 1 && isPinned == 0) {
       mexErrMsgIdAndTxt( "MATLAB:Atimesb:invalidInputs",
               "You must supply the matrix A for the first run, i.e. y=Atimesb(A,b)");
    }
    else if (nrhs < 1) {
        mexErrMsgIdAndTxt( "MATLAB:Atimesb:invalidNumInputs",
                "input is required");
    }
    else if (nrhs == 2 && isPinned == 1) {
        mexPrintf("Warning: A is already on GPU; ignoring this A input.\n");
    }

    if(!isPinned) {
        //pin square matrix to GPU, size dim * dim
        ndim = mxGetNumberOfDimensions(prhs[0]);
        dims = mxGetDimensions(prhs[0]);
        if( (dims[0] != dims[1]) || (ndim !=2) ){
            mexErrMsgIdAndTxt( "MATLAB:Atimesb:invalidDims", 
                    "A must be a square matrix");
        }
        dim = dims[0];
        A = mxGetPr(prhs[0]);
        cudaMalloc((void **)&pinnedA, dim*dim*sizeof(double));
        /* we should make sure cudaMalloc doesn't fail */
        cudaMemcpy(pinnedA, A, dim*dim*sizeof(double), cudaMemcpyHostToDevice);
        mexPrintf("Copied A to GPU.\n");

        //Get vector b and transfer to GPU, size dim * 1
        b = mxGetPr(prhs[1]);
        bdims = mxGetDimensions(prhs[1]);
        if (bdims[0] != dim) {
            mexErrMsgIdAndTxt( "MATLAB:Atimesb:invalidDim",
                    "b must have same leading dimension as A");
        }
        if (bdims[1] != 1) {
            mexErrMsgIdAndTxt( "MATLAB:Atimesb:invalidDim",
                    "b must second dimension equal to 1");
        }
        ndim = mxGetNumberOfDimensions(prhs[1]);
        if(ndim !=2) {
            mexErrMsgIdAndTxt( "MATLAB:Atimesb:invalidDims", 
                    "b must be a vector");
        }

        cudaMalloc((void **)&d_b, dim*sizeof(double));
        cudaMemcpy(d_b, b, dim*sizeof(double), cudaMemcpyHostToDevice);
        mexPrintf("Copied b to GPU.\n");

        //Calculate y=A*b
        /* see Dgemv documentation for explanation of params */
        cudaMalloc((void **)&d_y, dim*sizeof(double));
        mexPrintf("Before gemv!\n");
        stat = cublasDgemv(handle, CUBLAS_OP_N, 
                (int) dim, (int) dim, 
                alpha, 
                pinnedA, (int) dim,
                d_b, 1, 
                beta, 
                d_y, 1);
        mexPrintf("After gemv!\n");
        
        //Copy output
        plhs[0] = mxCreateDoubleMatrix(dim,1,mxREAL);
        y = mxGetPr(plhs[0]);
        mexPrintf("Copying result to y\n");
        cudaMemcpy(y, d_y, dim*sizeof(double), cudaMemcpyDeviceToHost);
        mexPrintf("Done copying result to y\n");
       
        //cleanup
        cudaFree(d_b);
        cudaFree(d_y);


        //set pinned flag and register mex cleanup callback
        mexAtExit(cleanup);
        isPinned =1;
    }
    else {
        //get b vector and check dimensions
        if(nrhs==2) {
            b = mxGetPr(prhs[1]);
            bdims = mxGetDimensions(prhs[1]);
            ndim = mxGetNumberOfDimensions(prhs[1]);
        }
        else {
            b = mxGetPr(prhs[0]);
            bdims = mxGetDimensions(prhs[0]);
            ndim = mxGetNumberOfDimensions(prhs[0]);
        }
        if (bdims[0] != dim) {
            mexErrMsgIdAndTxt( "MATLAB:Atimesb:invalidDim",
                    "b must have same leading dimension as A");
        }
        if (bdims[1] != 1) {
            mexErrMsgIdAndTxt( "MATLAB:Atimesb:invalidDim",
                    "b must second dimension equal to 1");
        }
        if(ndim !=2) {
            mexErrMsgIdAndTxt( "MATLAB:Atimesb:invalidDims", 
                    "b must be a vector");
        }
        cudaMalloc((void **)&d_b, dim*sizeof(double));
        cudaMemcpy(d_b, b, dim*sizeof(double), cudaMemcpyHostToDevice);
        mexPrintf("Copied b to GPU.\n");

        //Calculate y=A*b
        /* see Dgemv documentation for explanation of params */
        cudaMalloc((void **)&d_y, dim*sizeof(double));
        mexPrintf("Before gemv!\n");
        stat = cublasDgemv(handle, CUBLAS_OP_N, 
                (int) dim, (int) dim, 
                alpha, 
                pinnedA, (int) dim,
                d_b, 1, 
                beta, 
                d_y, 1);
        mexPrintf("After gemv!\n");
        
        //Copy output
        plhs[0] = mxCreateDoubleMatrix(dim,1,mxREAL);
        y = mxGetPr(plhs[0]);
        mexPrintf("Copying result to y\n");
        cudaMemcpy(y, d_y, dim*sizeof(double), cudaMemcpyDeviceToHost);
        mexPrintf("Done copying result to y\n");
    }
}
