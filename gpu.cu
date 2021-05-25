#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include<iostream>
//#include<opencv2/opencv.hpp>
//using namespace cv;
using namespace std;

typedef struct {
    float elements[3][3];
}Matrix;

typedef struct {
    float elements[16][3][3];
}MatrixCV2;

const int lenFilters = 3;
const int lenInputImage = 28;
const int NFiltersCV1 = 16;
const int lenOutImageCV1 = 26;
const int NFiltersCV2 = 32;
const int lenOutImageCV2 = 24;

__constant__ float d_ArrWeightsFC2[10][512];
__constant__ Matrix d_FiltersCV1[16];

//__constant__ int NFiltersCV2 = 32;
__constant__ int lenLayer12;

const int lenLayer1 = 18432;
const int lenLayer2 = 512;
const int lenOutLayer = 10;
int ex = 31;



/*
__global__ void relu(float ) {
    int i = threadIdx.x;
    int j = threadIdx.y;
    int z = threadIdx.z;


}*/


__global__ void CV1Kernel(float d_InputImage[28][28], int NFilters, int lenFilter, float d_OutImage[16][26][26]) {
    float val;
    __shared__ float Image[28][28];
    int f = blockIdx.x;
    val = 0;
    int i = threadIdx.x;
    int j = threadIdx.y;

    Image[i][j] = d_InputImage[i][j];

    __syncthreads();
    if (threadIdx.x < lenOutImageCV1 && threadIdx.y < lenOutImageCV1) {
        for (int p = 0; p < lenFilter; p++) {
            for (int q = 0; q < lenFilter; q++) {
                val = val + d_FiltersCV1[f].elements[p][q] * Image[i + p][j + q];
            }
        }
        d_OutImage[f][i][j] = val;
    }
}

__global__ void CV1Kernel2(float d_InputImage[28][28], int NFilters, int lenFilter, float d_OutImage[16][26][26]) {
    float val;
    __shared__ float Image[28][28];
    int f = blockIdx.x;
    val = 0;
    int i = threadIdx.x;
    int j = threadIdx.y;

    Image[i][j] = d_InputImage[i][j];

    __syncthreads();
    if (threadIdx.x < lenOutImageCV1 && threadIdx.y < lenOutImageCV1) {
        for (int p = 0; p < lenFilter; p++) {
            for (int q = 0; q < lenFilter; q++) {
                val = val + d_FiltersCV1[f].elements[p][q] * Image[i + p][j + q];
            }
        }
        d_OutImage[f][i][j] = val;
    }
}

__global__ void CV2Kernel(float d_InputImage[16][26][26], MatrixCV2* FiltersCV2, float d_OutImageCV2[32][24][24]) {
    int f = blockIdx.x;
    int r = threadIdx.x;
    int c = threadIdx.y;
    float sum = 0;
    //printf(" %d ", NFiltersCV1);
    for (int i = 0; i < NFiltersCV1; i++) {
        for (int j = 0; j < lenFilters; j++) {
            for (int k = 0; k < lenFilters; k++) {
                sum += d_InputImage[i][r + j][c + k] * FiltersCV2[f].elements[i][j][k];
            }
        }
    }
    //sum = (sum < 0) ? 0 : sum; //relu function
    d_OutImageCV2[f][r][c] = sum;
    //printf("\n %d,%d ", r,c);
}

__global__ void SharedFC1Kernel(float* d_ArrLayer1, float* d_ArrWeights, float* d_ArrLayer2, int lenLayer1, int  lenLayer2) {
    int idx = threadIdx.x;
    float val = 0;
    __shared__ float temp[32];
    for (int i = 0; i < 576; i++) {
        temp[idx] = d_ArrLayer1[i * 32 + idx];
        __syncthreads();
        for (int j = 0; j < 32; j++) {
            val += temp[j] * d_ArrWeights[idx * lenLayer1 + i * 32 + j];
        }
    }
    d_ArrLayer2[idx] = val;
}

__global__ void SharedFC1Kernel2(float* d_ArrLayer1, float* d_ArrWeights, float* d_ArrLayer2, int lenLayer1, int  lenLayer2) {
    int idx = threadIdx.x;
    float val = 0;
    for (int i = 0; i < lenLayer1; i++) {
        val += d_ArrLayer1[i] * d_ArrWeights[idx * lenLayer1 + i];
        /*temp[idx] = d_ArrLayer1[i * 32 + idx];
        __syncthreads();
        for (int j = 0; j < 32; j++) {
            val += temp[j] * d_ArrWeights[idx * lenLayer1 + i * 32 + j];
        }*/
    }
    d_ArrLayer2[idx] = val;
}
__global__ void SharedFC1Kernel3(float* d_ArrLayer1, float* d_ArrWeights, float* d_ArrLayer2, int lenLayer1, int  lenLayer2) {
    int idx = threadIdx.x;
    float val = 0;
    __shared__ float temp[512];
    for (int i = 0; i < 36; i++) {
        temp[idx] = d_ArrLayer1[i * 512 + idx];
        __syncthreads();
        for (int j = 0; j < 512; j++) {
            val += temp[j] * d_ArrWeights[idx * lenLayer1 + i * 512 + j];
        }
    }
    d_ArrLayer2[idx] = val;
}

__global__ void SharedFC1Kernel4(float* d_ArrLayer1, float* d_ArrWeights, float* d_ArrLayer2, int lenLayer1, int  lenLayer2) { // For 512 FC1 Layer
    int idx = threadIdx.x;
    float val = 0;
    for (int i = 0; i < lenLayer1; i++) {
        val += d_ArrLayer1[i] * d_ArrWeights[idx * lenLayer1 + i];
    }
    d_ArrLayer2[idx] = val;
}


__global__ void SharedFC2Kernel(float* d_ArrLayer1, float* d_ArrLayer2, int lenLayer1, int  lenLayer2) {
    int idx = threadIdx.x;
    float val = 0;
    __shared__ float temp[10];
    for (int i = 0; i < 3; i++) {
        temp[idx] = d_ArrLayer1[i * lenLayer2 + idx];
        __syncthreads();
        for (int j = 0; j < lenLayer2; j++) {
            val += temp[j] * d_ArrWeightsFC2[idx][i * 10 + j];
        }
    }
    val = val + d_ArrWeightsFC2[idx][30] * d_ArrLayer1[30] + d_ArrWeightsFC2[idx][31] * d_ArrLayer1[31];
    d_ArrLayer2[idx] = val;
}

__global__ void SharedFC2Kernel2(float* d_ArrLayer1, float* d_ArrLayer2, int lenLayer1, int  lenLayer2) {
    int idx = threadIdx.x;
    float val = 0;
    //__shared__ float temp[10];
    //float val = 0;
    for (int i = 0; i < lenLayer1; i++) {
        val += d_ArrLayer1[i] * d_ArrWeightsFC2[idx][i];
        /*temp[idx] = d_ArrLayer1[i * 32 + idx];
        __syncthreads();
        for (int j = 0; j < 32; j++) {
            val += temp[j] * d_ArrWeights[idx * lenLayer1 + i * 32 + j];
        }*/
    }
    d_ArrLayer2[idx] = val;
    /*for (int i = 0; i < 3; i++) {
        temp[idx] = d_ArrLayer1[i * lenLayer2 + idx];
        __syncthreads();
        for (int j = 0; j < lenLayer2; j++) {
            val += temp[j] * d_ArrWeightsFC2[idx][i * 10 + j];
        }
    }
    val = val + d_ArrWeightsFC2[idx][30] * d_ArrLayer1[30] + d_ArrWeightsFC2[idx][31] * d_ArrLayer1[31];
    d_ArrLayer2[idx] = val;*/
}
__global__ void SharedFC2Kernel3(float* d_ArrLayer1, float* d_ArrLayer2, int lenLayer1, int  lenLayer2) {
    int idx = threadIdx.x;
    float val = 0;
    __shared__ float temp[10];
    for (int i = 0; i < 51; i++) {
        temp[idx] = d_ArrLayer1[i * lenLayer2 + idx];
        __syncthreads();
        for (int j = 0; j < lenLayer2; j++) {
            val += temp[j] * d_ArrWeightsFC2[idx][i * 10 + j];
        }
    }
    val = val + d_ArrWeightsFC2[idx][510] * d_ArrLayer1[510] + d_ArrWeightsFC2[idx][511] * d_ArrLayer1[511];
    d_ArrLayer2[idx] = val;
}

__global__ void SharedFC2Kernel4(float* d_ArrLayer1, float* d_ArrLayer2, int lenLayer1, int  lenLayer2) {
    int idx = threadIdx.x;
    float val = 0;
    for (int i = 0; i < lenLayer1; i++) {
        val += d_ArrLayer1[i] * d_ArrWeightsFC2[idx][i];
    }
    d_ArrLayer2[idx] = val;
}



void generateMatrix(float* m, int size) { //Important for generating Matrix/Image
    for (int i = 0; i < size * size; i++) {
        m[i] = i % size;
    }
}


void generateFiltersCV1(Matrix* Filters) {
    for (int i = 0; i < NFiltersCV1; i++) {
        //printf("\nAllocation for Filter %f :\n", i);
        for (int j = 0; j < lenFilters; j++) {
            for (int k = 0; k < lenFilters; k++) {
                Filters[i].elements[j][k] = i * 0.01;
            }
        }
    }
}

void generateFiltersCV2(MatrixCV2* Filters) {
    for (int f = 0; f < NFiltersCV2; f++) {
        for (int i = 0; i < NFiltersCV1; i++) {
            for (int j = 0; j < lenFilters; j++) {
                for (int k = 0; k < lenFilters; k++) {
                    Filters[f].elements[i][j][k] = f * 0.01;
                }
            }
        }
    }
}


void generateWeightsFC1(float* weights, int lenLayer1, int lenLayer2) {
    for (int i = 0; i < lenLayer2; i++) {
        for (int j = 0; j < lenLayer1; j++) {
            weights[i * lenLayer1 + j] = i * 0.01;
            //printf("%d", i);
        }
        //printf("\n");
    }
}

void generateWeightsFC2(float* ArrWeightsFC2, int lenLayer2, int lenOutLayer) {
    for (int i = 0; i < lenOutLayer; i++) {
        for (int j = 0; j < lenLayer2; j++) {
            ArrWeightsFC2[i * lenLayer2 + j] = i * 0.099;
            //printf("%d", i);
        }
        //printf("\n");
    }
}



int main(int argc, char* argv[])
{
    size_t bytes_InputImage = lenInputImage * lenInputImage * sizeof(float);
    size_t bytes_FC1Weights = lenLayer1 * lenLayer2 * sizeof(float);
    size_t bytes_FiltersCV1 = NFiltersCV1 * lenFilters * lenFilters * sizeof(float);
    size_t bytes_OutImageCV1 = NFiltersCV1 * lenOutImageCV1 * lenOutImageCV1 * sizeof(float);
    size_t bytes_FiltersCV2 = NFiltersCV2 * NFiltersCV1 * lenFilters * lenFilters * sizeof(float);
    size_t bytes_OutImageCV2 = NFiltersCV2 * lenOutImageCV2 * lenOutImageCV2 * sizeof(float);

    size_t bytes_FC2Weights = lenLayer2 * lenOutLayer * sizeof(float);

    size_t bytes_ArrLayer1 = lenLayer1 * sizeof(float);
    size_t bytes_ArrLayer2 = lenLayer2 * sizeof(float);
    size_t bytes_ArrOutLayer = lenOutLayer * sizeof(float);



    /*----------------------Entire Initialization---------------------------------*/
    //Image Creation
    float* h_InputImage; // 28 x 28
    h_InputImage = (float*)malloc(bytes_InputImage);
    generateMatrix(h_InputImage, lenInputImage);
    //printMatrix(h_InputImage, lenInputImage);

    //  CV1 Filters

    Matrix* h_FiltersCV1; //16x3x3
    h_FiltersCV1 = (Matrix*)malloc(bytes_FiltersCV1);
    generateFiltersCV1(h_FiltersCV1);
    //printFilters(h_FiltersCV1);

    // CV2 Filters
    MatrixCV2* h_FiltersCV2; //32x16x3x3
    h_FiltersCV2 = (MatrixCV2*)malloc(bytes_FiltersCV2);
    generateFiltersCV2(h_FiltersCV2);

    // FC1 Weights

    float* h_ArrWeightsFC1; // 32 x 10816
    h_ArrWeightsFC1 = (float*)malloc(bytes_FC1Weights);
    generateWeightsFC1(h_ArrWeightsFC1, lenLayer1, lenLayer2);

    //FC2 Weights

    float* h_ArrWeightsFC2; // 10 x 32
    h_ArrWeightsFC2 = (float*)malloc(bytes_FC2Weights);
    generateWeightsFC2(h_ArrWeightsFC2, lenLayer2, lenOutLayer);
    float h_ArrWeightsFC2s[10][32];
    for (int i = 0; i < lenOutLayer; i++) {
        for (int j = 0; j < lenLayer2; j++) {
            h_ArrWeightsFC2s[i][j] = h_ArrWeightsFC2[i * 32 + j];
            //printf("%d", i);
        }
        //printf("\n");
    }

    /*----------------------------------------------------------------------------------------------------Entire CPU CNN-------------------------------------------------*/

    float InputImage[28][28];
    //copying pixels values from h_inputImage(GPU) to InputImage(CPU)
    for (int i = 0; i < 28; i++) {
        for (int j = 0; j < 28; j++) {
            InputImage[i][j] = h_InputImage[i * 28 + j];
        }
    }

    //copying filters of CV1 GPU to CPU

    float FiltersCV1[16][3][3];
    for (int f = 0; f < 16; f++) {
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                FiltersCV1[f][i][j] = h_FiltersCV1[f].elements[i][j];
            }
        }
    }
    clock_t tStart = clock();

    /*---------CV1 Convolution-----*/
    float OutMatrixCV1[16][26][26];
    float val;
    for (int f = 0; f < NFiltersCV1; f++) {
        for (int i = 0; i < 26; i++) {
            for (int j = 0; j < 26; j++) {
                val = 0;
                for (int p = 0; p < 3; p++) {
                    for (int q = 0; q < 3; q++) {

                        val = val + FiltersCV1[f][p][q] * InputImage[i + p][j + q];
                    }
                }
                OutMatrixCV1[f][i][j] = val;
            }
        }
    }
    /*printf("\n--------Convolution layer CPU outputs---------\n");
    for (int i = 0; i < 26; i++) {
        for (int j = 0; j < 26; j++) {
            cout << OutMatrixCV1[ex][i][j] << " ";
        }
        cout << endl;
    }

    /*---------CV2 Convolution-----*/
    //Copying Filters of CV2 from GPU to CPU
    float FiltersCV2[32][16][3][3];
    for (int f = 0; f < 32; f++) {
        for (int i = 0; i < 16; i++) {
            for (int j = 0; j < 3; j++) {
                for (int k = 0; k < 3; k++) {
                    FiltersCV2[f][i][j][k] = h_FiltersCV2[f].elements[i][j][k];
                }
            }
        }

    }
    //float inputCV2[16][26][26]; // is same as OutMatrixCV1[16][26][26]
    float OutMatrixCV2[32][24][24];
    for (int f = 0; f < NFiltersCV2; f++) {
        for (int r = 0; r < lenOutImageCV2; r++) {
            for (int c = 0; c < lenOutImageCV2; c++) {
                float sum = 0;
                for (int i = 0; i < NFiltersCV1; i++) {
                    for (int j = 0; j < lenFilters; j++) {
                        for (int k = 0; k < lenFilters; k++) {
                            sum += OutMatrixCV1[i][r + j][c + k] * FiltersCV2[f][i][j][k];
                        }
                    }
                }
                //sum = (sum < 0) ? 0 : sum; //relu function
                OutMatrixCV2[f][r][c] = sum;
            }
        }
    }

    /*printf("\n--------Convolution layer CPU outputs---------\n");

    for (int i = 0; i < 24; i++){
        for (int j = 0; j < 24; j++) {
            printf(" %f ", OutMatrixCV2[ex][i][j]);

        }
        printf("\n");
    }*/

    float* ArrLayer1;
    ArrLayer1 = (float*)malloc(bytes_ArrLayer1);

    int m = 0;
    // Flattening
    for (int i = 0; i < NFiltersCV2; i++) {
        for (int j = 0; j < lenOutImageCV2; j++) {
            for (int k = 0; k < lenOutImageCV2; k++) {
                ArrLayer1[m] = OutMatrixCV2[i][j][k];
                m = m + 1;
            }
        }
    }

    /*---------FC1---------*/

    float* ArrLayer2;
    ArrLayer2 = (float*)malloc(bytes_ArrLayer2);

    for (int i = 0; i < lenLayer2; i++) {
        float val = 0;
        for (int j = 0; j < lenLayer1; j++) {
            val += ArrLayer1[j] * h_ArrWeightsFC1[i * lenLayer1 + j];
        }
        ArrLayer2[i] = val;
    }

    /*printf("\n--------ArrLayer2 layer CPU outputs---------\n");
    for (int i = 0; i < lenLayer2; i++) {
        printf("\n%f", ArrLayer2[i]);
    }

    /*---------FC2---------*/

    float* ArrOutLayer;
    ArrOutLayer = (float*)malloc(bytes_ArrOutLayer);

    for (int i = 0; i < lenOutLayer; i++) {
        int val = 0;
        for (int j = 0; j < lenLayer2; j++) {
            val += ArrLayer2[j] * h_ArrWeightsFC2[i * 512 + j];
        }
        ArrOutLayer[i] = val;
    }
    printf("\nTime taken by CPU: %fs\n", (double)(clock() - tStart) / CLOCKS_PER_SEC);

    printf("\n--------final layer CPU outputs---------\n");
    for (int i = 0; i < lenOutLayer; i++) {
        printf("\n%f", ArrOutLayer[i]/10000);
    }



    /*-------------------------------------------------------------------------------------------------------------------Entire GPU CNN-------------------------------------------------*/
    //GPU Image Transfer
    float* d_InputImage;
    cudaMalloc(&d_InputImage, bytes_InputImage);
    cudaMemcpy(d_InputImage, h_InputImage, bytes_InputImage, cudaMemcpyHostToDevice);


    /*---------CV1 Convolution-----*/
    cudaMemcpyToSymbol(d_FiltersCV1, h_FiltersCV1, bytes_FiltersCV1);

    //GPU outputImage memory Allocation
    float h_OutImageCV1[16][26][26];
    //h_OutImageCV1 = (float*)malloc(bytes_OutImageCV1);
    float* d_OutImageCV1;
    cudaMalloc(&d_OutImageCV1, bytes_OutImageCV1);

    //CV1 Convolution Kernel
    int gridx = NFiltersCV1;
    int gridy = 1;
    int bckx = lenInputImage;
    int bcky = lenInputImage;
    dim3 dim_grid(gridx, gridy, 1);
    dim3 dim_block(bckx, bcky, 1);
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    cudaEvent_t startCV1, stopCV1;
    cudaEventCreate(&startCV1);
    cudaEventCreate(&stopCV1);
    cudaEventRecord(startCV1);

    CV1Kernel << <dim_grid, dim_block >> > ((float(*)[28])d_InputImage, NFiltersCV1, lenFilters, (float(*)[26][26])d_OutImageCV1);         // CV1 Kernel

    cudaEventRecord(stopCV1);
    cudaEventSynchronize(stopCV1);
    float millisecondsCV1 = 0;
    cudaEventElapsedTime(&millisecondsCV1, startCV1, stopCV1);
    printf("\ntime for execution on gpu CV1 is : %lf ms\n", millisecondsCV1);

    cudaMemcpy(h_OutImageCV1, d_OutImageCV1, bytes_OutImageCV1, cudaMemcpyDeviceToHost);
    /*printf("\n--------Convolution layer GPU outputs---------\n");
    for (int i = 0; i < 26; i++) {
        for (int j = 0; j < 26; j++) {
            printf(" %f ", h_OutImageCV1[ex][i][j]);

        }
        printf("\n");
    }

    /*---------CV2 ConvolutionLayer---------*/


    float h_OutImageCV2[32][24][24];
    float* d_OutImageCV2;
    cudaMalloc(&d_OutImageCV2, bytes_OutImageCV2);


    MatrixCV2* d_FiltersCV2;
    cudaMalloc(&d_FiltersCV2, bytes_FiltersCV2);
    cudaMemcpy(d_FiltersCV2, h_FiltersCV2, bytes_FiltersCV2, cudaMemcpyHostToDevice);
    cudaMemcpy(h_OutImageCV2, d_OutImageCV2, bytes_OutImageCV2, cudaMemcpyDeviceToHost);

    /*for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            printf(" %f ",h_FiltersCV2[ex].elements[10][i][j]);

        }
        printf("\n");
    }*/

    cudaMalloc(&d_OutImageCV2, bytes_OutImageCV2);
    cudaMemcpy(d_OutImageCV1, h_OutImageCV1, bytes_InputImage, cudaMemcpyHostToDevice);
    int gridx_CV2 = NFiltersCV2;
    int gridy_CV2 = 1;
    int bckx_CV2 = lenOutImageCV2;
    int bcky_CV2 = lenOutImageCV2;
    dim3 dim_grid_CV2(gridx_CV2, gridy_CV2, 1);
    dim3 dim_block_CV2(bckx_CV2, bcky_CV2, 1);


    cudaEvent_t startCV2, stopCV2;
    cudaEventCreate(&startCV2);
    cudaEventCreate(&stopCV2);
    cudaEventRecord(startCV2);

    CV2Kernel << <dim_grid_CV2, dim_block_CV2 >> > ((float(*)[26][26])d_OutImageCV1, d_FiltersCV2, (float(*)[24][24])d_OutImageCV2);
    cudaEventRecord(stopCV2);
    cudaEventSynchronize(stopCV2);
    float millisecondsCV2 = 0;
    cudaEventElapsedTime(&millisecondsCV2, startCV2, stopCV2);
    printf("\ntime for execution on gpu CV2 is : %lf ms\n", millisecondsCV1);
    cudaMemcpy(h_OutImageCV2, d_OutImageCV2, bytes_OutImageCV2, cudaMemcpyDeviceToHost);

    /*printf("\n--------Convolution layer GPU outputs---------\n");
    for (int i = 0; i < 24; i++) {
        for (int j = 0; j < 24; j++) {
            printf(" %f ", h_OutImageCV2[ex][i][j]);

        }
        printf("\n");
    }


    /*---------FC1-----------*/
    float* h_ArrLayer1; // 18432 x 1
    h_ArrLayer1 = (float*)malloc(bytes_ArrLayer1);
    m = 0;
    // Flattening
    for (int i = 0; i < NFiltersCV2; i++) {
        for (int j = 0; j < lenOutImageCV2; j++) {
            for (int k = 0; k < lenOutImageCV2; k++) {
                h_ArrLayer1[m] = h_OutImageCV2[i][j][k];
                m = m + 1;
            }
        }
    }

    float* d_ArrLayer1; // // 18432 x 1
    cudaMalloc(&d_ArrLayer1, bytes_ArrLayer1);
    cudaMemcpy(d_ArrLayer1, h_ArrLayer1, bytes_ArrLayer1, cudaMemcpyHostToDevice);


    float* d_ArrWeightsFC1;
    cudaMalloc(&d_ArrWeightsFC1, bytes_FC1Weights);
    cudaMemcpy(d_ArrWeightsFC1, h_ArrWeightsFC1, bytes_FC1Weights, cudaMemcpyHostToDevice);

    float* h_ArrLayer2;
    h_ArrLayer2 = (float*)malloc(bytes_ArrLayer2);

    float* d_ArrLayer2;
    cudaMalloc(&d_ArrLayer2, bytes_ArrLayer2);

    cudaEvent_t startFC1, stopFC1;
    cudaEventCreate(&startFC1);
    cudaEventCreate(&stopFC1);
    cudaEventRecord(startFC1);

    //SharedFC1Kernel << <1, lenLayer2 >> > (d_ArrLayer1, d_ArrWeightsFC1, d_ArrLayer2, lenLayer1, lenLayer2);            //FC1 kernel
    //SharedFC1Kernel2 << <1, lenLayer2 >> > (d_ArrLayer1, d_ArrWeightsFC1, d_ArrLayer2, lenLayer1, lenLayer2);            //FC1 kernel
    //SharedFC1Kernel3 << <1, lenLayer2 >> > (d_ArrLayer1, d_ArrWeightsFC1, d_ArrLayer2, lenLayer1, lenLayer2);
    SharedFC1Kernel4 << <1, lenLayer2 >> > (d_ArrLayer1, d_ArrWeightsFC1, d_ArrLayer2, lenLayer1, lenLayer2);

    cudaEventRecord(stopFC1);
    cudaEventSynchronize(stopFC1);
    float millisecondsFC1 = 0;
    cudaEventElapsedTime(&millisecondsFC1, startFC1, stopFC1);
    printf("\ntime for execution on gpu FC1 is : %lf ms\n", millisecondsFC1);
    cudaMemcpy(h_ArrLayer2, d_ArrLayer2, bytes_ArrLayer2, cudaMemcpyDeviceToHost);

    /*printf("\n--------ArrLayer2 layer GPU outputs---------\n");
    for (int i = 0; i < lenLayer2; i++) {
        printf("\n%f", h_ArrLayer2[i]);
    }

    /*---------FC2-----------*/
    /*printf("\n-----Nonsence------\n");
    for (int i = 0; i < 10*32; i++) {
        cout << h_ArrWeightsFC2[i] << endl;
    }*/
    cudaMemcpy(d_ArrLayer2, h_ArrLayer2, bytes_ArrLayer2, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(d_ArrWeightsFC2, h_ArrWeightsFC2s, bytes_FC2Weights);
    //float* d_ArrWeightsFC2;
    //cudaMalloc(&d_ArrWeightsFC2, bytes_FC2Weights);
    //cudaMemcpy(d_ArrWeightsFC2, h_ArrWeightsFC2, bytes_FC2Weights, cudaMemcpyHostToDevice);

    float* h_ArrOutLayer;
    h_ArrOutLayer = (float*)malloc(bytes_ArrOutLayer);

    float* d_ArrOutLayer;
    cudaMalloc(&d_ArrOutLayer, bytes_ArrOutLayer);

    cudaEvent_t startFC2, stopFC2;
    cudaEventCreate(&startFC2);
    cudaEventCreate(&stopFC2);
    cudaEventRecord(startFC2);

    //SharedFC2Kernel << <1, lenOutLayer >> > (d_ArrLayer2, d_ArrOutLayer, lenLayer2, lenOutLayer);                        //FC2 kernel
    //SharedFC2Kernel2 << <1, lenOutLayer >> > (d_ArrLayer2, d_ArrOutLayer, lenLayer2, lenOutLayer);                        //FC2 kernel
    //SharedFC2Kernel3 << <1, lenOutLayer >> > (d_ArrLayer2, d_ArrOutLayer, lenLayer2, lenOutLayer);
    SharedFC2Kernel4 << <1, lenOutLayer >> > (d_ArrLayer2, d_ArrOutLayer, lenLayer2, lenOutLayer);

    cudaEventRecord(stopFC2);
    cudaEventSynchronize(stopFC2);
    float millisecondsFC2 = 0;
    cudaEventElapsedTime(&millisecondsFC2, startFC2, stopFC2);
    printf("\ntime for execution on gpu FC2 is : %lf ms\n", millisecondsFC2);
    cudaMemcpy(h_ArrOutLayer, d_ArrOutLayer, bytes_ArrOutLayer, cudaMemcpyDeviceToHost);



    printf("\n--------final layer GPU outputs---------\n");
    for (int i = 0; i < lenOutLayer; i++) {
        printf("\n%f", h_ArrOutLayer[i]/10000);
    }

    return 0;
}
