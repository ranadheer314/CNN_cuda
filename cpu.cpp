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


const int lenLayer1 = 18432;
const int lenLayer2 = 512;
const int lenOutLayer = 10;
int ex = 31;

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

   int main(){
 /*---------FC1---------*/
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