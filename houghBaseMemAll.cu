#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>
#include <string.h>
#include "common/pgm.h"
#include <opencv2/opencv.hpp> // Usar OpenCV para manipular imágenes

const int degreeInc = 2;
const int degreeBins = 180 / degreeInc;
const int rBins = 100;
const float radInc = degreeInc * M_PI / 180;

// Declarar memoria Constante
__constant__ float d_Cos[degreeBins];
__constant__ float d_Sin[degreeBins];

// Función para dibujar líneas en la imagen
void drawLines(cv::Mat &image, int *hough, int width, int height, float rMax, float rScale, int threshold) {
    for (int tIdx = 0; tIdx < degreeBins; tIdx++) {
        for (int rIdx = 0; rIdx < rBins; rIdx++) {
            if (hough[rIdx * degreeBins + tIdx] > threshold) {
                float r = rIdx * rScale - rMax;
                float cosTheta = d_Cos[tIdx];
                float sinTheta = d_Sin[tIdx];

                cv::Point pt1, pt2;
                if (fabs(sinTheta) > 1e-6) {
                    pt1 = cv::Point(0, (int)((r - 0 * cosTheta) / sinTheta));
                    pt2 = cv::Point(width, (int)((r - width * cosTheta) / sinTheta));
                } else {
                    pt1 = cv::Point((int)(r / cosTheta), 0);
                    pt2 = cv::Point((int)(r / cosTheta), height);
                }

                if (pt1.y >= 0 && pt1.y < height && pt2.y >= 0 && pt2.y < height) {
                    cv::line(image, pt1, pt2, cv::Scalar(0, 0, 255), 1);
                }
            }
        }
    }
}

// Kernel combinado (memoria global, compartida y constante)
__global__ void GPU_HoughTranCombined(unsigned char *pic, int w, int h, int *acc, float rMax, float rScale) {
    int gloID = blockIdx.x * blockDim.x + threadIdx.x;
    if (gloID >= w * h) return;

    int xCent = w / 2;
    int yCent = h / 2;
    int xCoord = gloID % w - xCent;
    int yCoord = yCent - gloID / w;

    // Declarar memoria compartida para el acumulador temporal
    extern __shared__ int shared_acc[];

    // Inicializar memoria compartida a cero
    for (int i = threadIdx.x; i < degreeBins * rBins; i += blockDim.x) {
        shared_acc[i] = 0;
    }
    __syncthreads();

    if (pic[gloID] > 0) {
        for (int tIdx = 0; tIdx < degreeBins; tIdx++) {
            float r = xCoord * d_Cos[tIdx] + yCoord * d_Sin[tIdx];
            int rIdx = (r + rMax) / rScale;
            atomicAdd(&shared_acc[rIdx * degreeBins + tIdx], 1);
        }
    }
    __syncthreads();

    // Escribir los resultados acumulados en memoria global
    for (int i = threadIdx.x; i < degreeBins * rBins; i += blockDim.x) {
        atomicAdd(&acc[i], shared_acc[i]);
    }
}

int main(int argc, char **argv) {
    PGMImage inImg(argv[1]);

    int w = inImg.x_dim;
    int h = inImg.y_dim;

    // Precalcular valores de coseno y seno
    float *pcCos = (float *)malloc(sizeof(float) * degreeBins);
    float *pcSin = (float *)malloc(sizeof(float) * degreeBins);
    float rad = 0;
    for (int i = 0; i < degreeBins; i++) {
        pcCos[i] = cos(rad);
        pcSin[i] = sin(rad);
        rad += radInc;
    }

    float rMax = sqrt(1.0 * w * w + 1.0 * h * h) / 2;
    float rScale = 2 * rMax / rBins;

    // Copiar valores precalculados a la memoria Constante
    cudaMemcpyToSymbol(d_Cos, pcCos, sizeof(float) * degreeBins);
    cudaMemcpyToSymbol(d_Sin, pcSin, sizeof(float) * degreeBins);

    unsigned char *d_in;
    int *d_hough, *h_hough;

    h_hough = (int *)malloc(degreeBins * rBins * sizeof(int));
    cudaMalloc((void **)&d_in, sizeof(unsigned char) * w * h);
    cudaMalloc((void **)&d_hough, sizeof(int) * degreeBins * rBins);
    cudaMemcpy(d_in, inImg.pixels, sizeof(unsigned char) * w * h, cudaMemcpyHostToDevice);
    cudaMemset(d_hough, 0, sizeof(int) * degreeBins * rBins);

    int blockNum = ceil(w * h / 256);
    int sharedMemSize = sizeof(int) * degreeBins * rBins;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    GPU_HoughTranCombined<<<blockNum, 256, sharedMemSize>>>(d_in, w, h, d_hough, rMax, rScale);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Tiempo transcurrido para el kernel: %f ms\n", milliseconds);

    cudaMemcpy(h_hough, d_hough, sizeof(int) * degreeBins * rBins, cudaMemcpyDeviceToHost);

    cv::Mat outputImage(h, w, CV_8UC3);
    for (int i = 0; i < h; i++) {
        for (int j = 0; j < w; j++) {
            unsigned char pixel = inImg.pixels[i * w + j];
            outputImage.at<cv::Vec3b>(i, j) = cv::Vec3b(pixel, pixel, pixel);
        }
    }

    float mean = 0, stddev = 0;
    for (int i = 0; i < degreeBins * rBins; i++) mean += h_hough[i];
    mean /= (degreeBins * rBins);
    for (int i = 0; i < degreeBins * rBins; i++)
        stddev += pow(h_hough[i] - mean, 2);
    stddev = sqrt(stddev / (degreeBins * rBins));
    int threshold = mean + 2 * stddev;

    drawLines(outputImage, h_hough, w, h, rMax, rScale, threshold);

    cv::imwrite("output_combined.png", outputImage);

    free(pcCos);
    free(pcSin);
    free(h_hough);

    cudaFree(d_in);
    cudaFree(d_hough);

    return 0;
}
