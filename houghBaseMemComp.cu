
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>
#include <string.h>
#include "pgm.h"
#include <opencv2/opencv.hpp> // Usar OpenCV para manipular imágenes

const int degreeInc = 2;
const int degreeBins = 180 / degreeInc;
const int rBins = 100;
const float radInc = degreeInc * M_PI / 180;

// Declarar memoria Constante
__constant__ float d_Cos[degreeBins];
__constant__ float d_Sin[degreeBins];

// Función para dibujar líneas en la imagen resultante
void drawLines(
    cv::Mat &image, int *hough, int width, int height,
    float rMax, float rScale, int threshold) {
    float rad = 0;

    for (int tIdx = 0; tIdx < degreeBins; tIdx++) {
        rad = tIdx * radInc;
        float cosTheta = cos(rad);
        float sinTheta = sin(rad);

        for (int rIdx = 0; rIdx < rBins; rIdx++) {
            if (hough[rIdx * degreeBins + tIdx] > threshold) {
                float r = rIdx * rScale - rMax;

                // Calcular puntos de intersección con los bordes de la imagen
                cv::Point pt1, pt2;

                if (fabs(sinTheta) > 1e-6) {
                    pt1 = cv::Point(0, (int)((r - 0 * cosTheta) / sinTheta)); // Intersección con el borde izquierdo
                    pt2 = cv::Point(width, (int)((r - width * cosTheta) / sinTheta)); // Intersección con el borde derecho
                } else {
                    pt1 = cv::Point((int)(r / cosTheta), 0); // Intersección con el borde superior
                    pt2 = cv::Point((int)(r / cosTheta), height); // Intersección con el borde inferior
                }

                // Verificar que los puntos estén dentro de la imagen
                if (pt1.y >= 0 && pt1.y < height && pt2.y >= 0 && pt2.y < height) {
                    cv::line(image, pt1, pt2, cv::Scalar(0, 0, 255), 1); // Dibujar la línea en rojo
                }
            }
        }
    }
}

// Kernel GPU utilizando memoria compartida
__global__ void GPU_HoughTranShared(unsigned char *pic, int w, int h, int *acc, float rMax, float rScale) {
    int gloID = blockIdx.x * blockDim.x + threadIdx.x;
    int localID = threadIdx.x; // ID local del hilo dentro del bloque

    if (gloID >= w * h) return;

    int xCent = w / 2;
    int yCent = h / 2;
    int xCoord = gloID % w - xCent;
    int yCoord = yCent - gloID / w;

    // Declarar memoria compartida para los valores de seno y coseno
    __shared__ float shared_Cos[degreeBins];
    __shared__ float shared_Sin[degreeBins];

    // Cargar los valores de seno y coseno en la memoria compartida
    if (localID < degreeBins) {
        shared_Cos[localID] = cos(localID * radInc);
        shared_Sin[localID] = sin(localID * radInc);
    }

    // Sincronizar los hilos dentro del bloque para asegurarnos de que todos los valores estén cargados
    __syncthreads();

    // Asegurarse de que la memoria compartida está correctamente cargada antes de usarse
    if (pic[gloID] > 0) {
        for (int tIdx = 0; tIdx < degreeBins; tIdx++) {
            float r = xCoord * shared_Cos[tIdx] + yCoord * shared_Sin[tIdx];
            int rIdx = (r + rMax) / rScale;

            // Usar memoria compartida para el acumulador temporal
            extern __shared__ int shared_acc[];

            // Usar el acumulador compartido para evitar accesos globales
            atomicAdd(&shared_acc[rIdx * degreeBins + tIdx], 1);

            // Sincronizar los hilos del bloque antes de escribir los resultados en la memoria global
            __syncthreads();

            // Escribir los resultados acumulados en memoria global después de todos los hilos
            if (localID == 0) {
                for (int i = 0; i < degreeBins * rBins; i++) {
                    atomicAdd(acc + i, shared_acc[i]);
                }
            }
        }
    }
}

int main(int argc, char **argv) {
    // Leer la imagen de entrada
    PGMImage inImg(argv[1]);

    int *cpuht;
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

    // Configuración de memoria
    unsigned char *d_in, *h_in;
    int *d_hough, *h_hough;

    h_in = inImg.pixels;
    h_hough = (int *)malloc(degreeBins * rBins * sizeof(int));

    cudaMalloc((void **)&d_in, sizeof(unsigned char) * w * h);
    cudaMalloc((void **)&d_hough, sizeof(int) * degreeBins * rBins);
    cudaMemcpy(d_in, h_in, sizeof(unsigned char) * w * h, cudaMemcpyHostToDevice);
    cudaMemset(d_hough, 0, sizeof(int) * degreeBins * rBins);

    // Configuración y ejecución del kernel
    int blockNum = ceil(w * h / 256);

    // El tamaño de la memoria compartida se pasa al kernel
    int sharedMemSize = sizeof(int) * degreeBins * rBins;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    GPU_HoughTranShared<<<blockNum, 256, sharedMemSize>>>(d_in, w, h, d_hough, rMax, rScale);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // Calcular tiempo transcurrido
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Tiempo transcurrido para el kernel: %f ms\n", milliseconds);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // Copiar resultados de vuelta al host
    cudaMemcpy(h_hough, d_hough, sizeof(int) * degreeBins * rBins, cudaMemcpyDeviceToHost);

    // Crear una imagen de salida con OpenCV
    cv::Mat outputImage(h, w, CV_8UC3);
    for (int i = 0; i < h; i++) {
        for (int j = 0; j < w; j++) {
            unsigned char pixel = inImg.pixels[i * w + j];
            outputImage.at<cv::Vec3b>(i, j) = cv::Vec3b(pixel, pixel, pixel);
        }
    }

    // Criterio de selección: promedio + 2 desviaciones estándar
    float mean = 0, stddev = 0;
    for (int i = 0; i < degreeBins * rBins; i++) mean += h_hough[i];
    mean /= (degreeBins * rBins);
    for (int i = 0; i < degreeBins * rBins; i++)
        stddev += pow(h_hough[i] - mean, 2);
    stddev = sqrt(stddev / (degreeBins * rBins));
    int threshold = mean + 2 * stddev;

    // Dibujar líneas detectadas
    drawLines(outputImage, h_hough, w, h, rMax, rScale, threshold);

    // Guardar la imagen resultante
    cv::imwrite("output_with_lines_shared.png", outputImage);

    // Liberar memoria en host y dispositivo
    free(cpuht);
    free(pcCos);
    free(pcSin);
    free(h_hough);

    cudaFree(d_in);
    cudaFree(d_hough);

    return 0;
}
