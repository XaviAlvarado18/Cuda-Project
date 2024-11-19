/*
 ============================================================================
 Author        : G. Barlas
 Version       : 1.0
 Last modified : November 2023
 License       : Released under the GNU GPL 3.0
 Description   :
 To build use  : make
 ============================================================================
 */
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

// Variables de memoria Global
float *d_Cos, *d_Sin;

// Dibujar líneas en la imagen de salida
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

                // Dibujar líneas basadas en r y theta
                cv::Point pt1, pt2;
                if (sinTheta != 0) {
                    pt1 = cv::Point(0, (int)((r - 0 * cosTheta) / sinTheta));
                    pt2 = cv::Point(width, (int)((r - width * cosTheta) / sinTheta));
                } else {
                    pt1 = cv::Point((int)(r / cosTheta), 0);
                    pt2 = cv::Point((int)(r / cosTheta), height);
                }

                cv::line(image, pt1, pt2, cv::Scalar(0, 0, 255), 1); // Líneas rojas
            }
        }
    }
}


//*****************************************************************
// The CPU function returns a pointer to the accummulator
void CPU_HoughTran (unsigned char *pic, int w, int h, int **acc)
{
  float rMax = sqrt (1.0 * w * w + 1.0 * h * h) / 2;  //(w^2 + h^2)/2, radio max equivalente a centro -> esquina
  *acc = new int[rBins * degreeBins];            //el acumulador, conteo depixeles encontrados, 90*180/degInc = 9000
  memset (*acc, 0, sizeof (int) * rBins * degreeBins); //init en ceros
  int xCent = w / 2;
  int yCent = h / 2;
  float rScale = 2 * rMax / rBins;

  for (int i = 0; i < w; i++) //por cada pixel
    for (int j = 0; j < h; j++) //...
      {
        int idx = j * w + i;
        if (pic[idx] > 0) //si pasa thresh, entonces lo marca
          {
            int xCoord = i - xCent;
            int yCoord = yCent - j;  // y-coord has to be reversed
            float theta = 0;         // actual angle
            for (int tIdx = 0; tIdx < degreeBins; tIdx++) //add 1 to all lines in that pixel
              {
                float r = xCoord * cos (theta) + yCoord * sin (theta);
                int rIdx = (r + rMax) / rScale;
                (*acc)[rIdx * degreeBins + tIdx]++; //+1 para este radio r y este theta
                theta += radInc;
              }
          }
      }
}

// GPU kernel
__global__ void GPU_HoughTran(unsigned char *pic, int w, int h, int *acc, float rMax, float rScale, float *d_Cos, float *d_Sin) {
    int gloID = blockIdx.x * blockDim.x + threadIdx.x;
    if (gloID >= w * h) return;

    int xCent = w / 2;
    int yCent = h / 2;
    int xCoord = gloID % w - xCent;
    int yCoord = yCent - gloID / w;

    if (pic[gloID] > 0) {
        for (int tIdx = 0; tIdx < degreeBins; tIdx++) {
            float r = xCoord * d_Cos[tIdx] + yCoord * d_Sin[tIdx];
            int rIdx = (r + rMax) / rScale;
            atomicAdd(acc + (rIdx * degreeBins + tIdx), 1);
        }
    }
}

int main(int argc, char **argv) {
    // Leer la imagen de entrada
    PGMImage inImg(argv[1]);

    int *cpuht;
    int w = inImg.x_dim;
    int h = inImg.y_dim;

    // CPU: cálculo inicial
    CPU_HoughTran(inImg.pixels, w, h, &cpuht);

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

    // Asignar memoria Global para d_Cos y d_Sin
    cudaMalloc((void **)&d_Cos, sizeof(float) * degreeBins);
    cudaMalloc((void **)&d_Sin, sizeof(float) * degreeBins);

    // Copiar valores precalculados a memoria Global
    cudaMemcpy(d_Cos, pcCos, sizeof(float) * degreeBins, cudaMemcpyHostToDevice);
    cudaMemcpy(d_Sin, pcSin, sizeof(float) * degreeBins, cudaMemcpyHostToDevice);

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

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    GPU_HoughTran<<<blockNum, 256>>>(d_in, w, h, d_hough, rMax, rScale, d_Cos, d_Sin);
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
    cv::imwrite("output_with_lines.png", outputImage);

    // Liberar memoria en host y dispositivo
    free(cpuht);
    free(pcCos);
    free(pcSin);
    free(h_hough);

    cudaFree(d_Cos);
    cudaFree(d_Sin);
    cudaFree(d_in);
    cudaFree(d_hough);

    return 0;
}
