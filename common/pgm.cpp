/*/============================================================================
 Author        : G. Barlas
 Version       : 1.0
 Last modified : December 2014
 License       : Released under the GNU GPL 3.0
 Description   : Used in different projects to handle PGM I/O
 To build use  : 
 ============================================================================
 */
#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <jpeglib.h>
#include <math.h>
#include <vector>
#include "pgm.h"

using namespace std;

//-------------------------------------------------------------------
PGMImage::PGMImage(char *fname, int programID)
{
   if(programID == 0) {
        this->color = {255,0,0};
   } else if(programID == 1) {
        this->color = {0,255,0};
   } else {
        this->color = {0,0,255};
   }
   this->x_dim=this->y_dim=this->num_colors=0;
   this->pixels=NULL;
   
   FILE *ifile;
   ifile = fopen(fname, "rb");
   if(!ifile) return;

   char *buff = NULL;
   size_t temp;

   fscanf(ifile, "%*s %i %i %i", &this->x_dim, &this->y_dim, &this->num_colors);

   getline((char **)&buff, &temp, ifile); // eliminate CR-LF
   
   assert(this->x_dim >1 && this->y_dim >1 && this->num_colors >1);
   this->pixels = new unsigned char[this->x_dim * this->y_dim];
   fread((void *) this->pixels, 1, this->x_dim*this->y_dim, ifile);   
   
   fclose(ifile);
}

// Constructor que acepta const char* y redirige al constructor existente
PGMImage::PGMImage(const char *filename) : PGMImage(const_cast<char *>(filename), 0) {
    // Este constructor redirige al existente que acepta char * y programID
}


//-------------------------------------------------------------------
// Definición del constructor en pgm.cpp sin valores predeterminados
PGMImage::PGMImage(int x, int y, int col) {
    this->num_colors = (col > 1) ? col : 16;
    this->x_dim = (x > 1) ? x : 100;
    this->y_dim = (y > 1) ? y : 100;
    this->pixels = new unsigned char[x_dim * y_dim];
    memset(this->pixels, 0, this->x_dim * this->y_dim);
    assert(this->pixels);
}
//-------------------------------------------------------------------
PGMImage::~PGMImage()
{
  if(this->pixels != NULL)
     delete [] this->pixels;
  this->pixels = NULL;
}

// Helper function to color a pixel in the given buffer
void colorPixel(unsigned char *colorPixels, int i, unsigned char r, unsigned char g, unsigned char b) {
    colorPixels[i*3] = r;
    colorPixels[i*3+1] = g;
    colorPixels[i*3+2] = b;
}

// Cambia el tipo de retorno de write a bool para que coincida con la declaración en pgm.h
bool PGMImage::write(const char* outputFileName, std::vector<std::pair<int, int>> selectedLines, float radIncrement, int rBins) {
    float maxRadius = hypot(this->x_dim, this->y_dim) / 2;
    float rScale = 2 * maxRadius / rBins;
    int xCenter = this->x_dim / 2;
    int yCenter = this->y_dim / 2;

    // Allocate memory for RGB image
    unsigned char *colorPixels = new unsigned char[this->x_dim * this->y_dim * 3];

    // Store pixel coordinates
    std::vector<std::pair<int, int>> coordinates(this->x_dim * this->y_dim);
    for (int i = 0; i < x_dim * y_dim; ++i) {
        coordinates[i] = std::make_pair(i % this->x_dim - xCenter, yCenter - i / this->x_dim);
    }

    for (int i = 0; i < this->x_dim * this->y_dim; ++i) {
        bool isLine = false;
        auto [x, y] = coordinates[i];

        for (const auto& line: selectedLines) {
            int rIdx = line.first;
            int thetaIdx = line.second;
            float r = rIdx * rScale - maxRadius;
            float theta = thetaIdx * radIncrement;

            if (std::abs(r - x * std::cos(theta) - y * std::sin(theta)) < 0.5) {
                isLine = true;
                break;
            }
        }
        // Verify if pixel needs to be coloured
        if (isLine) {
            colorPixel(colorPixels, i, this->color.at(0), this->color.at(1), this->color.at(2));  // Red color for line pixels
        } else {
            colorPixel(colorPixels, i, pixels[i], pixels[i], pixels[i]);  // Grayscale for non-line pixels
        }
    }

    // Configuración para compresión JPEG
    struct jpeg_compress_struct compressInfo;
    struct jpeg_error_mgr errorManager;
    
    // Open the output file
    FILE* outputFile = fopen(outputFileName, "wb");
    if (!outputFile) {
        printf("Error al abrir el archivo.\n");
        delete[] colorPixels;
        return false;
    }

    // Initialize the JPEG compression structures
    compressInfo.err = jpeg_std_error(&errorManager);
    jpeg_create_compress(&compressInfo);
    jpeg_stdio_dest(&compressInfo, outputFile);

    // Set the image properties
    compressInfo.image_width = this->x_dim;
    compressInfo.image_height = this->y_dim;
    compressInfo.input_components = 3; // RGB image
    compressInfo.in_color_space = JCS_RGB; // RGB color space

    // Set the JPEG compression parameters
    jpeg_set_defaults(&compressInfo);
    jpeg_set_quality(&compressInfo, 75, TRUE);

    // Start compression and write the image data
    jpeg_start_compress(&compressInfo, TRUE);
    JSAMPROW rowPointer[1];
    int rowStride = x_dim * 3;

    // Write each row
    while (compressInfo.next_scanline < compressInfo.image_height) {
        rowPointer[0] = &colorPixels[compressInfo.next_scanline * rowStride];
        jpeg_write_scanlines(&compressInfo, rowPointer, 1);
    }

    // Clean up after compression
    jpeg_finish_compress(&compressInfo);
    fclose(outputFile);
    jpeg_destroy_compress(&compressInfo);

    // Deallocate memory
    delete[] colorPixels;
    return true;
}

int PGMImage::getXDim(void) {
    return this->x_dim;
}

int PGMImage::getYDim(void) {
    return this->y_dim;
}

int PGMImage::getNumColors(void) {
    return this->num_colors;
}

unsigned char* PGMImage::getPixels(void) {
    return this->pixels;
}
