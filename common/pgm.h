#ifndef PGM_H
#define PGM_H

#include <vector>
#include <utility>

class PGMImage {
public:
    // Constructores
    PGMImage(const char *filename);           // Constructor para const char*
    PGMImage(char *fname, int programID);     // Constructor adicional con programID
    PGMImage(int x = 100, int y = 100, int col = 16); // Constructor con valores predeterminados
    ~PGMImage();

    // Otros métodos
    bool write(const char *outputFileName, std::vector<std::pair<int, int>> selectedLines, float radIncrement, int rBins);
    int getXDim();
    int getYDim();
    int getNumColors();
    unsigned char* getPixels();

    // Variables miembro
    int x_dim;
    int y_dim;
    int num_colors;
    unsigned char *pixels;
    std::vector<unsigned char> color;
};

#endif // PGM_H
