#ifndef PPMH
#define PPMH

#define FILTER_SIZE (5)
#define OUTPUT_TILE_X (156)
#define OUTPUT_TILE_Y (156)
#define OUTPUT_TILE_Z (5)
#define INPUT_TILE_X  (OUTPUT_TILE_X + FILTER_SIZE - 1)
#define INPUT_TILE_Y  (OUTPUT_TILE_Y + FILTER_SIZE - 1)
#define INPUT_TILE_Z  (OUTPUT_TILE_Z + FILTER_SIZE - 1)
#define BLOCK_SIZE (8)

#define CREATOR ("DA BROS")
#define RGB_COMPONENT_COLOR (255)

// Pixel structure containing red green and blue values
typedef struct {
     unsigned char red,green,blue;
} PPMPixel;

// PPM image structure containing x, y dimensions and pixel data
typedef struct {
     int x, y;
     PPMPixel *data;
} PPMImage;

// 3D filter structure containing dimensions, factor, bias, and data values
typedef struct {
     int x, y, z;
     double data[FILTER_SIZE][FILTER_SIZE][FILTER_SIZE];
     double factor;
     double bias;
} Filter3D;

// An dim3 structure containing x y z dimensions
typedef struct {
     int x, y, z;
} cdim3;

#endif
