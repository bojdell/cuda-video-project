#ifndef PPMH
#define PPMH

#define OUTPUT_TILE_SIZE 12
#define INPUT_TILE_SIZE 22
#define FRAME_DEPTH 5

#define FILTER_SIZE 5

typedef struct {
     unsigned char red,green,blue;
} PPMPixel;

typedef struct {
     int x, y;
     PPMPixel *data;
} PPMImage;

typedef struct {
     int x, y, z;
     double data[FILTER_SIZE][FILTER_SIZE][FILTER_SIZE];
     double factor;
     double bias;
} Filter3D;


#endif
