#ifndef PPMH
#define PPMH


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
