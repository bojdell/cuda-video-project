#ifndef PPMH
#define PPMH


#define FILTER_SIZE 5
#define OUTPUT_TILE_SIZE 12
#define INPUT_TILE_SIZE (OUTPUT_TILE_SIZE + FILTER_SIZE - 1)

typedef struct {
     unsigned char red,green,blue;
} PPMPixel;

typedef struct {
     int x, y;
     PPMPixel *data;
} PPMImage;

typedef struct {
     int x, y;
     double data[FILTER_SIZE * FILTER_SIZE];
     double factor;
     double bias;
} Filter;


#endif
