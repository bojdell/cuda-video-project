#ifndef PPMH
#define PPMH

<<<<<<< HEAD
=======
#define OUTPUT_TILE_SIZE 12
#define INPUT_TILE_SIZE 22
#define FRAME_DEPTH 5

>>>>>>> 7554dbf0d68acad0b2df70ab9f33aa7b5e1a11a6
#define FILTER_SIZE 5
#define CREATOR "DA BROS"
#define RGB_COMPONENT_COLOR 255

#define OUTPUT_TILE_SIZE 12

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

typedef struct {
     int x, y, z;
} cdim3;

#endif
