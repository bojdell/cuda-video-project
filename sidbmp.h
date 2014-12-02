#ifndef SIDBMP_H
#define SIDBMP_H

typedef struct {
  int r;
  int g;
  int b;
} Pixel;

typedef struct {
  Pixel ** pixels;
  int height;
  int width;
} Image;

void print_image(Image * im);
Image * generate_image(int height, int width);
void delete_image(Image * im);



#endif
