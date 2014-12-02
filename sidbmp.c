#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "sidbmp.h"

void print_image(Image * im) {
  int i = 0,
      j = 0;

  for(i = 0; i < im->height; i++) {
    for(j = 0; j < im->width; j++) {
      int p = i * im->height + j;
      printf("(%d, %d) -> (%d) \t R: %d, G: %d, B: %d\n",j, i, p, im->pixels[p]->r, im->pixels[p]->g, im->pixels[p]->b);
    }
  }
}

Image * generate_image(int height, int width) {
  int i;
  Image * im = (Image *) malloc(sizeof(Image));
  im->height = height;
  im->width = width;

  im->pixels = (Pixel **) malloc(height * width * sizeof(Pixel *));
  for(i = 0; i < height * width; i++) {
    im->pixels[i] = (Pixel *) malloc(sizeof(Pixel));
    im->pixels[i]->r = rand() % 255;
    im->pixels[i]->g = rand() % 255;
    im->pixels[i]->b = rand() % 255;
  }

  return im;
}

void delete_image(Image * im) {
  int i;
  for(i = 0; i < im->height * im->width; i++) {
    free(im->pixels[i]);
  }

  free(im->pixels);
  free(im);
}