#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "sidbmp.h"

void grayscale(Image * im) {
	int i = 0;
	for(i = 0; i < im->height * im->width; i++) {
		int gs = (im->pixels[i]->r + im->pixels[i]->g + im->pixels[i]->b) / 3;
		im->pixels[i]->r = gs;
		im->pixels[i]->g = gs;
		im->pixels[i]->b = gs;
	}
}

void grayscale_flat(Image *im) {
	int i = 0, j = 0;
	int * flatpixs = (int *) malloc(im->height * im->width * 3);

	for(i = 0; i < im->height * im->width; i++) {
		flatpixs[j++] = im->pixels[i]->r;
		flatpixs[j++] = im->pixels[i]->g;
		flatpixs[j++] = im->pixels[i]->b;
	}

	int height = im->height;
	int width = im->width;
	// this is where we would invoke kernel with flatpix, height, and width
		for(i = 0; i < height * width; i++) {
			int start = 3 * i;
			int gs = (flatpixs[start] + flatpixs[start + 1] + flatpixs[start + 2]) / 3;
			flatpixs[start] = gs;
		}
	// end kernel, every third val in flatpix contains the grayscale value

	// this would be where we rebuild the image
	for(i = 0; i < im->height * im->width; i++) {
		im->pixels[i]->r = flatpixs[i];
		im->pixels[i]->g = flatpixs[i];
		im->pixels[i]->b = flatpixs[i];
	}
}

int main(int argc, char **argv)
{

  srand(time(NULL));
  Image * im = generate_image(3, 3);
  print_image(im);
  printf("\n");
  grayscale_flat(im);
  print_image(im);
  delete_image(im);


  return 1;
}