#include "ppm.h"
__global__ void blackAndWhite(PPMPixel *imageData, PPMPixel *outputData, int width, int height) {

	int tx = threadIdx.x;
	int ty = blockIdx.x;


	if(ty < height && tx < width) {
		int i = ty*width + tx;
		int avg = (imageData[i].red + imageData[i].green + imageData[i].blue) / 3;

		outputData[i].red = avg;
		outputData[i].green = avg;
		outputData[i].blue = avg;
	}



}

