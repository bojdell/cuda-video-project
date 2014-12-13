#include "ppm.h"
#include <math.h>

// Constant memory for convolution filter
__constant__ Filter filter_c;

// the black and white kernel, each thread changes a pixel
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

// the convolution kernel, each thread convolves for a pixel
__global__ void convolution(PPMPixel *imageData, PPMPixel *outputData, int width, int height)
{
    __shared__ PPMPixel imageData_s[INPUT_TILE_SIZE][INPUT_TILE_SIZE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // get output tile row and col
    int row_o = blockIdx.y * OUTPUT_TILE_SIZE + ty;
    int col_o = blockIdx.x * OUTPUT_TILE_SIZE + tx;

    // get input tile row and col
    int row_i = row_o - FILTER_SIZE / 2;
    int col_i = col_o - FILTER_SIZE / 2;

    // if input is in bounds read from global to shared memory
    if ((row_i >= 0) && (row_i < height) && (col_i >= 0) && (col_i < width))
    {
        imageData_s[ty][tx] = imageData[row_i * width + col_i];
    }
    else // set pixel to black (all zero)
    {
        imageData_s[ty][tx].red = 0;
        imageData_s[ty][tx].blue = 0;
        imageData_s[ty][tx].green = 0;
    }

    __syncthreads();

    int red = 0, blue = 0, green = 0;

    // if in bounds calculate convolution for this pixel
    if ((ty < OUTPUT_TILE_SIZE) && (tx < OUTPUT_TILE_SIZE))
    {
        int i, j;
        for (i = 0; i < FILTER_SIZE; i++)
        {
            for (j = 0; j < FILTER_SIZE; j++)
            {
                red   += filter_c.data[j * FILTER_SIZE + i] * imageData_s[j + ty][i + tx].red;
                blue  += filter_c.data[j * FILTER_SIZE + i] * imageData_s[j + ty][i + tx].blue;
                green += filter_c.data[j * FILTER_SIZE + i] * imageData_s[j + ty][i + tx].green;
            }
        }

        // write value to output, saturate between 0 and 255
        if ((row_o < height) && (col_o < width))
        {
            outputData[row_o * width + col_o].red   = min( max( (int)(filter_c.factor * red   + filter_c.bias), 0), 255);
            outputData[row_o * width + col_o].blue  = min( max( (int)(filter_c.factor * blue  + filter_c.bias), 0), 255);
            outputData[row_o * width + col_o].green = min( max( (int)(filter_c.factor * green + filter_c.bias), 0), 255);
        }
    }
}