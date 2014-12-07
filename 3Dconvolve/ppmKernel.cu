#include "ppm.h"
#include <math.h>

__constant__ Filter3D filter_c;

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

__global__ void convolution(PPMPixel *imageData, PPMPixel *outputData, int width, int height, int depth)
{
    __shared__ PPMPixel imageData_s[INPUT_TILE_SIZE][INPUT_TILE_SIZE][INPUT_TILE_SIZE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tz = threadIdx.z;

    int row_o = blockIdx.y * OUTPUT_TILE_SIZE + ty;
    int col_o = blockIdx.x * OUTPUT_TILE_SIZE + tx;
    int depth_o = blockIdx.x * OUTPUT_TILE_SIZE + tz;

    int row_i = row_o - FILTER_SIZE / 2;
    int col_i = col_o - FILTER_SIZE / 2;
    int depth_i = depth_o - FILTER_SIZE / 2;

    if ((row_i >= 0) && (row_i < height) && (col_i >= 0) && (col_i < width)
        && (depth_i >= 0) && (depth_i < depth))
    {
        imageData_s[tz][ty][tx] = imageData[depth_i * width * height + row_i * width + col_i];
    }
    else
    {
        imageData_s[tz][ty][tx].red = 0;
        imageData_s[tz][ty][tx].blue = 0;
        imageData_s[tz][ty][tx].green = 0;
    }

    __syncthreads();

    int red = 0, blue = 0, green = 0;

    if ((tz < OUTPUT_TILE_SIZE) && (ty < OUTPUT_TILE_SIZE) && (tx < OUTPUT_TILE_SIZE))
    {
        int i, j, k;
        for (i = 0; i < FILTER_SIZE; i++)
        {
            for (j = 0; j < FILTER_SIZE; j++)
            {
                for (k = 0; k < FILTER_SIZE; k++)
                {
                    red   += filter_c.data[k][j][i] * imageData_s[k + tz][j + ty][i + tx].red;
                    blue  += filter_c.data[k][j][i] * imageData_s[k + tz][j + ty][i + tx].blue;
                    green += filter_c.data[k][j][i] * imageData_s[k + tz][j + ty][i + tx].green;                    
                }
            }
        }

        if ((depth_o < depth) && (row_o < height) && (col_o < width))
        {
            outputData[depth_o * width * height + row_o * width + col_o].red   = min( max( (int)(filter_c.factor * red   + filter_c.bias), 0), 255);
            outputData[depth_o * width * height + row_o * width + col_o].blue  = min( max( (int)(filter_c.factor * blue  + filter_c.bias), 0), 255);
            outputData[depth_o * width * height + row_o * width + col_o].green = min( max( (int)(filter_c.factor * green + filter_c.bias), 0), 255);
        }
    }
}