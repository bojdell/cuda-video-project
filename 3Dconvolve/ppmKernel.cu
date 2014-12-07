#include "ppm.h"
#include <math.h>
#include <stdio.h>
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

__global__ void convolution(PPMPixel *imageData, PPMPixel *outputData)
{
    // __shared__ PPMPixel imageData_s[BLOCK_SIZE][BLOCK_SIZE][INPUT_TILE_Z];
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tz = threadIdx.z;

    int row_i = blockIdx.y * BLOCK_SIZE + ty;
    int col_i = blockIdx.x * BLOCK_SIZE + tx;
    int depth_i = blockIdx.z * INPUT_TILE_Z + tz;


    int row_o = row_i - FILTER_SIZE / 2;
    int col_o = col_i - FILTER_SIZE / 2;
    int depth_o = depth_i - FILTER_SIZE / 2;

    // // if ((row_ >= 0) && (row_i < height) && (col_i >= 0) && (col_i < width)
    // //     && (depth_i >= 0) && (depth_i < depth))
    // // {
    // imageData_s[tz][ty][tx] = imageData[depth_i*INPUT_TILE_X*INPUT_TILE_Y + col_i * INPUT_TILE_X + row_i];
    // // }
    // // else
    // // {
    // //     imageData_s[tz][ty][tx].red = 0;
    // //     imageData_s[tz][ty][tx].blue = 0;
    // //     imageData_s[tz][ty][tx].green = 0;
    // // }

    // __syncthreads();
    // printf("%d %d %d\n", row_o, col_o, depth_o);

    int red = 0, blue = 0, green = 0;
    if ( (FILTER_SIZE/2 <= col_i) && (col_i < INPUT_TILE_X - FILTER_SIZE /2) &&
         (FILTER_SIZE/2 <= row_i) && (row_i < INPUT_TILE_Y - FILTER_SIZE /2) &&
         (FILTER_SIZE/2 <= depth_i) && (depth_i < INPUT_TILE_Z - FILTER_SIZE /2) )
    {
        int i, j, k;
        for (i = 0; i < FILTER_SIZE; i++)
        {
            for (j = 0; j < FILTER_SIZE; j++)
            {
                for (k = 0; k < FILTER_SIZE; k++)
                {
                    red   += filter_c.data[k][j][i] * imageData[(k + depth_i - FILTER_SIZE/2)*INPUT_TILE_X*INPUT_TILE_Y + (j + col_i - FILTER_SIZE/2)*INPUT_TILE_X + (i + row_i - FILTER_SIZE/2)].red;
                    blue  += filter_c.data[k][j][i] * imageData[(k + depth_i - FILTER_SIZE/2)*INPUT_TILE_X*INPUT_TILE_Y + (j + col_i - FILTER_SIZE/2)*INPUT_TILE_X + (i + row_i - FILTER_SIZE/2)].blue;
                    green += filter_c.data[k][j][i] * imageData[(k + depth_i - FILTER_SIZE/2)*INPUT_TILE_X*INPUT_TILE_Y + (j + col_i - FILTER_SIZE/2)*INPUT_TILE_X + (i + row_i - FILTER_SIZE/2)].green;                    
                }
            }
        }

        outputData[depth_o * OUTPUT_TILE_X * OUTPUT_TILE_Y + row_o * OUTPUT_TILE_X + col_o].red   = min( max( (int)(filter_c.factor * red   + filter_c.bias), 0), 255);
        outputData[depth_o * OUTPUT_TILE_X * OUTPUT_TILE_Y + row_o * OUTPUT_TILE_X + col_o].blue  = min( max( (int)(filter_c.factor * blue  + filter_c.bias), 0), 255);
        outputData[depth_o * OUTPUT_TILE_X * OUTPUT_TILE_Y + row_o * OUTPUT_TILE_X + col_o].green = min( max( (int)(filter_c.factor * green + filter_c.bias), 0), 255);

    }
}