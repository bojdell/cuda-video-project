#include "ppm.h"
#include <math.h>
#include <stdio.h>

// Constant memory for the filter
__constant__ Filter3D filter_c;

__global__ void convolution(PPMPixel *imageData, PPMPixel *outputData)
{
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tz = threadIdx.z;

    // get input tile dimensions
    int row_i = blockIdx.y * BLOCK_SIZE + ty;
    int col_i = blockIdx.x * BLOCK_SIZE + tx;
    int depth_i = blockIdx.z * INPUT_TILE_Z + tz;

    // get output tile dimensions
    int row_o = row_i - FILTER_SIZE / 2;
    int col_o = col_i - FILTER_SIZE / 2;
    int depth_o = depth_i - FILTER_SIZE / 2;

    // initialize pixel value variables
    int red = 0, blue = 0, green = 0;
    // if pixel is within bounds calculate the convolution for it
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
        // write values out to outputData, saturating at 255 and 0
        outputData[depth_o * OUTPUT_TILE_X * OUTPUT_TILE_Y + row_o * OUTPUT_TILE_X + col_o].red   = min( max( (int)(filter_c.factor * red   + filter_c.bias), 0), 255);
        outputData[depth_o * OUTPUT_TILE_X * OUTPUT_TILE_Y + row_o * OUTPUT_TILE_X + col_o].blue  = min( max( (int)(filter_c.factor * blue  + filter_c.bias), 0), 255);
        outputData[depth_o * OUTPUT_TILE_X * OUTPUT_TILE_Y + row_o * OUTPUT_TILE_X + col_o].green = min( max( (int)(filter_c.factor * green + filter_c.bias), 0), 255);

    }
}