#include "ppm.h"
#include <math.h>

__constant__ Filter filter_c;

__global__ void blackAndWhite(PPMPixel *imageData, PPMPixel *outputData, int width, int height, int imagesPerBlock) {

	int tx = threadIdx.x;
	int ty = blockIdx.x;
    for(int x = 0; x < imagesPerBlock; x++) {

    	if(ty < height && tx < width) {
    		int i = x*width*height + ty*width + tx;
    		int avg = (imageData[i].red + imageData[i].green + imageData[i].blue) / 3;

    		outputData[i].red = avg;
    		outputData[i].green = avg;
    		outputData[i].blue = avg;
    	}
    }
}

__global__ void convolution(PPMPixel *imageData, PPMPixel *outputData, int width, int height, int imagesPerBlock)
{
    __shared__ PPMPixel imageData_s[IMAGES_PER_BLOCK][INPUT_TILE_SIZE][INPUT_TILE_SIZE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row_o = blockIdx.y * OUTPUT_TILE_SIZE + ty;
    int col_o = blockIdx.x * OUTPUT_TILE_SIZE + tx;

    int row_i = row_o - FILTER_SIZE / 2;
    int col_i = col_o - FILTER_SIZE / 2;
    for(int x = 0; x < imagesPerBlock; x++) {
        if ((row_i >= 0) && (row_i < height) && (col_i >= 0) && (col_i < width))
        {
            imageData_s[x][ty][tx] = imageData[x*width*height + row_i * width + col_i];
        }
        else
        {
            imageData_s[x][ty][tx].red = 0;
            imageData_s[x][ty][tx].blue = 0;
            imageData_s[x][ty][tx].green = 0;
        }
    }
        __syncthreads();

        for(int x = 0; x < imagesPerBlock; x++) {


        int red = 0, blue = 0, green = 0;

        if ((ty < OUTPUT_TILE_SIZE) && (tx < OUTPUT_TILE_SIZE))
        {
            int i, j;
            for (i = 0; i < FILTER_SIZE; i++)
            {
                for (j = 0; j < FILTER_SIZE; j++)
                {
                    red   += filter_c.data[j * FILTER_SIZE + i] * imageData_s[x][j + ty][i + tx].red;
                    blue  += filter_c.data[j * FILTER_SIZE + i] * imageData_s[x][j + ty][i + tx].blue;
                    green += filter_c.data[j * FILTER_SIZE + i] * imageData_s[x][j + ty][i + tx].green;
                }
            }

            if ((row_o < height) && (col_o < width))
            {
                outputData[x*width*height + row_o * width + col_o].red   = min( max( (int)(filter_c.factor * red   + filter_c.bias), 0), 255);
                outputData[x*width*height + row_o * width + col_o].blue  = min( max( (int)(filter_c.factor * blue  + filter_c.bias), 0), 255);
                outputData[x*width*height + row_o * width + col_o].green = min( max( (int)(filter_c.factor * green + filter_c.bias), 0), 255);
            }
        }
    }

}