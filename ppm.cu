#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "ppmKernel.cu"
#include "ppm.h"

// typedef struct {
//      unsigned char red,green,blue;
// } PPMPixel;

// typedef struct {
//      int x, y;
//      PPMPixel *data;
// } PPMImage;

#define CREATOR "RPFELGUEIRAS"
#define RGB_COMPONENT_COLOR 255

#define OUTPUT_TILE_SIZE 12


#define FATAL(msg, ...) \
    do {\
        fprintf(stderr, "[%s:%d] "msg"\n", __FILE__, __LINE__, ##__VA_ARGS__);\
        exit(-1);\
    } while(0)

static PPMImage *readPPM(const char *filename)
{
         char buff[16];
         PPMImage *img;
         FILE *fp;
         int c, rgb_comp_color;
         //open PPM file for reading
         fp = fopen(filename, "rb");
         if (!fp) {
              fprintf(stderr, "Unable to open file '%s'\n", filename);
              exit(1);
         }

         //read image format
         if (!fgets(buff, sizeof(buff), fp)) {
              perror(filename);
              exit(1);
         }

    //check the image format
    if (buff[0] != 'P' || buff[1] != '6') {
         fprintf(stderr, "Invalid image format (must be 'P6')\n");
         exit(1);
    }

    //alloc memory form image
    img = (PPMImage *)malloc(sizeof(PPMImage));
    if (!img) {
         fprintf(stderr, "Unable to allocate memory\n");
         exit(1);
    }

    //check for comments
    c = getc(fp);
    while (c == '#') {
    while (getc(fp) != '\n') ;
         c = getc(fp);
    }

    ungetc(c, fp);
    //read image size information
    if (fscanf(fp, "%d %d", &img->x, &img->y) != 2) {
         fprintf(stderr, "Invalid image size (error loading '%s')\n", filename);
         exit(1);
    }

    //read rgb component
    if (fscanf(fp, "%d", &rgb_comp_color) != 1) {
         fprintf(stderr, "Invalid rgb component (error loading '%s')\n", filename);
         exit(1);
    }

    //check rgb component depth
    if (rgb_comp_color!= RGB_COMPONENT_COLOR) {
         fprintf(stderr, "'%s' does not have 8-bits components\n", filename);
         exit(1);
    }

    while (fgetc(fp) != '\n') ;
    //memory allocation for pixel data
    img->data = (PPMPixel*)malloc(img->x * img->y * sizeof(PPMPixel));

    if (!img) {
         fprintf(stderr, "Unable to allocate memory\n");
         exit(1);
    }

    //read pixel data from file
    if (fread(img->data, 3 * img->x, img->y, fp) != img->y) {
         fprintf(stderr, "Error loading image '%s'\n", filename);
         exit(1);
    }

    fclose(fp);
    return img;
}
void writePPM(const char *filename, PPMImage *img)
{
    FILE *fp;
    //open file for output
    fp = fopen(filename, "wb");
    if (!fp) {
         fprintf(stderr, "Unable to open file '%s'\n", filename);
         exit(1);
    }

    //write the header file
    //image format
    fprintf(fp, "P6\n");

    //comments
    fprintf(fp, "# Created by %s\n",CREATOR);

    //image size
    fprintf(fp, "%d %d\n",img->x,img->y);

    // rgb component depth
    fprintf(fp, "%d\n",RGB_COMPONENT_COLOR);

    // pixel data
    fwrite(img->data, 3 * img->x, img->y, fp);
    fclose(fp);
}

void changeColorPPM(PPMImage *img)
{
    int i;
    if(img){

         for(i=0;i<img->x*img->y;i++){
              int avg = (img->data[i].red + img->data[i].green + img ->data[i].blue) / 3;

              /*
              img->data[i].red=RGB_COMPONENT_COLOR-img->data[i].red;
              img->data[i].green=RGB_COMPONENT_COLOR-img->data[i].green;
              img->data[i].blue=RGB_COMPONENT_COLOR-img->data[i].blue;
              */

              img->data[i].red = avg;
              img->data[i].green = avg;
              img->data[i].blue = avg;
         }
    }
}

int main(){

    clock_t begin, end;
    double time_spent;

    
    /* here, do your time-consuming job */

    char instr[80];
    char outstr[80];
    int i = 0;

    PPMImage images[301];

    // for(i = 0; i < 301; i++) {
    //     sprintf(instr, "infiles/baby001.ppm", i+1);
    //     images[i] = *readPPM(instr);
    // }
    
    PPMPixel *imageData_d, *outputData_d, *outputData_h;

    cudaError_t cuda_ret;

    for(i = 0; i < 301; i++) {
        sprintf(instr, "infiles/baby%03d.ppm", i+1);
        images[i] = *readPPM(instr);
    }
    PPMImage *image;
    image = &images[0];
    outputData_h = (PPMPixel *)malloc(image->x*image->y*sizeof(PPMPixel));

    cuda_ret = cudaMalloc((void**)&(imageData_d), image->x*image->y*sizeof(PPMPixel));
    if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory");

    cuda_ret = cudaMalloc((void**)&(outputData_d), image->x*image->y*sizeof(PPMPixel));
    if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory");

    PPMImage *outImage;
    outImage = (PPMImage *)malloc(sizeof(PPMImage));
    outImage->x = image->x;
    outImage->y = image->y;


    begin = clock();

    for(i = 0; i < 301; i++) {
        sprintf(outstr, "outfiles/baby%03d.ppm", i+1);

        image = &images[i];

        cuda_ret = cudaMemcpy(imageData_d, image->data, image->x*image->y*sizeof(PPMPixel), cudaMemcpyHostToDevice);
        if(cuda_ret != cudaSuccess) FATAL("Unable to copy to device");


        dim3 dim_grid, dim_block;
        dim_grid = dim3(image->y, 1,1);
        dim_block = dim3(image->x, 1, 1);
        // if(image->x % TILE_SIZE)
        //     dim_grid.x++;

        // if(image->y % TILE_SIZE)
        //     dim_grid.y++;


        blackAndWhite<<<dim_grid, dim_block>>>(imageData_d, outputData_d, image->x, image->y);        

        cuda_ret = cudaDeviceSynchronize();
        if(cuda_ret != cudaSuccess) FATAL("Unable to launch/execute kernel");

        cuda_ret = cudaMemcpy(outputData_h, outputData_d, image->x*image->y*sizeof(PPMPixel), cudaMemcpyDeviceToHost);
        if(cuda_ret != cudaSuccess) FATAL("Unable to copy to host");


        outImage->data = outputData_h;

        writePPM(outstr,outImage);

    }
    end = clock();

    free(outputData_h);
    free(outImage);
    cudaFree(imageData_d);
    cudaFree(outputData_d);

    // for(i = 1; i <= 1; i++) {
        // sprintf(instr, "infiles/baby001.ppm");
        // sprintf(outstr, "outfiles/baby001.ppm");

        // PPMImage *image;
        // sprintf(instr, "infiles/baby001.ppm");

        // image = readPPM(instr);



        // changeColorPPM(&images[i-1]);


    // }

    time_spent = (double)(end - begin) / CLOCKS_PER_SEC;

    printf("%f seconds spent\n", time_spent);
    
}