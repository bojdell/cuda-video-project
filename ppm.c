#include <stdio.h>
#include <stdlib.h>
#include <time.h>

typedef struct {
     unsigned char red,green,blue;
} PPMPixel;

typedef struct {
     int x, y;
     PPMPixel *data;
} PPMImage;

typedef struct {
     int x, y;
     int *data;
} Filter;

#define CREATOR "DA BROS"
#define RGB_COMPONENT_COLOR 255

static PPMImage *readPPM(const char *filename)
{
         char buff[16];
         PPMImage *img;
         FILE *fp;
         int c, rgb_comp_color;
         //open PPM file for reading
         fp = fopen(filename, "rb");
         if (!fp) {
              // fprintf(stderr, "Hey! Unable to open file '%s'\n", filename);
              // exit(1);
            return NULL;
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

// process a single PPM image
void processPPM(PPMImage *img)
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

// ppm <stride_len> (<max_frames>)
int main(int argc, char *argv[]){
    if ( argc < 2 ) /* argc should be at least 2 for correct execution */
    {
        /* We print argv[0] assuming it is the program name */
        printf("usage: %s <stride_len> (<max_frames>)\n", argv[0]);
        return -1;
    }

    char instr[80];
    char outstr[80];
    int i = 0, j = 0;
    int numFrames = 301;
    int stride_len = atoi(argv[1]);
    int numChunks = numFrames / stride_len;
    if(numFrames % stride_len) numChunks++;

    clock_t begin, end;
    double time_spent[numChunks + 1];   // time spent for each chunk, plus accumulate total at end
    time_spent[numChunks - 1] = 0;      // init accumulator to 0

    PPMImage images[stride_len];

    // loop over all frames, in chunks of stride_len
    for(i = 0; i < numChunks; i ++) {

        // read 1 chunk of stride_len frames into images[]
        for(j = 0; j < stride_len && j + i*stride_len < numFrames; j++) {
            sprintf(instr, "infiles/baby%03d.ppm", i*stride_len + j + 1);
            PPMImage * img = readPPM(instr);
            if(img == NULL) {
                printf("All files processed\n");
                return 0;
            }
            images[j] = *img;
        }
        
        begin = clock();

        // process chunk of frames in images[]
        for(j = 0; j < stride_len && j + i*stride_len < numFrames; j++) {
            processPPM(&images[j]);
        }
        
        end = clock();
        time_spent[i] = (double)(end - begin) / CLOCKS_PER_SEC;
        time_spent[numChunks - 1] += time_spent[i];

        // write 1 chunk of stride_len frames from images[]
        for(j = 0; j < stride_len && j + i*stride_len < numFrames; j++) {
            sprintf(outstr, "outfiles/baby%03d.ppm", i*stride_len + j + 1);
            writePPM(outstr, &images[j]);
        }
    }

    printf("%f seconds spent\n", time_spent[numChunks - 1]);

    return 0;
}