#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "sidbmp.h"

int main(int argc, char **argv)
{

  srand(time(NULL));
  Image * im = generate_image(3, 3);
  print_image(im);
  delete_image(im);


  return 1;
}