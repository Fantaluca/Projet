#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <math.h>


#if defined(_OPENMP)
#include <omp.h>
#define GET_TIME() (omp_get_wtime()) // wall time
#else
#define GET_TIME() ((double)clock() / CLOCKS_PER_SEC) // cpu time
#endif

// Define macros
#define INPUT_DIR "input_data/base_case/"
#define MAX_PATH_LENGTH 512

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define GET(data, i, j) ((data)->values[(data)->nx * (j) + (i)])
#define SET(data, i, j, val) ((data)->values[(data)->nx * (j) + (i)] = (val))

// Define structures
struct parameters {
  double dx, dy, dt, max_t;
  double g, gamma;
  int source_type;
  int sampling_rate;
  char input_h_filename[MAX_PATH_LENGTH];
  char output_eta_filename[MAX_PATH_LENGTH];
  char output_u_filename[MAX_PATH_LENGTH];
  char output_v_filename[MAX_PATH_LENGTH];
};

struct data {
  int nx, ny;
  double dx, dy;
  double *values;
};


