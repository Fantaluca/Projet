#ifndef SHALLOW_H
#define SHALLOW_H

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <string.h>
#include <time.h>

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

// Declare function prototypes

// From "shallow.c"
double update_velocities(int nx, 
                         int ny, 
                         const struct parameters param, 
                         struct data *u, 
                         struct data *v, 
                         struct data *eta);
double update_eta(int nx, 
                  int ny, 
                  const struct parameters param, 
                  struct data *u, 
                  struct data *v, 
                  struct data *eta, 
                  struct data *h_interp);

double interpolate_data(const struct data *data, 
                        double x, 
                        double y);


// From "tools.c"
int write_data_vtk_temp(const struct data *data, const char *name,
                   const char *filename, int step);
int read_parameters(struct parameters *param, const char *filename);
void print_parameters(const struct parameters *param);
int read_data(struct data *data, const char *filename);
int write_data(const struct data *data, const char *filename, int step);
int write_data_vtk(const struct data *data, const char *name, const char *filename, int step);
int write_manifest_vtk(const char *filename, double dt, int nt, int sampling_rate);
int init_data(struct data *data, int nx, int ny, double dx, double dy, double val);
void free_data(struct data *data);

#endif // SHALLOW_H