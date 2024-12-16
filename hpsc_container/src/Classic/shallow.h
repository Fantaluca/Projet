#ifndef SHALLOW_H
#define SHALLOW_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <stdint.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif



// Define macros
#define INPUT_DIR "input_data/base_case/"
#define MAX_PATH_LENGTH 512

#define GET(data, i, j) ((data)->values[(i) + (j) * (data)->nx])
#define SET(data, i, j, val) ((data)->values[(i) + (j) * (data)->nx] = (val))
#define GET_TIME() clock()/(double)CLOCKS_PER_SEC

// Structure definitions
typedef struct {
    double *values;
    int nx, ny;
    double dx, dy;
} data_t;

struct parameters {
    double dx, dy, dt, max_t;
    double g, gamma;
    int source_type;
    int sampling_rate;
    int boundary_type;
    char input_h_filename[MAX_PATH_LENGTH];
    char output_eta_filename[MAX_PATH_LENGTH];
    char output_u_filename[MAX_PATH_LENGTH];
    char output_v_filename[MAX_PATH_LENGTH];
};
// Function declarations
double interpolate_data(const data_t *data, double x, double y);
double update_eta(int nx, int ny, const struct parameters param, data_t *u, data_t *v, data_t *eta, data_t *h_interp);
double update_velocities(int nx, int ny, const struct parameters param, data_t *u, data_t *v, data_t *eta);
void boundary_condition(int n, int nx, int ny, const struct parameters param, data_t *u, data_t *v, data_t *eta, const data_t *h_interp);
void interp_bathy(int nx, int ny, const struct parameters param, data_t *h_interp, data_t *h);

// Utility functions (presumably defined elsewhere but needed by main)
int read_parameters(struct parameters *param, const char *filename);
void print_parameters(const struct parameters *param);
int read_data(data_t *data, const char *filename);
int init_data(data_t *data, int nx, int ny, double dx, double dy, double initial_value);
int write_data_vtk(const data_t *data, const char *name, const char *filename, int step);
int write_manifest_vtk(const char *filename, double dt, int nt, int sampling_rate);
void free_data(data_t *data);

#endif // SHALLOW_H