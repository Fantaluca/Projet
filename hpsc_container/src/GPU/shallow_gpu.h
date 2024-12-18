/*===========================================================
 * SHALLOW WATER EQUATIONS SOLVER - PARALLEL GPU
 * Header File
 * Contains type definitions, macros, and function prototypes
 ===========================================================*/

#ifndef SHALLOW_GPU_H
#define SHALLOW_GPU_H

/*===========================================================
 * STANDARD LIBRARY INCLUDES
 ===========================================================*/
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <string.h>
#include <time.h>

/*===========================================================
 * PARALLEL COMPUTING AND GPU LIBRARIES
 ===========================================================*/
#if defined(_OPENMP)
#include <omp.h>
#define GET_TIME() (omp_get_wtime())  // wall time
#else
#define GET_TIME() ((double)clock() / CLOCKS_PER_SEC)  // cpu time
#endif

/*===========================================================
 * CONSTANTS AND CONFIGURATION MACROS
 ===========================================================*/
#define INPUT_DIR "../../input_data/base_case/"
#define MAX_PATH_LENGTH 512

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/*===========================================================
 * DATA ACCESS AND MANIPULATION MACROS
 ===========================================================*/
#define GET(data, i, j) ((data)->values[(data)->nx * (j) + (i)])
#define SET(data, i, j, val) ((data)->values[(data)->nx * (j) + (i)] = (val))

/*===========================================================
 * TYPE DEFINITIONS AND STRUCTURES
 ===========================================================*/
// Simulation parameters
typedef struct {
    double dx, dy, dt, max_t;
    double g, gamma;
    int source_type;
    int sampling_rate;
    double latitude;
    int boundary_type;
    double f;
    char input_h_filename[MAX_PATH_LENGTH];
    char output_eta_filename[MAX_PATH_LENGTH];
    char output_u_filename[MAX_PATH_LENGTH];
    char output_v_filename[MAX_PATH_LENGTH];
} parameters_t;

// Grid data structure
typedef struct {
    double *values;
    int nx, ny;
    double dx, dy;
} data_t;

// Combined simulation data
typedef struct {
    data_t *u;
    data_t *v;
    data_t *eta;
    data_t *h;
    data_t *h_interp;
} all_data_t;

/*===========================================================
 * GPU DATA MAPPING DECLARATIONS
 ===========================================================*/
#pragma omp declare mapper(data_t data) \
    map(to: data.nx, data.ny, data.dx, data.dy) \
    map(tofrom: data.values[0:data.nx*data.ny])

/*===========================================================
 * FUNCTION PROTOTYPES - CORE COMPUTATION
 ===========================================================*/
void update_velocities(int nx, int ny, const parameters_t param, all_data_t *all_data);
void update_eta(int nx, int ny, const parameters_t param, all_data_t *all_data);

/*===========================================================
 * FUNCTION PROTOTYPES - INTERPOLATION AND PREPROCESSING
 ===========================================================*/
double interpolate_data(const data_t *data, double x, double y);
void interp_bathy(int nx, int ny, const parameters_t param, all_data_t *all_data);

/*===========================================================
 * FUNCTION PROTOTYPES - BOUNDARY CONDITIONS AND SOURCE TERMS
 ===========================================================*/
void apply_source(int n, int nx, int ny, const parameters_t param, all_data_t *all_data);
void boundary_conditions(int nx, int ny, const parameters_t param, all_data_t *all_data);

/*===========================================================
 * FUNCTION PROTOTYPES - FILE I/O AND PARAMETERS
 ===========================================================*/
int read_parameters(parameters_t *param, const char *filename);
void print_parameters(const parameters_t *param);
int read_data(data_t *data, const char *filename);
int write_data(const data_t *data, const char *filename, int step);
int write_data_vtk(const data_t *data, const char *name, const char *filename, int step);
int write_manifest_vtk(const char *filename, double dt, int nt, int sampling_rate);

/*===========================================================
 * FUNCTION PROTOTYPES - INITIALIZATION AND MEMORY MANAGEMENT
 ===========================================================*/
int init_data(data_t *data, int nx, int ny, double dx, double dy, double val);
void free_data(data_t *data);
all_data_t* init_all_data(const parameters_t *param);
void free_all_data(all_data_t* all_data);

/*===========================================================
 * FUNCTION PROTOTYPES - UTILITY FUNCTIONS
 ===========================================================*/
void print_progress(int current_step, int total_steps, double start_time);

#endif // SHALLOW_GPU_H