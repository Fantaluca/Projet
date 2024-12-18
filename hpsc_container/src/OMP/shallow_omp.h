/*===========================================================
 * SHALLOW WATER EQUATIONS SOLVER - SERIAL VERSION
 * Header File
 * Contains type definitions, macros, and function prototypes
 ===========================================================*/

#ifndef SHALLOW_H
#define SHALLOW_H

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
 * TIMING AND PERFORMANCE MACROS
 ===========================================================*/
#if defined(_OPENMP)
#include <omp.h>
#define GET_TIME() (omp_get_wtime())     // wall time
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
 * DATA ACCESS MACROS
 ===========================================================*/
#define GET(data, i, j) ((data)->values[(data)->nx * (j) + (i)])
#define SET(data, i, j, val) ((data)->values[(data)->nx * (j) + (i)] = (val))

/*===========================================================
 * TYPE DEFINITIONS
 ===========================================================*/

/**
 * Simulation parameters structure
 */
typedef struct {
    double dx, dy, dt, max_t;    // Grid and time step parameters
    double g, gamma;             // Physical parameters
    int source_type;             // Source configuration
    int sampling_rate;           // Output frequency
    char input_h_filename[MAX_PATH_LENGTH];      // Input file paths
    char output_eta_filename[MAX_PATH_LENGTH];   // Output file paths
    char output_u_filename[MAX_PATH_LENGTH];
    char output_v_filename[MAX_PATH_LENGTH];
} parameters_t;

/**
 * Grid data structure
 */
typedef struct {
    double *values;              // Field values
    int nx, ny;                 // Grid dimensions
    double dx, dy;              // Grid spacing
} data_t;

/**
 * Collection of all simulation fields
 */
typedef struct {
    data_t *u;                  // X-velocity
    data_t *v;                  // Y-velocity
    data_t *eta;                // Water elevation
    data_t *h;                  // Bathymetry
    data_t *h_interp;           // Interpolated bathymetry
} all_data_t;

/*===========================================================
 * COMPUTATION FUNCTION PROTOTYPES
 ===========================================================*/

// Core computation functions
void update_velocities(int nx, int ny, const parameters_t param, all_data_t *all_data);
void update_eta(int nx, int ny, const parameters_t param, all_data_t *all_data);

// Boundary and source terms
void boundary_conditions(int nx, int ny, const parameters_t param, all_data_t *all_data);
void apply_source(int timestep, int nx, int ny, const parameters_t param, all_data_t *all_data);

// Interpolation functions
double interpolate_data(const data_t *data, double x, double y);
void interp_bathy(int nx, int ny, const parameters_t param, all_data_t *all_data);

/*===========================================================
 * I/O AND INITIALIZATION FUNCTION PROTOTYPES
 ===========================================================*/

// Parameter and data I/O
int read_parameters(parameters_t *param, const char *filename);
void print_parameters(const parameters_t *param);
int read_data(data_t *data, const char *filename);

// Data output functions
int write_data(const data_t *data, const char *filename, int step);
int write_data_vtk(const data_t *data, const char *name, const char *filename, int step);
int write_manifest_vtk(const char *filename, double dt, int nt, int sampling_rate);

// Initialization and cleanup
int init_data(data_t *data, int nx, int ny, double dx, double dy, double val);
void free_data(data_t *data);
all_data_t* init_all_data(const parameters_t *param);
void free_all_data(all_data_t* all_data);

// Utility functions
void print_progress(int current_step, int total_steps, double start_time);

#endif // SHALLOW_H