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
#include <math.h>
#include <time.h>
#include <stdint.h>

/*===========================================================
 * CONSTANTS AND CONFIGURATION MACROS
 ===========================================================*/
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define INPUT_DIR getenv("SHALLOW_INPUT_DIR")
#define MAX_PATH_LENGTH 512

/*===========================================================
 * DATA ACCESS AND TIMING MACROS
 ===========================================================*/
// Access data in 2D array stored in 1D
#define GET(data, i, j) ((data)->values[(i) + (j) * (data)->nx])
#define SET(data, i, j, val) ((data)->values[(i) + (j) * (data)->nx] = (val))

// Timing functions
#define GET_TIME() clock()/(double)CLOCKS_PER_SEC

/*===========================================================
 * TYPE DEFINITIONS
 ===========================================================*/

/**
 * Grid data structure
 * Contains field values and grid information
 */
typedef struct {
    double *values;              // Field values
    int nx, ny;                 // Grid dimensions
    double dx, dy;              // Grid spacing
} data_t;

/**
 * Simulation parameters structure
 * Contains all configuration parameters for the simulation
 */
typedef struct  {
    double dx, dy, dt, max_t;    // Grid and time step parameters
    double g, gamma;             // Physical parameters
    int source_type;             // Source configuration
    int sampling_rate;           // Output frequency
    int boundary_type;           // Boundary condition type
    char input_h_filename[MAX_PATH_LENGTH];      // Input file paths
    char output_eta_filename[MAX_PATH_LENGTH];   // Output file paths
    char output_u_filename[MAX_PATH_LENGTH];
    char output_v_filename[MAX_PATH_LENGTH];
} parameters_t;

/*===========================================================
 * COMPUTATION FUNCTION PROTOTYPES
 ===========================================================*/

/**
 * Performs bilinear interpolation of data at given coordinates
 */
double interpolate_data(const data_t *data, double x, double y);

/**
 * Updates water height (eta) using shallow water equations
 */
double update_eta(int nx, int ny, parameters_t param, 
                 data_t *u, data_t *v, data_t *eta, data_t *h_interp);

/**
 * Updates velocity fields (u,v) using shallow water equations
 */
double update_velocities(int nx, int ny, parameters_t param,
                        data_t *u, data_t *v, data_t *eta);

/**
 * Apply boundary conditions and source terms
 */
void boundary_condition(int n, int nx, int ny, parameters_t param,
                       data_t *u, data_t *v, data_t *eta, const data_t *h_interp);

/**
 * Interpolates bathymetry data onto computation grid
 */
void interp_bathy(int nx, int ny, parameters_t param,
                  data_t *h_interp, data_t *h);

/*===========================================================
 * I/O AND INITIALIZATION FUNCTION PROTOTYPES
 ===========================================================*/

/**
 * Read simulation parameters from configuration file
 */
int read_parameters(parameters_t *param, const char *filename);

/**
 * Display current simulation parameters
 */
void print_parameters(parameters_t *param);

/**
 * Read data from input file into data structure
 */
int read_data(data_t *data, const char *filename);

/**
 * Initialize data structure with given dimensions and value
 */
int init_data(data_t *data, int nx, int ny, double dx, double dy, 
              double initial_value);

/**
 * Write data to VTK format file for visualization
 */
int write_data_vtk(const data_t *data, const char *name, 
                   const char *filename, int step);

/**
 * Write VTK manifest file for time series data
 */
int write_manifest_vtk(const char *filename, double dt, int nt, 
                      int sampling_rate);

/**
 * Free memory allocated for data structure
 */
void free_data(data_t *data);

#endif // SHALLOW_H