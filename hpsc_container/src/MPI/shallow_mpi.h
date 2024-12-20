#ifndef SHALLOW_OMP_MPI_H
#define SHALLOW_OMP_MPI_H

/*===========================================================
 * SHALLOW WATER EQUATIONS SOLVER - PARALLEL MPI
 * Header File
 * Contains type definitions, macros, and function prototypes
 ===========================================================*/

// Standard Library Headers
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <ctype.h>

// Parallel Computing Libraries
#include <mpi.h>

/*===========================================================
 * TIMING AND PERFORMANCE MACROS
 ===========================================================*/
#if defined(_OPENMP)
#include <omp.h>
#define GET_TIME() (omp_get_wtime())
#else
#define GET_TIME() ((double)clock() / CLOCKS_PER_SEC)
#endif

/*===========================================================
 * CONSTANTS AND CONFIGURATION MACROS
 ===========================================================*/
#define INPUT_DIR getenv("SHALLOW_INPUT_DIR")
#define MAX_PATH_LENGTH 512

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/*===========================================================
 * DATA ACCESS AND MANIPULATION MACROS
 ===========================================================*/
#define GET(data, i, j) ((data)->vals[(data)->nx * (j) + (i)])
#define SET(data, i, j, val) ((data)->vals[(data)->nx * (j) + (i)] = (val))
#define RANK_NX(gdata, rank) ((gdata)->rank_glob[rank][0].n)
#define RANK_NY(gdata, rank) ((gdata)->rank_glob[rank][1].n)
#define START_I(gdata, rank) ((gdata)->rank_glob[rank][0].start)
#define END_I(gdata, rank)   ((gdata)->rank_glob[rank][0].end)
#define START_J(gdata, rank) ((gdata)->rank_glob[rank][1].start)
#define END_J(gdata, rank)   ((gdata)->rank_glob[rank][1].end)

/*===========================================================
 * ENUMERATION AND TYPE DEFINITIONS
 ===========================================================*/
// Neighbor directions
typedef enum {
    LEFT = 0,
    RIGHT = 1,
    UP = 2,
    DOWN = 3,
    NEIGHBOR_NUM = 4
} neighbour_t;

// Domain boundary limits
typedef struct {
    int start;
    int end;
    int n;
} limit_t;

typedef struct {
    double *vals;
    double **edge_vals;
    int nx, ny;
    double dx, dy;
} data_t;

// MPI Topology management
typedef struct {
    int nb_process;
    int rank;
    int cart_rank;
    int dims[2];
    int coords[2];
    int neighbors[NEIGHBOR_NUM];
    MPI_Comm cart_comm;
} MPITopology;

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

typedef struct {
    data_t *u;
    data_t *v;
    data_t *eta;
    data_t *h;
    data_t *h_interp;
} all_data_t;

typedef struct {
    data_t *gathered_output;
    double *receive_data_eta;
    double *receive_data_u;
    double *receive_data_v;
    limit_t **rank_glob;
    int *recv_size_eta;
    int *recv_size_u;
    int *recv_size_v;
    int *displacements_eta;
    int *displacements_u;
    int *displacements_v;
} gather_data_t;


/*===========================================================
 * FUNCTION PROTOTYPES
 ===========================================================*/

// MPI and Initialization Functions
int initialize_mpi_topology(int argc, char **argv, MPITopology *topo);
int initialize_gather_structures(const MPITopology *topo, 
                                 gather_data_t *gdata,
                                 int nx, int ny, 
                                 double dx, double dy);

// Simulation Core Functions
void check_cfl(parameters_t param, all_data_t *all_data, MPITopology *topo);
void update_velocities(const parameters_t param,
                       all_data_t *all_data,
                       gather_data_t *gdata,
                       MPITopology *topo);
void update_eta(const parameters_t param, 
                all_data_t *all_data,
                gather_data_t *gdata,
                MPITopology *topo);

// Interpolation and Preprocessing
double interpolate_data(const data_t *data, 
                        int nx_glob, 
                        int ny_glob, 
                        double x, 
                        double y);
void interp_bathy(const parameters_t param, 
                  int nx_glob, 
                  int ny_glob, 
                  all_data_t *all_data, 
                  gather_data_t *gdata, 
                  MPITopology *topo);

// Boundary and Source Management
void apply_source(int timestep, 
                  int nx_glob, 
                  int ny_glob, 
                  const parameters_t param, 
                  all_data_t *all_data, 
                  gather_data_t *gdata, 
                  MPITopology *topo);
void boundary_conditions(const parameters_t param, 
                         all_data_t *all_data, 
                         MPITopology *topo);

// Data Gathering and Output
void gather_and_assemble_data(const parameters_t param,
                               all_data_t *all_data,
                               gather_data_t *gdata,
                               MPITopology *topo,
                               int nx_glob, 
                               int ny_glob,
                               int timestep);

// Additional Tool Functions
int read_parameters(parameters_t *param, const char *filename);
void print_parameters(const parameters_t *param);
int read_data(data_t *data, const char *filename);
int write_data(const data_t *data, const char *filename, int step);
int write_data_vtk(const data_t *data, const char *name, const char *filename, int step);
int write_manifest_vtk(const char *filename, double dt, int nt, int sampling_rate);
int init_data(data_t *data, int nx, int ny, double dx, double dy, double val);
void free_data(data_t *data);
all_data_t* init_all_data(const parameters_t *param, MPITopology *topo);
void print_progress(int current_step, int total_steps, double start_time, MPITopology *top);
void free_all_data(all_data_t* all_data);
void cleanup(parameters_t *param, MPITopology *topo, gather_data_t *gdata);
void cleanup_mpi_topology(MPITopology *topo);

#endif // SHALLOW_OMP_MPI_H