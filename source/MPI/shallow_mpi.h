#ifndef SHALLOW_H
#define SHALLOW_H

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <string.h>
#include <time.h>

#include <mpi.h>

#if defined(_OPENMP)
#include <omp.h>
#define GET_TIME() (omp_get_wtime()) // wall time
#else
#define GET_TIME() ((double)clock() / CLOCKS_PER_SEC) // cpu time
#endif

/*-------------------*/
/*   Define Macros   */
/*-------------------*/
#define INPUT_DIR "input_data/base_case/"
#define MAX_PATH_LENGTH 512

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define GET(data, i, j) ((data)->vals[(data)->nx * (j) + (i)])
#define SET(data, i, j, val) ((data)->vals[(data)->nx * (j) + (i)] = (val))


// To obtain number of nodes and limits of rank
#define RANK_NX(gdata, rank) ((*gdata)->rank_glob[rank][0].n)
#define RANK_NY(gdata, rank) ((*gdata)->rank_glob[rank][1].n)
#define START_I(gdata, rank) ((*gdata)->rank_glob[rank][0].start)
#define END_I(gdata, rank)   ((*gdata)->rank_glob[rank][0].end)
#define START_J(gdata, rank) ((*gdata)->rank_glob[rank][1].start)
#define END_J(gdata, rank)   ((*gdata)->rank_glob[rank][1].end)


/*-------------------*/
/* Define structures */
/*-------------------*/
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

typedef enum {
  LEFT  = 0,
  RIGHT = 1,
  UP = 2,
  DOWN  = 3,
  NEIGHBOR_NUM = 4,
} neighbour_t;

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

typedef struct {
    data_t *u;
    data_t *v;
    data_t *eta;
    data_t *h;
    data_t *h_interp;
} all_data_t;

typedef struct {
    int nb_process;
    int rank;
    int cart_rank;
    int dims[2];
    int coords[2];
    int neighbors[NEIGHBOR_NUM];
    MPI_Comm cart_comm;
} MPITopology;

typedef struct {
    data_t *gathered_output;
    double *receive_data;
    limit_t **rank_glob;
    int *recv_size;
    int *displacements;
} gather_data_t;



/*---------------------------*/
/* Define functon prototypes */
/*---------------------------*/

/*------ From "shallow_MPI.c" ------*/
void update_velocities(const struct parameters param,
                       all_data_t **all_data,
                       limit_t **rank_glob,
                       int cart_rank,
                       MPI_Comm cart_comm,
                       neighbour_t *direction);

void update_eta(const struct parameters param, 
                all_data_t **all_data,
                gather_data_t *gdata,
                MPITopology *topo,
                neighbour_t *direction);

void boundary_source_condition(int n,
                               int nx, 
                               int ny, 
                               const struct parameters param, 
                               all_data_t **all_data);

double interpolate_data(const data_t *data, 
                        double x, 
                        double y);

void interp_bathy(int nx,
                  int ny, 
                  const struct parameters param,
                  all_data_t *all_data);

void cleanup(all_data_t *all_data, struct parameters *param, int cart_rank, int size, 
             data_t *gathered_output, double *receive_data, limit_t **rank_glob, 
             int *recv_size, int *displacements);



/*------ From "tools.c" ------*/

all_data_t* allocate_all_data();

int read_parameters(struct parameters *param,
                    const char *filename);



void print_parameters(const struct parameters *param);


int read_data(data_t *data, 
              const char *filename);


int write_data(const data_t *data, 
               const char *filename, 
               int step);


int write_data_vtk(data_t **data, 
                   const char *name, 
                   const char *filename,
                   int step);


int write_manifest_vtk(const char *filename,
                       double dt, 
                       int nt, 
                       int sampling_rate);


int init_data(data_t *data, 
              int nx, 
              int ny, 
              double dx, 
              double dy, 
              double val);

double get_value_MPI(data_t *data, 
                     int i, 
                     int j, 
                     gather_data_t *gdata,
                     MPITopology *topo);

double set_value_MPI(data_t *data, 
                     int i, 
                     int j, 
                     gather_data_t *gdata,
                     MPITopology *topo,
                     double val);


void free_data(data_t *data);

#endif // SHALLOW_H