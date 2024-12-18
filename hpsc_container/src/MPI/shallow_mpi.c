/*===========================================================
 * SHALLOW WATER EQUATIONS SOLVER - PARALLEL MPI
 * Implementation File
 * Contains core computation routines and parallel algorithms
 ===========================================================*/

#include "shallow_mpi.h"

#undef MPI_STATUSES_IGNORE
#define MPI_STATUSES_IGNORE (MPI_Status *)0 

/*===========================================================
 * DOMAIN DECOMPOSITION AND INDEXING NOTES
 ===========================================================*/
/*  
 * Grid coordinates mapping:
 * - (i,j) : Global domain coordinates
 * - (START_I(cart_rank), START_J(cart_rank)) : Global to local subdomain mapping
 * - (i_rank, j_rank) : Local subdomain coordinates
 * 
 * Example grid decomposition (3x3):
 *     0    4    8   12
 *   0  +----+----+----+
 *      |    |    |    |
 *      | P0 | P1 | P2 |
 *   4  +----+----+----+
 *      |    |    |    |
 *      | P3 | P4 | P5 |
 *   8  +----+----+----+
 *      |    |    |    |
 *      | P6 | P7 | P8 |
 *   12 +----+----+----+
 */

/*===========================================================
 * INITIALIZATION AND CONFIGURATION FUNCTIONS 
 ===========================================================*/
 int initialize_mpi_topology(int argc, char **argv, MPITopology *topo) {
    int err;
    char error_string[MPI_MAX_ERROR_STRING];
    int length;
    
    int periods[2] = {0, 0};
    int reorder = 1;
    topo->dims[0] = 0;
    topo->dims[1] = 0;
    
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &topo->nb_process);
    MPI_Comm_rank(MPI_COMM_WORLD, &topo->rank);
    MPI_Dims_create(topo->nb_process, 2, topo->dims);
    
    if (topo->dims[0] * topo->dims[1] != topo->nb_process) {
        fprintf(stderr, "Process %d: Error dimensions (%d x %d) != nb_process (%d)\n",
                topo->rank, topo->dims[0], topo->dims[1], topo->nb_process);
        fflush(stderr);
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }
    
    MPI_Cart_create(MPI_COMM_WORLD, 2, topo->dims, periods, reorder, &topo->cart_comm);
    MPI_Comm_rank(topo->cart_comm, &topo->cart_rank);
    MPI_Cart_coords(topo->cart_comm, topo->cart_rank, 2, topo->coords);
    MPI_Cart_shift(topo->cart_comm, 0, 1, &topo->neighbors[LEFT], &topo->neighbors[RIGHT]);
    MPI_Cart_shift(topo->cart_comm, 1, 1, &topo->neighbors[DOWN], &topo->neighbors[UP]);
        
    if (topo->rank == 0) {
        printf("Dimensions of the grid : %d x %d\n", topo->dims[0], topo->dims[1]);
        fflush(stdout);
    }
    
    
    return 0;
}

int initialize_gather_structures(const MPITopology *topo, 
                               gather_data_t *gdata,
                               int nx_glob, int ny_glob,
                               double dx, double dy) {
    
    // Initialize structure
    memset(gdata, 0, sizeof(gather_data_t));
    
    // Allocate arrays for gather operation
    gdata->recv_size_eta = calloc(topo->nb_process, sizeof(int));
    gdata->recv_size_u = calloc(topo->nb_process, sizeof(int));
    gdata->recv_size_v = calloc(topo->nb_process, sizeof(int));
    gdata->displacements_eta = calloc(topo->nb_process, sizeof(int));
    gdata->displacements_u = calloc(topo->nb_process, sizeof(int));
    gdata->displacements_v = calloc(topo->nb_process, sizeof(int));
    
    if (!gdata->recv_size_eta || !gdata->recv_size_u || !gdata->recv_size_v ||
        !gdata->displacements_eta || !gdata->displacements_u || !gdata->displacements_v) {
        fprintf(stderr, "Rank %d: Failed to allocate size arrays\n", topo->cart_rank);
        return 1;
    }

    // Allocate rank_glob array
    gdata->rank_glob = calloc(topo->nb_process, sizeof(limit_t*));
    if (!gdata->rank_glob) {
        fprintf(stderr, "Rank %d: Failed to allocate rank_glob\n", topo->cart_rank);
        return 1;
    }

    for (int r = 0; r < topo->nb_process; r++) {
        gdata->rank_glob[r] = calloc(2, sizeof(limit_t));
        if (!gdata->rank_glob[r]) {
            fprintf(stderr, "Rank %d: Failed to allocate rank_glob[%d]\n", topo->cart_rank, r);
            return 1;
        }
    }

    // Calculate base dimensions and remainders
    int base_nx = nx_glob / topo->dims[0];
    int base_ny = ny_glob / topo->dims[1];
    int remainder_x = nx_glob % topo->dims[0];
    int remainder_y = ny_glob % topo->dims[1];

    // Keep track of current position
    int current_x = 0;
    int current_y = 0;

    if (topo->cart_rank == 0) {
        int total_offset = 0;
        
        // Browse processes in the order of the cartesian grid
        for (int j = 0; j < topo->dims[1]; j++) {
            current_x = 0;  // Reset X position for each row
            
            for (int i = 0; i < topo->dims[0]; i++) {
                int coords[2] = {i, j};
                int rank;
                MPI_Cart_rank(topo->cart_comm, coords, &rank);

                // Calculate local dimensions
                int local_nx = base_nx + (i < remainder_x ? 1 : 0);
                int local_ny = base_ny + (j < remainder_y ? 1 : 0);

                // Set start and end positions for X direction
                gdata->rank_glob[rank][0].start = current_x;
                gdata->rank_glob[rank][0].n = local_nx;
                gdata->rank_glob[rank][0].end = current_x + local_nx;
                current_x += local_nx;

                // Set start and end positions for Y direction
                gdata->rank_glob[rank][1].start = current_y;
                gdata->rank_glob[rank][1].n = local_ny;
                gdata->rank_glob[rank][1].end = current_y + local_ny;

                // Calculate receive sizes and displacements
                gdata->recv_size_eta[rank] = local_nx * local_ny;
                gdata->recv_size_u[rank] = (local_nx + 1) * local_ny;
                gdata->recv_size_v[rank] = local_nx * (local_ny + 1);
                
                gdata->displacements_eta[rank] = total_offset;
                gdata->displacements_u[rank] = total_offset;
                gdata->displacements_v[rank] = total_offset;
                
                total_offset += gdata->recv_size_eta[rank];
            }
            current_y += base_ny + (j < remainder_y ? 1 : 0);  // Update Y position after each row
        }

        // Allocate reception buffers for rank 0
        gdata->receive_data_eta = calloc(nx_glob * ny_glob, sizeof(double));
        gdata->receive_data_u = calloc((nx_glob + 1) * ny_glob, sizeof(double));
        gdata->receive_data_v = calloc(nx_glob * (ny_glob + 1), sizeof(double));
        gdata->gathered_output = calloc(3, sizeof(data_t));
        
        if (!gdata->receive_data_eta || !gdata->receive_data_u || !gdata->receive_data_v || 
            !gdata->gathered_output) {
            return 1;
        }

        // Initialize gathered_output structures
        for (int i = 0; i < 3; i++) {
            gdata->gathered_output[i].nx = nx_glob;
            gdata->gathered_output[i].ny = ny_glob;
            gdata->gathered_output[i].dx = dx;
            gdata->gathered_output[i].dy = dy;
            gdata->gathered_output[i].vals = calloc(nx_glob * ny_glob, sizeof(double));
            if (!gdata->gathered_output[i].vals) return 1;
        }
    }

    // Synchronize all processes
    MPI_Barrier(topo->cart_comm);
    
    // Broadcast all necessary information
    MPI_Bcast(gdata->recv_size_eta, topo->nb_process, MPI_INT, 0, topo->cart_comm);
    MPI_Bcast(gdata->recv_size_u, topo->nb_process, MPI_INT, 0, topo->cart_comm);
    MPI_Bcast(gdata->recv_size_v, topo->nb_process, MPI_INT, 0, topo->cart_comm);
    MPI_Bcast(gdata->displacements_eta, topo->nb_process, MPI_INT, 0, topo->cart_comm);
    MPI_Bcast(gdata->displacements_u, topo->nb_process, MPI_INT, 0, topo->cart_comm);
    MPI_Bcast(gdata->displacements_v, topo->nb_process, MPI_INT, 0, topo->cart_comm);

    for (int r = 0; r < topo->nb_process; r++) {
        MPI_Bcast(&(gdata->rank_glob[r][0]), sizeof(limit_t), MPI_BYTE, 0, topo->cart_comm);
        MPI_Bcast(&(gdata->rank_glob[r][1]), sizeof(limit_t), MPI_BYTE, 0, topo->cart_comm);
    }

    MPI_Barrier(topo->cart_comm);
    return 0;
}

/*===========================================================
 * INTERPOLATION AND PREPROCESSING FUNCTIONS 
 ===========================================================*/
double interpolate_data(const data_t *data,
                        int nx_glob, int ny_glob,
                        double x, 
                        double y) {

   int i = (int)(x / data->dx);
   int j = (int)(y / data->dy);

   // Boundary cases
   if (i < 0 || j < 0 || i >= nx_glob - 1 || j >= ny_glob - 1) {
       i = (i < 0) ? 0 : (i >= nx_glob) ? nx_glob - 1 : i;
       j = (j < 0) ? 0 : (j >= ny_glob) ? ny_glob - 1 : j;
       return GET(data, i, j);
   }

   // Four positions surrounding (x,y)
   double x1 = i * data->dx;
   double x2 = (i + 1) * data->dx;
   double y1 = j * data->dy;
   double y2 = (j + 1) * data->dy;

   // Four vals of data surrounding (i,j)
   double Q11 = GET(data, i, j);
   double Q12 = GET(data, i, j + 1);
   double Q21 = GET(data, i + 1, j);
   double Q22 = GET(data, i + 1, j + 1);

   // Weighted coef
   double wx = (x2 - x) / (x2 - x1);
   double wy = (y2 - y) / (y2 - y1);

   // interpolated value
   double val = wx * wy * Q11 +
               (1 - wx) * wy * Q21 +
               wx * (1 - wy) * Q12 +
               (1 - wx) * (1 - wy) * Q22;

   return val;
}

void interp_bathy(const parameters_t param,
                  int nx_glob, int ny_glob,
                  all_data_t *all_data,
                  gather_data_t *gdata, 
                  MPITopology *topo) {
    

    int start_i = START_I(gdata, topo->cart_rank);
    int start_j = START_J(gdata, topo->cart_rank);
    int local_nx = all_data->h_interp->nx;
    int local_ny = all_data->h_interp->ny;

    for(int i = 0; i < local_nx; i++) {
        for(int j = 0; j < local_ny; j++) {

            double x = (i + start_i) * param.dx;
            double y = (j + start_j) * param.dy;
            double val = interpolate_data(all_data->h, nx_glob, ny_glob, x, y);
            SET(all_data->h_interp, i, j, val);
        }
    }
}

/*===========================================================
 * CORE COMPUTATION FUNCTIONS
 ===========================================================*/
void update_eta(const parameters_t param, 
                all_data_t *all_data,
                gather_data_t *gdata,
                MPITopology *topo) {

    // Create separate request arrays for sends and receives
    MPI_Request request_recv[4] = {MPI_REQUEST_NULL, MPI_REQUEST_NULL, 
                                  MPI_REQUEST_NULL, MPI_REQUEST_NULL};
    MPI_Request request_send[4] = {MPI_REQUEST_NULL, MPI_REQUEST_NULL,
                                  MPI_REQUEST_NULL, MPI_REQUEST_NULL};
    MPI_Status status[4];

    int nx = all_data->eta->nx;
    int ny = all_data->eta->ny;

    // Allocate all buffers
    double *send_left = calloc(ny, sizeof(double));
    double *send_right = calloc(ny, sizeof(double));
    double *send_down = calloc(nx, sizeof(double));
    double *send_up = calloc(nx, sizeof(double));
    
    double *recv_left = NULL;
    double *recv_right = NULL;
    double *recv_down = NULL;
    double *recv_up = NULL;

    if (topo->neighbors[LEFT] != MPI_PROC_NULL) recv_left = calloc(ny, sizeof(double));
    if (topo->neighbors[RIGHT] != MPI_PROC_NULL) recv_right = calloc(ny, sizeof(double));
    if (topo->neighbors[DOWN] != MPI_PROC_NULL) recv_down = calloc(nx, sizeof(double));
    if (topo->neighbors[UP] != MPI_PROC_NULL) recv_up = calloc(nx, sizeof(double));

    // Prepare send buffers
    for (int j = 0; j < ny; j++) {
        send_left[j] = GET(all_data->u, 0, j);
        send_right[j] = GET(all_data->u, nx, j);  // Note: u has nx+1 points
    }
    for (int i = 0; i < nx; i++) {
        send_down[i] = GET(all_data->v, i, 0);
        send_up[i] = GET(all_data->v, i, ny);  // Note: v has ny+1 points
    }

    // Post all receives first
    int recv_count = 0;
    if (topo->neighbors[LEFT] != MPI_PROC_NULL)
        MPI_Irecv(recv_left, ny, MPI_DOUBLE, topo->neighbors[LEFT], 100,
                  topo->cart_comm, &request_recv[recv_count++]);
    if (topo->neighbors[RIGHT] != MPI_PROC_NULL)
        MPI_Irecv(recv_right, ny, MPI_DOUBLE, topo->neighbors[RIGHT], 101,
                  topo->cart_comm, &request_recv[recv_count++]);
    if (topo->neighbors[DOWN] != MPI_PROC_NULL)
        MPI_Irecv(recv_down, nx, MPI_DOUBLE, topo->neighbors[DOWN], 102,
                  topo->cart_comm, &request_recv[recv_count++]);
    if (topo->neighbors[UP] != MPI_PROC_NULL)
        MPI_Irecv(recv_up, nx, MPI_DOUBLE, topo->neighbors[UP], 103,
                  topo->cart_comm, &request_recv[recv_count++]);

    // Post all sends
    int send_count = 0;
    if (topo->neighbors[RIGHT] != MPI_PROC_NULL)
        MPI_Isend(send_right, ny, MPI_DOUBLE, topo->neighbors[RIGHT], 100,
                  topo->cart_comm, &request_send[send_count++]);
    if (topo->neighbors[LEFT] != MPI_PROC_NULL)
        MPI_Isend(send_left, ny, MPI_DOUBLE, topo->neighbors[LEFT], 101,
                  topo->cart_comm, &request_send[send_count++]);
    if (topo->neighbors[UP] != MPI_PROC_NULL)
        MPI_Isend(send_up, nx, MPI_DOUBLE, topo->neighbors[UP], 102,
                  topo->cart_comm, &request_send[send_count++]);
    if (topo->neighbors[DOWN] != MPI_PROC_NULL)
        MPI_Isend(send_down, nx, MPI_DOUBLE, topo->neighbors[DOWN], 103,
                  topo->cart_comm, &request_send[send_count++]);

    // Wait for all receives to complete
    if (recv_count > 0) {
        MPI_Waitall(recv_count, request_recv, status);
    }

    // Update eta
    for (int j = 0; j < ny; j++) {
        for (int i = 0; i < nx; i++) {
            // Get h values at the required positions
            double h_ui_ip1_j = GET(all_data->h_interp, i + 1, j);    // h at (i+1, j) for u
            double h_ui_j = GET(all_data->h_interp, i, j);            // h at (i, j) for u
            double h_vi_jp1 = GET(all_data->h_interp, i, j + 1);      // h at (i, j+1) for v
            double h_vi_j = GET(all_data->h_interp, i, j);            // h at (i, j) for v

            // Get u values with boundary handling
            double u_i = GET(all_data->u, i, j);
            double u_ip1 = (i < nx - 1) ? GET(all_data->u, i + 1, j)
                                        : (topo->neighbors[RIGHT] != MPI_PROC_NULL) ? recv_right[j] : u_i;

            // Get v values with boundary handling
            double v_j = GET(all_data->v, i, j);
            double v_jp1 = (j < ny - 1) ? GET(all_data->v, i, j + 1)
                                        : (topo->neighbors[UP] != MPI_PROC_NULL) ? recv_up[i] : v_j;

            // Calculate spatial derivatives with the correct h values
            double du_dx = (h_ui_ip1_j * u_ip1 - h_ui_j * u_i) / param.dx;
            double dv_dy = (h_vi_jp1 * v_jp1 - h_vi_j * v_j) / param.dy;

            double eta_old = GET(all_data->eta, i, j);
            double eta_new = eta_old - param.dt * (du_dx + dv_dy);
            SET(all_data->eta, i, j, eta_new);
        }
    }

    // Wait for all sends to complete
    if (send_count > 0) {
        MPI_Waitall(send_count, request_send, status);
    }

    // Clean up
    free(send_left);
    free(send_right);
    free(send_down);
    free(send_up);
    if (recv_left) free(recv_left);
    if (recv_right) free(recv_right);
    if (recv_down) free(recv_down);
    if (recv_up) free(recv_up);
}

void update_velocities(const parameters_t param,
                      all_data_t *all_data,
                      gather_data_t *gdata,
                      MPITopology *topo) {

    MPI_Request request_recv[4] = {MPI_REQUEST_NULL, MPI_REQUEST_NULL,
                                  MPI_REQUEST_NULL, MPI_REQUEST_NULL};
    MPI_Request request_send[4] = {MPI_REQUEST_NULL, MPI_REQUEST_NULL,
                                  MPI_REQUEST_NULL, MPI_REQUEST_NULL};
    MPI_Status status[4];

    int nx = all_data->eta->nx;
    int ny = all_data->eta->ny;
    
    // Allocate all buffers
    double *send_left = calloc(ny, sizeof(double));
    double *send_right = calloc(ny, sizeof(double));
    double *send_down = calloc(nx, sizeof(double));
    double *send_up = calloc(nx, sizeof(double));
    
    double *recv_left = NULL;
    double *recv_right = NULL;
    double *recv_down = NULL;
    double *recv_up = NULL;

    if (topo->neighbors[LEFT] != MPI_PROC_NULL) recv_left = calloc(ny, sizeof(double));
    if (topo->neighbors[RIGHT] != MPI_PROC_NULL) recv_right = calloc(ny, sizeof(double));
    if (topo->neighbors[DOWN] != MPI_PROC_NULL) recv_down = calloc(nx, sizeof(double));
    if (topo->neighbors[UP] != MPI_PROC_NULL) recv_up = calloc(nx, sizeof(double));

    // Prepare send buffers with eta values
    for (int j = 0; j < ny; j++) {
        send_left[j] = GET(all_data->eta, 0, j);
        send_right[j] = GET(all_data->eta, nx-1, j);
    }
    for (int i = 0; i < nx; i++) {
        send_down[i] = GET(all_data->eta, i, 0);
        send_up[i] = GET(all_data->eta, i, ny-1);
    }

    // Post all receives first
    int recv_count = 0;
    if (topo->neighbors[LEFT] != MPI_PROC_NULL)
        MPI_Irecv(recv_left, ny, MPI_DOUBLE, topo->neighbors[LEFT], 200,
                  topo->cart_comm, &request_recv[recv_count++]);
    if (topo->neighbors[RIGHT] != MPI_PROC_NULL)
        MPI_Irecv(recv_right, ny, MPI_DOUBLE, topo->neighbors[RIGHT], 201,
                  topo->cart_comm, &request_recv[recv_count++]);
    if (topo->neighbors[DOWN] != MPI_PROC_NULL)
        MPI_Irecv(recv_down, nx, MPI_DOUBLE, topo->neighbors[DOWN], 202,
                  topo->cart_comm, &request_recv[recv_count++]);
    if (topo->neighbors[UP] != MPI_PROC_NULL)
        MPI_Irecv(recv_up, nx, MPI_DOUBLE, topo->neighbors[UP], 203,
                  topo->cart_comm, &request_recv[recv_count++]);

    // Post all sends
    int send_count = 0;
    if (topo->neighbors[RIGHT] != MPI_PROC_NULL)
        MPI_Isend(send_right, ny, MPI_DOUBLE, topo->neighbors[RIGHT], 200,
                  topo->cart_comm, &request_send[send_count++]);
    if (topo->neighbors[LEFT] != MPI_PROC_NULL)
        MPI_Isend(send_left, ny, MPI_DOUBLE, topo->neighbors[LEFT], 201,
                  topo->cart_comm, &request_send[send_count++]);
    if (topo->neighbors[UP] != MPI_PROC_NULL)
        MPI_Isend(send_up, nx, MPI_DOUBLE, topo->neighbors[UP], 202,
                  topo->cart_comm, &request_send[send_count++]);
    if (topo->neighbors[DOWN] != MPI_PROC_NULL)
        MPI_Isend(send_down, nx, MPI_DOUBLE, topo->neighbors[DOWN], 203,
                  topo->cart_comm, &request_send[send_count++]);

    // Wait for all receives to complete before updating
    if (recv_count > 0) {
        MPI_Waitall(recv_count, request_recv, status);
    }
    
    // Update velocities
    double dx = param.dx;
    double dy = param.dy;
    double c1 = param.dt * param.g;
    double c2 = param.dt * param.gamma;

    // Update u (includes one extra point in x direction)
    for (int j = 0; j < ny; j++) {
        for (int i = 0; i < nx + 1; i++) {
            double eta_ij;
            double eta_im1j;

            if (i < nx) {
                eta_ij = GET(all_data->eta, i, j);
            } else if (topo->neighbors[RIGHT] != MPI_PROC_NULL) {
                eta_ij = recv_right[j];
            } else {
                eta_ij = GET(all_data->eta, nx-1, j);  // Extrapolate at boundary
            }

            if (i > 0) {
                eta_im1j = GET(all_data->eta, i-1, j);
            } else if (topo->neighbors[LEFT] != MPI_PROC_NULL) {
                eta_im1j = recv_left[j];
            } else {
                eta_im1j = eta_ij;  // Extrapolate at boundary
            }

            double u_ij = GET(all_data->u, i, j);
            double new_u = (1.0 - c2) * u_ij - c1 / dx * (eta_ij - eta_im1j);
            SET(all_data->u, i, j, new_u);
        }
    }

    // Update v (includes one extra point in y direction)
    for (int j = 0; j < ny + 1; j++) {
        for (int i = 0; i < nx; i++) {
            double eta_ij;
            double eta_ijm1;

            if (j < ny) {
                eta_ij = GET(all_data->eta, i, j);
            } else if (topo->neighbors[UP] != MPI_PROC_NULL) {
                eta_ij = recv_up[i];
            } else {
                eta_ij = GET(all_data->eta, i, ny-1);  // Extrapolate at boundary
            }

            if (j > 0) {
                eta_ijm1 = GET(all_data->eta, i, j-1);
            } else if (topo->neighbors[DOWN] != MPI_PROC_NULL) {
                eta_ijm1 = recv_down[i];
            } else {
                eta_ijm1 = eta_ij;  // Extrapolate at boundary
            }

            double v_ij = GET(all_data->v, i, j);
            double new_v = (1.0 - c2) * v_ij - c1 / dy * (eta_ij - eta_ijm1);
            SET(all_data->v, i, j, new_v);
        }
    }

    // Wait for all sends to complete
    if (send_count > 0) {
        MPI_Waitall(send_count, request_send, status);
    }
    
    // Clean up
    free(send_left);
    free(send_right);
    free(send_down);
    free(send_up);
    if (recv_left) free(recv_left);
    if (recv_right) free(recv_right);
    if (recv_down) free(recv_down);
    if (recv_up) free(recv_up);
}


/*===========================================================
 * BOUNDARY CONDITIONS AND SOURCE TERMS
 ===========================================================*/
void boundary_conditions(const parameters_t param, all_data_t *all_data, MPITopology *topo) {

    #pragma omp parallel for
    for (int j = 0; j < all_data->u->ny; j++) {
        if (topo->neighbors[LEFT] == MPI_PROC_NULL) {
            SET(all_data->u, 0, j, 0.0);
        }
        if (topo->neighbors[RIGHT] == MPI_PROC_NULL) {
            SET(all_data->u, all_data->u->nx - 1, j, 0.0);
        }
    }

    #pragma omp parallel for
    for (int i = 0; i < all_data->v->nx; i++) {
        if (topo->neighbors[DOWN] == MPI_PROC_NULL) {
            SET(all_data->v, i, 0, 0.0);
        }
        if (topo->neighbors[UP] == MPI_PROC_NULL) {
            SET(all_data->v, i, all_data->v->ny - 1, 0.0);
        }
    }
}

void apply_source(int timestep, int nx_glob, int ny_glob,
                  const parameters_t param,
                  all_data_t *all_data,
                  gather_data_t *gdata,
                  MPITopology *topo) {
    
    double t = timestep * param.dt;
    const double A = 5.0;        
    const double f = 1.0 / 20.0; 
    
    double source = A * sin(2.0 * M_PI * f * t);
    
    double t_start = 5.0 / f; 
    double envelope = 1.0 - exp(-(t/t_start) * (t/t_start));
    source *= envelope;
    
    switch(param.source_type) {
        case 1: {  // Top boundary wave maker
            if (topo->neighbors[UP] == MPI_PROC_NULL) {
                for (int i = 0; i < all_data->v->nx; i++) {

                    double x_pos = (START_I(gdata, topo->cart_rank) + i) * param.dx;
                    double spatial_mod = sin(2.0 * M_PI * x_pos / (nx_glob * param.dx) * 2);
                    SET(all_data->v, i, all_data->v->ny-1, source * (1.0 + 0.3 * spatial_mod));
                }
            }
            break;
        }
        
        case 2: {  // Point source with circular waves
            int global_middle_i = nx_glob / 2;
            int global_middle_j = ny_glob / 2;
            
            int local_i = global_middle_i - START_I(gdata, topo->cart_rank);
            int local_j = global_middle_j - START_J(gdata, topo->cart_rank);
            
            if (local_i >= 0 && local_i < all_data->eta->nx &&
                local_j >= 0 && local_j < all_data->eta->ny) {
                SET(all_data->eta, local_i, local_j, source);
            }
            break;
        }
        
        case 3: {  // Multiple point sources
            const int num_sources = 3;
            int source_positions[3][2] = {
                {nx_glob/4, ny_glob/4},
                {nx_glob/2, ny_glob/2},
                {3*nx_glob/4, 3*ny_glob/4}
            };
            double phase_shifts[3] = {0.0, 2.0*M_PI/3.0, 4.0*M_PI/3.0};
            
            for (int s = 0; s < num_sources; s++) {
                int local_i = source_positions[s][0] - START_I(gdata, topo->cart_rank);
                int local_j = source_positions[s][1] - START_J(gdata, topo->cart_rank);
                
                if (local_i >= 0 && local_i < all_data->eta->nx &&
                    local_j >= 0 && local_j < all_data->eta->ny) {
                    double phase_shifted_source = A * sin(2.0 * M_PI * f * t + phase_shifts[s]) * envelope;
                    SET(all_data->eta, local_i, local_j, phase_shifted_source);
                }
            }
            break;
        }
        
        case 4: {  // Moving source
            double speed = 0.004; 
            int source_i = (int)(nx_glob/4 + (nx_glob/2) * sin(speed * t));
            int source_j = (int)(ny_glob/2 + (ny_glob/4) * cos(speed * t));
            
            int local_i = source_i - START_I(gdata, topo->cart_rank);
            int local_j = source_j - START_J(gdata, topo->cart_rank);
            
            if (local_i >= 0 && local_i < all_data->eta->nx &&
                local_j >= 0 && local_j < all_data->eta->ny) {
                SET(all_data->eta, local_i, local_j, source);
            }
            break;
        }

        default:
            if (topo->cart_rank == 0) {
                printf("Warning: Unknown source type %d\n", param.source_type);
            }
            break;
    }
}


/*===========================================================
 * DATA COLLECTION AND OUTPUT FUNCTIONS
 ===========================================================*/
void gather_and_assemble_data(const parameters_t param,
                             all_data_t *all_data,
                             gather_data_t *gdata,
                             MPITopology *topo,
                             int nx_glob, int ny_glob,
                             int timestep) {
    
    data_t *output_data[] = {all_data->eta, all_data->u, all_data->v};
    double *receive_buffers[] = {gdata->receive_data_eta,
                               gdata->receive_data_u,
                               gdata->receive_data_v};
    int *recv_sizes[] = {gdata->recv_size_eta,
                        gdata->recv_size_u,
                        gdata->recv_size_v};
    int *displacements[] = {gdata->displacements_eta,
                          gdata->displacements_u,
                          gdata->displacements_v};

    // Gather and assemble each field
    for (int field = 0; field < 3; field++) {
        data_t *local_data = output_data[field];
        int send_size = local_data->nx * local_data->ny;

        // Gathering data
        MPI_Gatherv(local_data->vals, send_size, MPI_DOUBLE,
                    receive_buffers[field], recv_sizes[field],
                    displacements[field], MPI_DOUBLE, 0, topo->cart_comm);

        // Assembly on rank 0
        if (topo->cart_rank == 0) {
            // Reset output array
            memset(gdata->gathered_output[field].vals, 0,
                  nx_glob * ny_glob * sizeof(double));

            // Assemble data from each process
            for (int r = 0; r < topo->nb_process; r++) {
                int local_nx = RANK_NX(gdata, r);
                int local_ny = RANK_NY(gdata, r);
                int global_start_i = START_I(gdata, r);
                int global_start_j = START_J(gdata, r);

                // Copy with verification
                for (int j = 0; j < local_ny; j++) {
                    for (int i = 0; i < local_nx; i++) {
                        int global_i = global_start_i + i;
                        int global_j = global_start_j + j;
                        int src_idx = j * local_nx + i;
                        int src_with_displacement = displacements[field][r] + src_idx;
                        int dst_idx = global_j * nx_glob + global_i;

                        double val = receive_buffers[field][src_with_displacement];
                        gdata->gathered_output[field].vals[dst_idx] = val;
                    }
                }
            }
            
            // Update metadata
            gdata->gathered_output[field].nx = nx_glob;
            gdata->gathered_output[field].ny = ny_glob;
            gdata->gathered_output[field].dx = param.dx;
            gdata->gathered_output[field].dy = param.dy;
        }
    }
}