#include "shallow_opti.h"

#undef MPI_STATUSES_IGNORE
#define MPI_STATUSES_IGNORE (MPI_Status *)0 // temporary to get rid of "warning: 'MPI_Waitall' accessing 20 bytes in a region of size 0 [-Wstringop-overflow=]"


/*  ----------- WARNING --------------
    - (i,j) : coords domain (global)
    - (START_I(cart_rank), START_J(cart_rank)) : coords of subdomain (global -> local)
    - (i_rank, j_rank) : relative coords in subdomain (local)

    ex :       0    4    8   12
            0  +----+----+----+
               |    |    |    |
               | P0 | P1 | P2 |
            4  +----+----+----+
               |    |    |    |
               | P3 | P4 | P5 |
            8  +----+----+----+
               |    |    |    |
               | P6 | P7 | P8 |
            12 +----+----+----+

    If (i,j) = (6.7)

    -> cart_rank = 4
    -> (START_I(cart_rank), START_J(cart_rank)) = (4,4)
    -> (i_rank, j_rank) = (6-4, 7-4) = (2,3)
------------------------------------ */ 


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

    // Verification des dimensions
    if (nx <= 0 || ny <= 0) {
        fprintf(stderr, "Error: Invalid dimensions nx=%d, ny=%d\n", nx, ny);
        return;
    }

    // Allocate all buffers
    double *send_left = calloc(ny, sizeof(double));
    double *send_right = calloc(ny, sizeof(double));
    double *send_down = calloc(nx, sizeof(double));
    double *send_up = calloc(nx, sizeof(double));
    
    /*
    double *recv_left = calloc(topo->neighbors[LEFT] != MPI_PROC_NULL ? ny : 1, sizeof(double));
    double *recv_right = calloc(topo->neighbors[RIGHT] != MPI_PROC_NULL ? ny : 1, sizeof(double));
    double *recv_down = calloc(topo->neighbors[DOWN] != MPI_PROC_NULL ? nx : 1, sizeof(double));
    double *recv_up = calloc(topo->neighbors[UP] != MPI_PROC_NULL ? nx : 1, sizeof(double));
    */
    double *recv_left = calloc(ny, sizeof(double));
    double *recv_right = calloc(ny, sizeof(double));
    double *recv_down = calloc(nx, sizeof(double));
    double *recv_up = calloc(nx, sizeof(double));

    // Vérification des allocations
    if (!send_left || !send_right || !send_down || !send_up ||
        !recv_left || !recv_right || !recv_down || !recv_up) {
        fprintf(stderr, "Error: Failed to allocate buffers\n");
        goto cleanup;
    }

    // Prepare send buffers
    for (int j = 0; j < ny; j++) {
        send_left[j] = GET(all_data->u, 0, j);
        send_right[j] = GET(all_data->u, nx, j);  
    }
    for (int i = 0; i < nx; i++) {
        send_down[i] = GET(all_data->v, i, 0);
        send_up[i] = GET(all_data->v, i, ny);  
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

    // Vérification des données avant le kernel
    if (!all_data->eta->vals || !all_data->u->vals || 
        !all_data->v->vals || !all_data->h_interp->vals) {
        fprintf(stderr, "Error: null pointers in update_eta\n");
        goto cleanup;
    }

    // Kernel OpenMP avec mapping corrigé
    #pragma omp target teams distribute parallel for collapse(2) 
    map(to: recv_right[0:ny], recv_left[0:ny], recv_up[0:nx], recv_down[0:nx])
    for (int j = 0; j < ny; j++) {
        for (int i = 0; i < nx; i++) {
            const int idx = j * nx + i;
            double h_ij = all_data->h_interp->vals[idx];
            if (h_ij <= 0) continue;

            double u_i = all_data->u->vals[idx];
            double u_ip1;
            if (i < nx-1) {
                u_ip1 = all_data->u->vals[idx + 1];
            } else {
                u_ip1 = (topo->neighbors[RIGHT] != MPI_PROC_NULL) ? 
                        recv_right[j] : u_i;
            }

            double v_j = all_data->v->vals[idx];
            double v_jp1;
            if (j < ny-1) {
                v_jp1 = all_data->v->vals[idx + nx];
            } else {
                v_jp1 = (topo->neighbors[UP] != MPI_PROC_NULL) ? 
                        recv_up[i] : v_j;
            }

            double du_dx = (u_ip1 - u_i) / param.dx;
            double dv_dy = (v_jp1 - v_j) / param.dy;

            double eta_old = all_data->eta->vals[idx];
            all_data->eta->vals[idx] = eta_old - param.dt * h_ij * (du_dx + dv_dy);
        }
    }

    #pragma omp target exit data map(delete: recv_right[0:ny], recv_left[0:ny], \
                                      recv_up[0:nx], recv_down[0:nx])

    // Wait for all sends to complete
    if (send_count > 0) {
        MPI_Waitall(send_count, request_send, status);
    }

cleanup:
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

    // Allocation des buffers de réception avec taille minimale de 1 pour les cas sans voisins
    /*
    double *recv_left = calloc(topo->neighbors[LEFT] != MPI_PROC_NULL ? ny : 1, sizeof(double));
    double *recv_right = calloc(topo->neighbors[RIGHT] != MPI_PROC_NULL ? ny : 1, sizeof(double));
    double *recv_down = calloc(topo->neighbors[DOWN] != MPI_PROC_NULL ? nx : 1, sizeof(double));
    double *recv_up = calloc(topo->neighbors[UP] != MPI_PROC_NULL ? nx : 1, sizeof(double));
    */
    double *recv_left = calloc(ny, sizeof(double));
    double *recv_right = calloc(ny, sizeof(double));
    double *recv_down = calloc(nx, sizeof(double));
    double *recv_up = calloc(nx, sizeof(double));

    // Vérification des allocations
    if (!send_left || !send_right || !send_down || !send_up ||
        !recv_left || !recv_right || !recv_down || !recv_up) {
        fprintf(stderr, "Error: Failed to allocate buffers\n");
        goto cleanup;
    }

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
    #pragma omp target teams distribute parallel for collapse(2) \
        map(to: recv_right[0:ny], recv_left[0:ny], recv_up[0:nx], recv_down[0:nx])
    for (int j = 0; j < ny; j++) {
        for (int i = 0; i < nx + 1; i++) {
            const int idx = j * nx + i;
            double eta_ij, eta_im1j;

            if (i < nx) {
                eta_ij = all_data->eta->vals[idx];
            } else if (topo->neighbors[RIGHT] != MPI_PROC_NULL) {
                eta_ij = recv_right[j];
            } else {
                eta_ij = all_data->eta->vals[j * nx + (nx-1)];  // Extrapolate at boundary
            }

            if (i > 0) {
                eta_im1j = all_data->eta->vals[idx - 1];
            } else if (topo->neighbors[LEFT] != MPI_PROC_NULL) {
                eta_im1j = recv_left[j];
            } else {
                eta_im1j = eta_ij;  // Extrapolate at boundary
            }

            double u_ij = all_data->u->vals[idx];
            double new_u = (1.0 - c2) * u_ij - c1 / dx * (eta_ij - eta_im1j);
            all_data->u->vals[idx] = new_u;
        }
    }

    // Update v (includes one extra point in y direction)
    #pragma omp target teams distribute parallel for collapse(2) \
        map(to: recv_right[0:ny], recv_left[0:ny], recv_up[0:nx], recv_down[0:nx])
    for (int j = 0; j < ny + 1; j++) {
        for (int i = 0; i < nx; i++) {
            const int idx = j * nx + i;
            double eta_ij, eta_ijm1;

            if (j < ny) {
                eta_ij = all_data->eta->vals[idx];
            } else if (topo->neighbors[UP] != MPI_PROC_NULL) {
                eta_ij = recv_up[i];
            } else {
                eta_ij = all_data->eta->vals[i + (ny-1) * nx];  
            }

            if (j > 0) {
                eta_ijm1 = all_data->eta->vals[idx - nx];
            } else if (topo->neighbors[DOWN] != MPI_PROC_NULL) {
                eta_ijm1 = recv_down[i];
            } else {
                eta_ijm1 = eta_ij; 
            }

            double v_ij = all_data->v->vals[idx];
            double new_v = (1.0 - c2) * v_ij - c1 / dy * (eta_ij - eta_ijm1);
            all_data->v->vals[idx] = new_v;
        }
    }

    // Wait for all sends to complete
    if (send_count > 0) {
        MPI_Waitall(send_count, request_send, status);
    }

cleanup:    
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


    #pragma omp parallel for collapse(2)
    for(int i = 0; i < local_nx; i++) {
        for(int j = 0; j < local_ny; j++) {

            double x = (i + start_i) * param.dx;
            double y = (j + start_j) * param.dy;
            double val = interpolate_data(all_data->h, nx_glob, ny_glob, x, y);
            SET(all_data->h_interp, i, j, val);
        }
    }
}

void boundary_conditions(const parameters_t param, all_data_t *all_data, MPITopology *topo) {
    switch(param.boundary_type) {
        case 1: {  // Reflective boundary conditions (original)
            if (topo->neighbors[LEFT] == MPI_PROC_NULL) {
                for (int j = 0; j < all_data->u->ny; j++) {
                    SET(all_data->u, 0, j, 0.0);
                }
            }
            if (topo->neighbors[RIGHT] == MPI_PROC_NULL) {
                for (int j = 0; j < all_data->u->ny; j++) {
                    SET(all_data->u, all_data->u->nx-1, j, 0.0);
                }
            }
            if (topo->neighbors[DOWN] == MPI_PROC_NULL) {
                for (int i = 0; i < all_data->v->nx; i++) {
                    SET(all_data->v, i, 0, 0.0);
                }
            }
            if (topo->neighbors[UP] == MPI_PROC_NULL) {
                for (int i = 0; i < all_data->v->nx; i++) {
                    SET(all_data->v, i, all_data->v->ny-1, 0.0);
                }
            }
            break;
        }
        
        case 2: {  // Optimized Sommerfeld with absorption control
            const double dt = param.dt;
            const double dx = param.dx;
            const double dy = param.dy;
            const double g = 9.81;
            
            // Coefficient d'absorption: 
            const double absorption_strength = 1.0;
            
            // Fonction inline pour calculer le coefficient de damping
            #define CALC_DAMPING(depth, delta) \
                fmax(0.0, fmin(1.0, 1.0 - absorption_strength * sqrt(g * fmax(0.01, depth)) * dt / delta))
            
            // Les variables nx et ny sont utilisées plusieurs fois
            const int u_nx = all_data->u->nx;
            const int u_ny = all_data->u->ny;
            const int v_nx = all_data->v->nx;
            const int v_ny = all_data->v->ny;
            
            // Left boundary - combine u and eta processing
            if (topo->neighbors[LEFT] == MPI_PROC_NULL) {
                for (int j = 0; j < u_ny; j++) {
                    double local_depth = GET(all_data->h_interp, 1, j);
                    double damping = CALC_DAMPING(local_depth, dx);
                    
                    // Process u
                    SET(all_data->u, 0, j, GET(all_data->u, 1, j) * damping);
                    
                    // Process eta if within bounds
                    if (all_data->eta != NULL && j > 0 && j < all_data->eta->ny - 1) {
                        SET(all_data->eta, 0, j, GET(all_data->eta, 1, j) * damping);
                    }
                }
            }
            
            // Right boundary - combine u and eta processing
            if (topo->neighbors[RIGHT] == MPI_PROC_NULL) {
                for (int j = 0; j < u_ny; j++) {
                    double local_depth = GET(all_data->h_interp, u_nx-2, j);
                    double damping = CALC_DAMPING(local_depth, dx);
                    
                    // Process u
                    SET(all_data->u, u_nx-1, j, GET(all_data->u, u_nx-2, j) * damping);
                    
                    // Process eta if within bounds
                    if (all_data->eta != NULL && j > 0 && j < all_data->eta->ny - 1) {
                        SET(all_data->eta, all_data->eta->nx-1, j, 
                            GET(all_data->eta, all_data->eta->nx-2, j) * damping);
                    }
                }
            }
            
            // Bottom boundary - combine v and eta processing
            if (topo->neighbors[DOWN] == MPI_PROC_NULL) {
                for (int i = 0; i < v_nx; i++) {
                    double local_depth = GET(all_data->h_interp, i, 1);
                    double damping = CALC_DAMPING(local_depth, dy);
                    
                    // Process v
                    SET(all_data->v, i, 0, GET(all_data->v, i, 1) * damping);
                    
                    // Process eta if within bounds
                    if (all_data->eta != NULL && i > 0 && i < all_data->eta->nx - 1) {
                        SET(all_data->eta, i, 0, GET(all_data->eta, i, 1) * damping);
                    }
                }
            }
            
            // Top boundary - combine v and eta processing
            if (topo->neighbors[UP] == MPI_PROC_NULL) {
                for (int i = 0; i < v_nx; i++) {
                    double local_depth = GET(all_data->h_interp, i, v_ny-2);
                    double damping = CALC_DAMPING(local_depth, dy);
                    
                    // Process v
                    SET(all_data->v, i, v_ny-1, GET(all_data->v, i, v_ny-2) * damping);
                    
                    // Process eta if within bounds
                    if (all_data->eta != NULL && i > 0 && i < all_data->eta->nx - 1) {
                        SET(all_data->eta, i, all_data->eta->ny-1, 
                            GET(all_data->eta, i, all_data->eta->ny-2) * damping);
                    }
                }
            }
            
            #undef CALC_DAMPING
            break;
        }
        
        default:
            if (topo->cart_rank == 0) {
                printf("Warning: Unknown boundary type %d\n", param.boundary_type);
            }
            break;
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
    
    // Basic sine wave source
    double source = A * sin(2.0 * M_PI * f * t);
    
    // Gaussian envelope for smooth start
    double t_start = 5.0 / f; // Start time for envelope
    double envelope = 1.0 - exp(-(t/t_start) * (t/t_start));
    source *= envelope;
    
    switch(param.source_type) {
        case 1: {  // Top boundary wave maker
            if (topo->neighbors[UP] == MPI_PROC_NULL) {
                for (int i = 0; i < all_data->v->nx; i++) {
                    SET(all_data->v, i, all_data->v->ny-1, A * sin(2 * M_PI * f * t));
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

int initialize_mpi_topology(int argc, char **argv, MPITopology *topo) {
    int err;
    char error_string[MPI_MAX_ERROR_STRING];
    int length;
    
    int periods[2] = {0, 0};
    int reorder = 1;
    topo->dims[0] = 0;
    topo->dims[1] = 0;
    int local_gpu;
    
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &topo->nb_process);
    MPI_Comm_rank(MPI_COMM_WORLD, &topo->rank);

    // Assigner a GPU to each process MPI
    int num_devices = omp_get_num_devices();
    if (num_devices <= 0) {
        printf("Error: No OpenMP devices found\n");
        return -1;
    }
    local_gpu = topo->rank % num_devices;
    if (omp_set_default_device(local_gpu), local_gpu != omp_get_default_device()) {
        printf("Error setting OpenMP device\n");
        return -1;
    }

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
