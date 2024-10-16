#include "shallow_MPI.h"

#undef MPI_STATUSES_IGNORE
#define MPI_STATUSES_IGNORE (MPI_Status *)0 // temporary to get rid of "warning: 'MPI_Waitall' accessing 20 bytes in a region of size 0 [-Wstringop-overflow=]"

void cleanup(all_data_t *all_data, struct parameters *param, int cart_rank, int size, 
             data_t *gathered_output, double *receive_data, limit_t **rank_glob, 
             int *recv_size, int *displacements) {
    
    // Free all_data structure
    if (all_data != NULL) {
        if (all_data->u != NULL) {
            free(all_data->u->vals);
            if (all_data->u->edge_vals != NULL) {
                for (int i = 0; i < NEIGHBOR_NUM; i++) {
                    free(all_data->u->edge_vals[i]);
                }
                free(all_data->u->edge_vals);
            }
            free(all_data->u);
        }
        if (all_data->v != NULL) {
            free(all_data->v->vals);
            if (all_data->v->edge_vals != NULL) {
                for (int i = 0; i < NEIGHBOR_NUM; i++) {
                    free(all_data->v->edge_vals[i]);
                }
                free(all_data->v->edge_vals);
            }
            free(all_data->v);
        }
        if (all_data->eta != NULL) {
            free(all_data->eta->vals);
            if (all_data->eta->edge_vals != NULL) {
                for (int i = 0; i < NEIGHBOR_NUM; i++) {
                    free(all_data->eta->edge_vals[i]);
                }
                free(all_data->eta->edge_vals);
            }
            free(all_data->eta);
        }
        if (all_data->h != NULL) {
            free(all_data->h->vals);
            free(all_data->h);
        }
        if (all_data->h_interp != NULL) {
            free(all_data->h_interp->vals);
            free(all_data->h_interp);
        }
        free(all_data);
    }

    // Free parameters (if dynamically allocated)
    // Note: In your current structure, param doesn't need to be freed as it's not dynamically allocated

    // Free MPI-related allocations (only for rank 0)
    if (cart_rank == 0) {
        if (gathered_output != NULL) {
            free(gathered_output->vals);
            free(gathered_output);
        }
        free(receive_data);
        if (rank_glob != NULL) {
            for (int r = 0; r < size; r++) {
                free(rank_glob[r]);
            }
            free(rank_glob);
        }
        free(recv_size);
        free(displacements);
    }
}




double get_value_MPI(data_t *data, 
                     int i, 
                     int j, 
                     limit_t **rank_glob,
                     int cart_rank,
                     MPI_Comm cart_comm) {

  /*  ----------- WARNING --------------
    - (i,j) : coords domain (global)
    - (START_I(cart_rank), START_J(cart_rank)) : coords of subdomain (global -> local)
    - (i_rank, j_rank) : relative coords in subdomain (local)

    ex :     0    4    8   12
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

  int i_rank = i - START_I(cart_rank); 
  int j_rank = j - START_J(cart_rank); 
  int nx = RANK_NX(cart_rank);
  int ny = RANK_NY(cart_rank); 

  // (i_rank, j_rank) inside local domain
  if(i_rank >= 0 && i_rank < nx  && j_rank >= 0 && j_rank < ny )
    return data->vals[j_rank*nx + i_rank];
  
  // (i_rank, j_rank) on horizontal edge (up/down)
  else if(i_rank >= 0 && i_rank < nx) {
    if (j_rank == -1) return data->edge_vals[DOWN][j_rank*nx + i_rank];
    else return data->edge_vals[UP][j_rank*nx + i_rank];
  }

  // (i_rank, j_rank) on vertical edge (left/right)
  else if(j_rank >= 0 && j_rank < ny) {
    if (i_rank == -1) return data->edge_vals[LEFT][j_rank];
    else return data->edge_vals[RIGHT][j_rank];
  }
  else{ printf("Error while accessing subdomain"); MPI_Abort(cart_comm, MPI_ERR_ACCESS); }

}

double set_value_MPI(data_t *data, 
                     int i, 
                     int j, 
                     limit_t **rank_glob,
                     int cart_rank,
                     MPI_Comm cart_comm,
                     double val) {

  /*  ----------- WARNING --------------
    - (i,j) : coords domain (global)
    - (START_I(cart_rank), START_J(cart_rank)) : coords of subdomain (global -> local)
    - (i_rank, j_rank) : relative coords in subdomain (local)

    ex :     0    4    8   12
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

  int i_rank = i - START_I(cart_rank); 
  int j_rank = j - START_J(cart_rank); 
  int nx = RANK_NX(cart_rank);
  int ny = RANK_NY(cart_rank); 

  // (i_rank, j_rank) inside local domain
  if(i_rank >= 0 && i_rank < nx  && j_rank >= 0 && j_rank < ny )
    data->vals[j_rank*nx + i_rank]= val;
  
  // (i_rank, j_rank) on horizontal edge (up/down)
  else if(i_rank >= 0 && i_rank < nx) {
    if (j_rank == -1) data->edge_vals[DOWN][j_rank*nx + i_rank]= val;
    else data->edge_vals[UP][j_rank*nx + i_rank] = val;
  }

  // (i_rank, j_rank) on vertical edge (left/right)
  else if(j_rank >= 0 && j_rank < ny) {
    if (i_rank == -1) data->edge_vals[LEFT][j_rank] = val;
    else data->edge_vals[RIGHT][j_rank] = val;
  }
  else{ printf("Error while accessing subdomain"); MPI_Abort(cart_comm, MPI_ERR_ACCESS); }

}

void update_eta(const struct parameters param, 
                all_data_t **all_data,
                limit_t **rank_glob,
                int cart_rank,
                MPI_Comm cart_comm,
                neighbour_t *direction) {

  MPI_Request request_recv[2];
  MPI_Request request_send[2];

  // Find process size (nx,ny) and every (i,j) within
  int nx = RANK_NX(cart_rank);
  int i_start = START_I(cart_rank);
  int i_end = END_I(cart_rank);

  int ny = RANK_NY(cart_rank);
  int j_start = START_J(cart_rank);
  int j_end = END_J(cart_rank);

  // Receive u(i+1,j) and v(i,j+1) from neighbouring processes
  MPI_Irecv((*all_data)->u->edge_vals[RIGHT], ny, MPI_DOUBLE, direction[RIGHT], 100, cart_comm, &request_recv[0]);
  MPI_Irecv((*all_data)->v->edge_vals[UP], nx, MPI_DOUBLE, direction[UP], 101, cart_comm, &request_recv[1]);

  // Prepare data to send
  double *send_u = malloc(nx * sizeof(double));;
  double *send_v =  malloc(ny * sizeof(double));;

  for (int j_rank = 0; j_rank < ny; j_rank++) 
    send_u[j_rank] = get_value_MPI((*all_data)->u, i_end, j_start+j_rank, rank_glob, cart_rank, cart_comm);

  for (int i_rank = 0; i_rank < nx; i_rank++) 
    send_v[i_rank] = get_value_MPI((*all_data)->u, i_start+i_rank, j_end, rank_glob, cart_rank, cart_comm);

  // Send u(i-1,j) and v(i,j-1) to neighbouring processes
  MPI_Isend(send_u, ny, MPI_DOUBLE, direction[LEFT], 100, cart_comm, &request_recv[0]);
  MPI_Isend(send_v, nx, MPI_DOUBLE, direction[DOWN], 101, cart_comm, &request_recv[0]);

  free(send_u);
  free(send_v);

  // Updating (inside domain)
  double dx = param.dx;
  double dy = param.dy;
  
  for (int i_rank = i_start+1; i_rank <= i_end; i_rank++) {
    for (int j_rank = j_start+1; j_rank <= j_end; j_rank++) {

      int i_local = i_rank - i_start;
      int j_local = j_rank - j_start;

      double h_ij = GET((*all_data)->h_interp, i_local, j_local);
      double c1 = param.dt * h_ij;
      
      double du_dx = (GET((*all_data)->u, i_local + 1, j_local) - GET((*all_data)->u, i_local, j_local))/dx;
      double dv_dy = (GET((*all_data)->v, i_local, j_local + 1) - GET((*all_data)->v, i_local, j_local))/dy;
      
      double eta_ij = GET((*all_data)->eta, i_local, j_local) - c1 * (du_dx + dv_dy);
      
      SET((*all_data)->eta, i_local, j_local, eta_ij);
    }
  }

  // Wait for received data
  MPI_Waitall(2, request_recv, MPI_STATUSES_IGNORE);

  // Update (right boundary)
  int i_rank = i_end;
  for (int j_rank = j_start+1; j_rank < j_end; j_rank++) {
      int i_local = i_rank - i_start;
      int j_local = j_rank - j_start;
      
      double h_ij = GET((*all_data)->h_interp, i_local, j_local);
      double c1 = param.dt * h_ij;
      
      double du_dx = ((*all_data)->u->edge_vals[RIGHT][j_local] - GET((*all_data)->u, i_local, j_local)) / dx;
      double dv_dy = (GET((*all_data)->v, i_local, j_local + 1) - GET((*all_data)->v, i_local, j_local)) / dy;
      
      double eta_ij = GET((*all_data)->eta, i_local, j_local) - c1 * (du_dx + dv_dy);
      
      SET((*all_data)->eta, i_local, j_local, eta_ij);
  }

  // Update (top boundary)
  int j_rank = j_end;
  for (int i_rank = i_start+1; i_rank < i_end; i_rank++) {
      int i_local = i_rank - i_start;
      int j_local = j_rank - j_start;
      
      double h_ij = GET((*all_data)->h_interp, i_local, j_local);
      double c1 = param.dt * h_ij;
      
      double du_dx = (GET((*all_data)->u, i_local + 1, j_local) - GET((*all_data)->u, i_local, j_local)) / dx;
      double dv_dy = ((*all_data)->v->edge_vals[UP][i_local] - GET((*all_data)->v, i_local, j_local)) / dy;
      
      double eta_ij = GET((*all_data)->eta, i_local, j_local) - c1 * (du_dx + dv_dy);
      
      SET((*all_data)->eta, i_local, j_local, eta_ij);
  }

  // Wait for sent data to complete
  MPI_Waitall(2, request_send, MPI_STATUSES_IGNORE);

  free(send_u);
  free(send_v);


}

void update_velocities(const struct parameters param,
                       all_data_t **all_data,
                       limit_t **rank_glob,
                       int cart_rank,
                       MPI_Comm cart_comm,
                       neighbour_t *direction) {

    MPI_Request request_recv[2];
    MPI_Request request_send[2];

    // Find process size (nx,ny) and every (i,j) within
    int nx = RANK_NX(cart_rank);
    int i_start = START_I(cart_rank);
    int i_end = END_I(cart_rank);
    int ny = RANK_NY(cart_rank);
    int j_start = START_J(cart_rank);
    int j_end = END_J(cart_rank);

    // Receive eta(i-1,j) and eta(i,j-1) from neighbouring processes
    MPI_Irecv((*all_data)->eta->edge_vals[LEFT], ny, MPI_DOUBLE, direction[LEFT], 200, cart_comm, &request_recv[0]);
    MPI_Irecv((*all_data)->eta->edge_vals[DOWN], nx, MPI_DOUBLE, direction[DOWN], 201, cart_comm, &request_recv[1]);

    // Prepare data to send
    double *send_eta_right = malloc(ny * sizeof(double));
    double *send_eta_up = malloc(nx * sizeof(double));

    for (int j_rank = 0; j_rank < ny; j_rank++)
        send_eta_right[j_rank] = get_value_MPI((*all_data)->eta, i_end, j_start+j_rank, rank_glob, cart_rank, cart_comm);
    for (int i_rank = 0; i_rank < nx; i_rank++)
        send_eta_up[i_rank] = get_value_MPI((*all_data)->eta, i_start+i_rank, j_end, rank_glob, cart_rank, cart_comm);

    // Send eta(i+1,j) and eta(i,j+1) to neighbouring processes
    MPI_Isend(send_eta_right, ny, MPI_DOUBLE, direction[RIGHT], 200, cart_comm, &request_send[0]);
    MPI_Isend(send_eta_up, nx, MPI_DOUBLE, direction[UP], 201, cart_comm, &request_send[1]);

    free(send_eta_right);
    free(send_eta_up);

    double dx = param.dx;
    double dy = param.dy;
    double c1 = param.dt * param.g;
    double c2 = param.dt * param.gamma;

    // Update velocities (inside domain)
    for (int i_rank = i_start+1; i_rank < i_end; i_rank++) {
        for (int j_rank = j_start+1; j_rank < j_end; j_rank++) {
            int i_local = i_rank - i_start;
            int j_local = j_rank - j_start;

            double eta_ij = GET((*all_data)->eta, i_local, j_local);
            double eta_imj = GET((*all_data)->eta, i_local - 1, j_local);
            double eta_ijm = GET((*all_data)->eta, i_local, j_local - 1);

            double u_ij = (1. - c2) * GET((*all_data)->u, i_local, j_local)
                - c1 / dx * (eta_ij - eta_imj);
            double v_ij = (1. - c2) * GET((*all_data)->v, i_local, j_local)
                - c1 / dy * (eta_ij - eta_ijm);

            SET((*all_data)->u, i_local, j_local, u_ij);
            SET((*all_data)->v, i_local, j_local, v_ij);
        }
    }

    // Wait for received data
    MPI_Waitall(2, request_recv, MPI_STATUSES_IGNORE);

    // Update velocities (left boundary)
    int i_rank = i_start;
    for (int j_rank = j_start+1; j_rank < j_end; j_rank++) {
        int i_local = i_rank - i_start;
        int j_local = j_rank - j_start;

        double eta_ij = GET((*all_data)->eta, i_local, j_local);
        double eta_imj = (*all_data)->eta->edge_vals[LEFT][j_local];
        double eta_ijm = GET((*all_data)->eta, i_local, j_local - 1);

        double u_ij = (1. - c2) * GET((*all_data)->u, i_local, j_local)
            - c1 / dx * (eta_ij - eta_imj);
        double v_ij = (1. - c2) * GET((*all_data)->v, i_local, j_local)
            - c1 / dy * (eta_ij - eta_ijm);

        SET((*all_data)->u, i_local, j_local, u_ij);
        SET((*all_data)->v, i_local, j_local, v_ij);
    }

    // Update velocities (bottom boundary)
    int j_rank = j_start;
    for (int i_rank = i_start+1; i_rank < i_end; i_rank++) {
        int i_local = i_rank - i_start;
        int j_local = j_rank - j_start;

        double eta_ij = GET((*all_data)->eta, i_local, j_local);
        double eta_imj = GET((*all_data)->eta, i_local - 1, j_local);
        double eta_ijm = (*all_data)->eta->edge_vals[DOWN][i_local];

        double u_ij = (1. - c2) * GET((*all_data)->u, i_local, j_local)
            - c1 / dx * (eta_ij - eta_imj);
        double v_ij = (1. - c2) * GET((*all_data)->v, i_local, j_local)
            - c1 / dy * (eta_ij - eta_ijm);

        SET((*all_data)->u, i_local, j_local, u_ij);
        SET((*all_data)->v, i_local, j_local, v_ij);
    }

    // Wait for sent data to complete
    MPI_Waitall(2, request_send, MPI_STATUSES_IGNORE);

    free(send_eta_right);
    free(send_eta_up);
}

double interpolate_data(const data_t *data, 
                        double x, 
                        double y) {

  int i = (int)(x / data->dx);
  int j = (int)(y / data->dy);

  // Boundary cases
  if (i < 0 || j < 0 || i >= data->nx - 1 || j >= data->ny - 1) {

      i = (i < 0) ? 0 : (i >= data->nx) ? data->nx - 1 : i;
      j = (j < 0) ? 0 : (j >= data->ny) ? data->ny - 1 : j;
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

void interp_bathy(int nx,
                  int ny, 
                  const struct parameters param,
                  all_data_t *all_data) {

  for(int i = 0; i < nx; i++){
    for(int j = 0; j < ny; j++){

      double x = i * param.dx;
      double y = j * param.dy;
      double val = interpolate_data(all_data->h, x, y);
      SET(all_data->h_interp, i, j, val);
    }
  }
}


void boundary_source_condition(int n,
                        int nx, 
                        int ny, 
                        const struct parameters param, 
                        all_data_t **all_data) {
    double t = n * param.dt;
    if(param.source_type == 1) {
      // sinusoidal velocity on top boundary
      double A = 5;
      double f = 1. / 20.;
      for(int i = 0; i < nx; i++) {
        for(int j = 0; j < ny; j++) {
          SET((*all_data)->u, 0, j, 0.);
          SET((*all_data)->u, nx, j, 0.);
          SET((*all_data)->v, i, 0, 0.);
          SET((*all_data)->v, i, ny, A * sin(2 * M_PI * f * t));
        }
      }
    }
    else if(param.source_type == 2) {
      // sinusoidal elevation in the middle of the domain
      double A = 5;
      double f = 1. / 20.;
      SET((*all_data)->eta, nx / 2, ny / 2, A * sin(2 * M_PI * f * t));
    }
    else {
      // TODO: add other sources
      printf("Error: Unknown source type %d\n", param.source_type);
      exit(0);
    }
}



int main(int argc, char **argv) {


    if(argc != 2){
        printf("Usage: %s parameter_file\n", argv[0]);
        return 1;
    }

    //---------------------//
    // INITIALIZE TOPOLOGY //
    //---------------------//
    int nb_process, rank, cart_rank, ierr;
    int dims[2] = {0, 0};
    int periods[2] = {0, 0};
    int reorder = 1;
    MPI_Comm cart_comm;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &nb_process);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); // give current rank (in global communicator) i.e. physical topology

    // Create 2D Cartesian communicator
    MPI_Dims_create(nb_process, 2, dims);
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, reorder, &cart_comm); // create 2D cartesian communicator
    MPI_Comm_rank(cart_comm, &cart_rank); // give current rank (in new cartesian communicator) i.e. logic topology

    int coords[2];
    MPI_Cart_coords(cart_comm, cart_rank, 2, coords);

    int neighbors[4];
    MPI_Cart_shift(cart_comm, 0, 1, &neighbors[LEFT], &neighbors[RIGHT]);
    MPI_Cart_shift(cart_comm, 1, 1, &neighbors[UP], &neighbors[DOWN]);


    all_data_t* all_data = allocate_all_data();
    struct parameters param;
    

    if (cart_rank == 0) {
      if(read_parameters(&param, argv[1])) return 1;
      if(read_data(all_data->h, param.input_h_filename)) return 1;
      print_parameters(&param);
    } 
    
    // Infer size of domain from input bathymetric data
    double hx = all_data->h->nx * all_data->h->dx;
    double hy = all_data->h->ny * all_data->h->dy;
    double dx = param.dx;
    double dy = param.dy;
    int nx = floor(hx / param.dx);
    int ny = floor(hy / param.dy);
    if(nx <= 0) nx = 1;
    if(ny <= 0) ny = 1;
    int nt = floor(param.max_t / param.dt);


    //------------------------------------------------------//
    // INITIALIZE COMMUNICATING VARIABLES FOR MPI PROCESSES //
    //------------------------------------------------------//
    data_t *gathered_output;
    double *receive_data;
    limit_t **rank_glob;
    neighbour_t direction[NEIGHBOR_NUM];

    int *recv_size = malloc(sizeof(int) * nb_process);
    int *displacements = malloc(sizeof(int) * nb_process);
    if (recv_size == NULL || displacements == NULL) {
        printf("Rank %d: Error allocating recv_size or displacements\n", cart_rank);
        MPI_Abort(cart_comm, MPI_ERR_NO_MEM);
    }

    // cart_rank 0 is the "master rank" where all info are gathered
    if (cart_rank == 0) {

        int rank_coords[2];
        int num[2] = {nx, ny};

        gathered_output = malloc(sizeof(data_t));
        if  (gathered_output == NULL ||
            (displacements = malloc(sizeof(int) * nb_process)) == NULL ||
            (receive_data = malloc(sizeof(double) * nx * ny)) == NULL ||
            (rank_glob = malloc(sizeof(limit_t*) * nb_process)) == NULL ||
            (recv_size = malloc(sizeof(int) * nb_process)) == NULL) {
            printf("Error when allocating memory\n");
            MPI_Abort(cart_comm, MPI_ERR_NO_MEM);
        }

        gathered_output->vals = receive_data;
        gathered_output->nx = nx;
        gathered_output->ny = ny;
        gathered_output->dx = param.dx;
        gathered_output->dy = param.dy;
        gathered_output->edge_vals = NULL;  // Not needed for gathered output

        for (int r = 0; r < nb_process; r++) {
            if ((rank_glob[r] = malloc(2 * sizeof(limit_t))) == NULL) {
                printf("Error when allocating memory\n");
                MPI_Abort(cart_comm, MPI_ERR_NO_MEM);
            }

            MPI_Cart_coords(cart_comm, r, 2, rank_coords);
            recv_size[r] = 1;
            for (int i = 0; i < 2; i++) {
                rank_glob[r][i].start = num[i] * rank_coords[i] / dims[i];
                rank_glob[r][i].end = num[i] * (rank_coords[i] + 1) / dims[i] - 1;
                rank_glob[r][i].n = (rank_glob[r][i].end - rank_glob[r][i].start + 1);

                recv_size[r] *= rank_glob[r][i].n;
            }

            // Displacement is the index (in the final buffer) where to store data from rank "r"
            if (r == 0) displacements[r] = 0;
            else displacements[r] = displacements[r - 1] + recv_size[r - 1];
        }
    }

    // Après l'initialisation par le processus 0
    MPI_Bcast(recv_size, nb_process, MPI_INT, 0, cart_comm);
    MPI_Bcast(displacements, nb_process, MPI_INT, 0, cart_comm);

    MPI_Barrier(cart_comm); printf("Process %d: recv_size = %d, displacement = %d\n", cart_rank, recv_size[cart_rank], displacements[cart_rank]); fflush(stdout);

    double start = GET_TIME(); 
    // Loop over timestep
    for (int n = 0; n < nt; n++) {

      // print elapsed time
      if (n && (n % (nt / 10)) == 0 && cart_rank == 0) {
        double time_sofar = GET_TIME() - start;
        double eta = (nt - n) * time_sofar / n;
        printf("Computing step %d/%d (ETA: %g seconds)     \r", n, nt, eta);
        fflush(stdout);
      }

      // impose boundary conditions
      //boundary_source_condition(n, nx, ny, param, &all_data);

      // Gather all output from processes
      if (param.sampling_rate && !(n % param.sampling_rate)) {

        int send_size = RANK_NX(cart_rank) * RANK_NY(cart_rank);
        data_t *output_data[] = {all_data->eta, all_data->u, all_data->v};
        const char *output_files[] = {param.output_eta_filename, param.output_u_filename, param.output_v_filename};

        
        for (int i = 0; i < 3; i++) {

          
          // Tentative de MPI_Gatherv avec gestion d'erreur
          int gatherv_result = MPI_Gatherv(output_data[i]->vals, send_size, MPI_DOUBLE, 
                                     receive_data, recv_size, displacements, MPI_DOUBLE, 
                                        0, cart_comm);
          
          if (gatherv_result != MPI_SUCCESS) {
              char error_string[MPI_MAX_ERROR_STRING];
              int length_of_error_string;
              MPI_Error_string(gatherv_result, error_string, &length_of_error_string);
              printf("Rank %d: MPI_Gatherv failed: %s\n", cart_rank, error_string);
              MPI_Abort(cart_comm, gatherv_result);
          }
          

   
          if (cart_rank == 0) {
            for (int r = 0; r < nb_process; r++) {
              for (int j = 0; j < RANK_NY(r); j++) {
                for (int i = 0; i < RANK_NX(r); i++) {
                  int global_i = START_I(r) + i;
                  int global_j = START_J(r) + j;
                  int local_index = j * RANK_NX(r) + i;
                  int global_index = global_j * nx + global_i;

                  gathered_output->vals[global_index] = 
                  receive_data[displacements[r] + local_index];
                }
              }
            }
            //write_data_vtk(&all_data->eta, "water elevation", param.output_eta_filename, n);
            //write_data_vtk(&u, "x velocity", param.output_u_filename, n);
            //write_data_vtk(&v, "y velocity", param.output_v_filename, n);
          }
        }

        printf("Exiting sampling block\n");
      }
      
      

      // Update variables
      //printf("Before update_eta\n");
      //update_eta(param, &all_data, rank_glob, cart_rank, cart_comm, direction);
      //printf("After update_eta\n");
      
      //printf("Before update_velocities\n");
      ////update_velocities(param, &all_data, rank_glob, cart_rank, cart_comm, direction);
      //printf("After update_velocities\n");
      
      //printf("Iteration %d end\n", n);
      }
      
      

    // Clean up
    /*
    if (cart_rank == 0) {
        printf("Start clean up\n");
        cleanup(all_data, &param, cart_rank, nb_process, gathered_output, receive_data, rank_glob, recv_size, displacements);
        printf("End clean up \n");
        printf("Start MPI_finalized\n");
    }
    */

      MPI_Finalize();

    if (cart_rank == 0) {
        printf("End MPI_finalized\n");
        printf("finished\n");
    }

  return 0;
}
