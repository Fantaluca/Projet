#include "shallow_mpi.h"

#undef MPI_STATUSES_IGNORE
#define MPI_STATUSES_IGNORE (MPI_Status *)0 // temporary to get rid of "warning: 'MPI_Waitall' accessing 20 bytes in a region of size 0 [-Wstringop-overflow=]"

double get_value_MPI(data_t *data, 
                     int i, 
                     int j, 
                     gather_data_t *gdata,
                     MPITopology *topo) {

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


  int i_rank = i - START_I(gdata, topo->cart_rank); 
  int j_rank = j - START_J(gdata, topo->cart_rank); 
  int nx = RANK_NX(gdata, topo->cart_rank);
  int ny = RANK_NY(gdata, topo->cart_rank); 

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
  else{ printf("Error while accessing subdomain"); MPI_Abort(topo->cart_comm, MPI_ERR_ACCESS); }

}

double set_value_MPI(data_t *data, 
                     int i, 
                     int j, 
                     gather_data_t *gdata,
                     MPITopology *topo,
                     double val) {
                      
  int i_rank = i - START_I(gdata, topo->cart_rank); 
  int j_rank = j - START_J(gdata, topo->cart_rank); 
  int nx = RANK_NX(gdata, topo->cart_rank);
  int ny = RANK_NY(gdata, topo->cart_rank); 

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
  else{ printf("Error while accessing subdomain"); MPI_Abort(topo->cart_comm, MPI_ERR_ACCESS); }

}

void update_eta(const parameters_t param, 
                all_data_t **all_data,
                gather_data_t *gdata,
                MPITopology *topo) {

    MPI_Request request_recv[2] = {MPI_REQUEST_NULL, MPI_REQUEST_NULL};
    MPI_Request request_send[2] = {MPI_REQUEST_NULL, MPI_REQUEST_NULL};
    MPI_Status status[2];

    int nx = (*all_data)->eta->nx;
    int ny = (*all_data)->eta->ny;

    // Allocate buffers
    double *send_u = calloc(ny, sizeof(double));
    double *send_v = calloc(nx, sizeof(double));
    double *recv_buffer_u = NULL;
    double *recv_buffer_v = NULL;

    if (!send_u || !send_v) {
        fprintf(stderr, "Process %d: Failed to allocate send buffers\n", topo->cart_rank);
        MPI_Abort(topo->cart_comm, 1);
        return;
    }

    // Allouer les tampons de réception si nécessaire
    if (topo->neighbors[RIGHT] != MPI_PROC_NULL) {
        recv_buffer_u = calloc(ny, sizeof(double));
        if (!recv_buffer_u) {
            fprintf(stderr, "Process %d: Failed to allocate recv_buffer_u\n", topo->cart_rank);
            MPI_Abort(topo->cart_comm, 1);
            return;
        }
    }
    
    if (topo->neighbors[UP] != MPI_PROC_NULL) {
        recv_buffer_v = calloc(nx, sizeof(double));
        if (!recv_buffer_v) {
            fprintf(stderr, "Process %d: Failed to allocate recv_buffer_v\n", topo->cart_rank);
            MPI_Abort(topo->cart_comm, 1);
            return;
        }
    }


    // Remplir les tampons d'envoi
    for (int j = 0; j < ny; j++) send_u[j] = GET((*all_data)->u, 0, j);
    for (int i = 0; i < nx; i++) send_v[i] = GET((*all_data)->v, i, ny-1);


    // Communications MPI
    int recv_count = 0;
    if (topo->neighbors[RIGHT] != MPI_PROC_NULL) {
       
        MPI_Irecv(recv_buffer_u, ny, MPI_DOUBLE,
                  topo->neighbors[RIGHT], 100, topo->cart_comm, &request_recv[recv_count++]);
    }

    if (topo->neighbors[UP] != MPI_PROC_NULL) {
        MPI_Irecv(recv_buffer_v, nx, MPI_DOUBLE,
                  topo->neighbors[UP], 101, topo->cart_comm, &request_recv[recv_count++]);
    }

    // Envois
    int send_count = 0;
    if (topo->neighbors[LEFT] != MPI_PROC_NULL) {
       
        
        MPI_Isend(send_u, ny, MPI_DOUBLE, topo->neighbors[LEFT], 100,
                  topo->cart_comm, &request_send[send_count++]);
    }

    if (topo->neighbors[DOWN] != MPI_PROC_NULL) {
       
        MPI_Isend(send_v, nx, MPI_DOUBLE, topo->neighbors[DOWN], 101,
                  topo->cart_comm, &request_send[send_count++]);
    }

    // Attendre la fin des réceptions
    if (recv_count > 0) {
       
        MPI_Waitall(recv_count, request_recv, status);
    }

   

    // Update eta
    for (int j = 0; j < ny; j++) {
        for (int i = 0; i < nx; i++) {
            double h_ij = GET((*all_data)->h_interp, i, j);
            if (h_ij <= 0) continue;

            // Calculer les différences finies
            double u_i = GET((*all_data)->u, i, j);
            double u_ip1 = (i < nx-1) ? GET((*all_data)->u, i+1, j) : 0.0;
            
            double v_j = GET((*all_data)->v, i, j);
            double v_jp1 = (j < ny-1) ? GET((*all_data)->v, i, j+1) : 0.0;

            // Si on a reçu des données des voisins, les utiliser
            if (i == nx-1 && recv_buffer_u != NULL) {
                u_ip1 = recv_buffer_u[j];
            }
            if (j == ny-1 && recv_buffer_v != NULL) {
                v_jp1 = recv_buffer_v[i];
            }

            double du_dx = (u_ip1 - u_i) / param.dx;
            double dv_dy = (v_jp1 - v_j) / param.dy;

            double eta_new = GET((*all_data)->eta, i, j) - param.dt * h_ij * (du_dx + dv_dy);
            SET((*all_data)->eta, i, j, eta_new);
        }
    }

    // Attendre la fin des envois
    if (send_count > 0) {
      
        MPI_Waitall(send_count, request_send, status);
    }

    // Nettoyage
    free(send_u);
    free(send_v);
    if (recv_buffer_u) free(recv_buffer_u);
    if (recv_buffer_v) free(recv_buffer_v);

}

void update_velocities(const parameters_t param,
                      all_data_t **all_data,
                      gather_data_t *gdata,
                      MPITopology *topo) {

    MPI_Request request_recv[2] = {MPI_REQUEST_NULL, MPI_REQUEST_NULL};
    MPI_Request request_send[2] = {MPI_REQUEST_NULL, MPI_REQUEST_NULL};
    MPI_Status status[2];

    // Dimensions des domaines locaux
    int nx = (*all_data)->eta->nx;
    int ny = (*all_data)->eta->ny;
    
    // Allocation des tampons
    double *send_eta_right = calloc(ny, sizeof(double));
    double *send_eta_up = calloc(nx, sizeof(double));
    double *recv_buffer_left = NULL;
    double *recv_buffer_down = NULL;

    if (!send_eta_right || !send_eta_up) {
        fprintf(stderr, "Process %d: Failed to allocate send buffers\n", topo->cart_rank);
        MPI_Abort(topo->cart_comm, 1);
        return;
    }

    // Allocation des tampons de réception si nécessaire
    if (topo->neighbors[LEFT] != MPI_PROC_NULL) {
        recv_buffer_left = calloc(ny, sizeof(double));
        if (!recv_buffer_left) {
            fprintf(stderr, "Process %d: Failed to allocate recv_buffer_left\n", topo->cart_rank);
            MPI_Abort(topo->cart_comm, 1);
            return;
        }
    }
    
    if (topo->neighbors[DOWN] != MPI_PROC_NULL) {
        recv_buffer_down = calloc(nx, sizeof(double));
        if (!recv_buffer_down) {
            fprintf(stderr, "Process %d: Failed to allocate recv_buffer_down\n", topo->cart_rank);
            MPI_Abort(topo->cart_comm, 1);
            return;
        }
    }

    // Remplir les tampons d'envoi
    for (int j = 0; j < ny; j++) send_eta_right[j] = GET((*all_data)->eta, nx-1, j);
    for (int i = 0; i < nx; i++) send_eta_up[i] = GET((*all_data)->eta, i, ny-1);
    

    // Communications MPI
    int recv_count = 0;
    if (topo->neighbors[LEFT] != MPI_PROC_NULL) {
       
        MPI_Irecv(recv_buffer_left, ny, MPI_DOUBLE,
                  topo->neighbors[LEFT], 200, topo->cart_comm, &request_recv[recv_count++]);
    }

    if (topo->neighbors[DOWN] != MPI_PROC_NULL) {
      
        MPI_Irecv(recv_buffer_down, nx, MPI_DOUBLE,
                  topo->neighbors[DOWN], 201, topo->cart_comm, &request_recv[recv_count++]);
    }

    // Envois
    int send_count = 0;
    if (topo->neighbors[RIGHT] != MPI_PROC_NULL) {
        
        MPI_Isend(send_eta_right, ny, MPI_DOUBLE, topo->neighbors[RIGHT], 200,
                  topo->cart_comm, &request_send[send_count++]);
    }

    if (topo->neighbors[UP] != MPI_PROC_NULL) {
        
        MPI_Isend(send_eta_up, nx, MPI_DOUBLE, topo->neighbors[UP], 201,
                  topo->cart_comm, &request_send[send_count++]);
    }

    // Attendre la réception avant la mise à jour
    if (recv_count > 0) {
     
        MPI_Waitall(recv_count, request_recv, status);
    }

   

    // Mise à jour des vitesses
    double dx = param.dx;
    double dy = param.dy;
    double c1 = param.dt * param.g;
    double c2 = param.dt * param.gamma;

    for (int j = 0; j < ny; j++) {
        for (int i = 0; i < nx; i++) {

            double eta_ij = GET((*all_data)->eta, i, j);
            double eta_imj = (i > 0) ? GET((*all_data)->eta, i-1, j) : 
                           (recv_buffer_left ? recv_buffer_left[j] : eta_ij);
            double eta_ijm = (j > 0) ? GET((*all_data)->eta, i, j-1) :
                           (recv_buffer_down ? recv_buffer_down[i] : eta_ij);

            // Récupérer les vitesses actuelles
            double u_ij = GET((*all_data)->u, i, j);
            double v_ij = GET((*all_data)->v, i, j);

            // Calculer les nouvelles vitesses
            double new_u = (1.0 - c2) * u_ij - c1 / dx * (eta_ij - eta_imj);
            double new_v = (1.0 - c2) * v_ij - c1 / dy * (eta_ij - eta_ijm);

            // Mettre à jour les vitesses
            SET((*all_data)->u, i, j, new_u);
            SET((*all_data)->v, i, j, new_v);
        }
    }

    // Attendre la fin des envois
    if (send_count > 0) {
     
        MPI_Waitall(send_count, request_send, status);
    }

    // Nettoyage
    free(send_eta_right);
    free(send_eta_up);
    if (recv_buffer_left) free(recv_buffer_left);
    if (recv_buffer_down) free(recv_buffer_down);

   
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
                  all_data_t **all_data,
                  gather_data_t *gdata, 
                  MPITopology *topo) {
    
    // 1. Vérification initiale des pointeurs
    printf("Rank %d: Starting pointer checks in interp_bathy\n", topo->cart_rank);
    fflush(stdout);

    if (!all_data) {
        fprintf(stderr, "Rank %d: all_data pointer is NULL\n", topo->cart_rank);
        MPI_Abort(topo->cart_comm, 1);
        return;
    }
    printf("Rank %d: all_data pointer OK\n", topo->cart_rank);
    fflush(stdout);

    if (!(*all_data)) {
        fprintf(stderr, "Rank %d: *all_data is NULL\n", topo->cart_rank);
        MPI_Abort(topo->cart_comm, 1);
        return;
    }
    printf("Rank %d: *all_data pointer OK\n", topo->cart_rank);
    fflush(stdout);

    if (!(*all_data)->h) {
        fprintf(stderr, "Rank %d: (*all_data)->h is NULL\n", topo->cart_rank);
        MPI_Abort(topo->cart_comm, 1);
        return;
    }
    printf("Rank %d: h structure OK\n", topo->cart_rank);
    fflush(stdout);

    if (!(*all_data)->h_interp) {
        fprintf(stderr, "Rank %d: (*all_data)->h_interp is NULL\n", topo->cart_rank);
        MPI_Abort(topo->cart_comm, 1);
        return;
    }
    printf("Rank %d: h_interp structure OK\n", topo->cart_rank);
    fflush(stdout);

    // 2. Vérification des tableaux de valeurs
    if (!(*all_data)->h->vals) {
        fprintf(stderr, "Rank %d: h->vals is NULL\n", topo->cart_rank);
        MPI_Abort(topo->cart_comm, 1);
        return;
    }
    printf("Rank %d: h->vals OK\n", topo->cart_rank);
    fflush(stdout);

    if (!(*all_data)->h_interp->vals) {
        fprintf(stderr, "Rank %d: h_interp->vals is NULL\n", topo->cart_rank);
        MPI_Abort(topo->cart_comm, 1);
        return;
    }
    printf("Rank %d: h_interp->vals OK\n", topo->cart_rank);
    fflush(stdout);

    // 3. Vérification des dimensions
    printf("Rank %d: h dimensions: nx=%d, ny=%d, dx=%f, dy=%f\n",
           topo->cart_rank, 
           (*all_data)->h->nx, (*all_data)->h->ny,
           (*all_data)->h->dx, (*all_data)->h->dy);
    printf("Rank %d: h_interp dimensions: nx=%d, ny=%d, dx=%f, dy=%f\n",
           topo->cart_rank,
           (*all_data)->h_interp->nx, (*all_data)->h_interp->ny,
           (*all_data)->h_interp->dx, (*all_data)->h_interp->dy);
    printf("Rank %d: Global dimensions: nx_glob=%d, ny_glob=%d\n",
           topo->cart_rank, nx_glob, ny_glob);
    fflush(stdout);

    // 4. Vérification des données gather
    printf("Rank %d: Checking gather data structure\n", topo->cart_rank);
    fflush(stdout);

    if (!gdata) {
        fprintf(stderr, "Rank %d: gdata is NULL\n", topo->cart_rank);
        MPI_Abort(topo->cart_comm, 1);
        return;
    }
    printf("Rank %d: gdata OK\n", topo->cart_rank);
    fflush(stdout);

    if (!gdata->rank_glob) {
        fprintf(stderr, "Rank %d: gdata->rank_glob is NULL\n", topo->cart_rank);
        MPI_Abort(topo->cart_comm, 1);
        return;
    }
    printf("Rank %d: rank_glob OK\n", topo->cart_rank);
    fflush(stdout);

    int start_i = START_I(gdata, topo->cart_rank);
    int start_j = START_J(gdata, topo->cart_rank);

    printf("Rank %d: Local domain starts at (%d, %d)\n", 
           topo->cart_rank, start_i, start_j);
    fflush(stdout);

    // 5. Ajouter une vérification MPI
    int mpi_initialized;
    MPI_Initialized(&mpi_initialized);
    if (!mpi_initialized) {
        fprintf(stderr, "Rank %d: MPI not initialized!\n", topo->cart_rank);
        return;
    }
    printf("Rank %d: MPI status OK\n", topo->cart_rank);
    fflush(stdout);

    // Si nous arrivons ici, nous pouvons commencer l'interpolation
    printf("Rank %d: Starting interpolation loop\n", topo->cart_rank);
    fflush(stdout);

    int local_nx = (*all_data)->h_interp->nx;
    int local_ny = (*all_data)->h_interp->ny;

    // Premier point seulement pour test
    printf("Rank %d: Testing first point interpolation\n", topo->cart_rank);
    fflush(stdout);

    double x = (0 + start_i) * param.dx;
    double y = (0 + start_j) * param.dy;

    printf("Rank %d: First point coordinates: x=%f, y=%f\n", 
           topo->cart_rank, x, y);
    fflush(stdout);

    // Test interpolation sur un seul point
    double test_val = interpolate_data((*all_data)->h, nx_glob, ny_glob, x, y);
    printf("Rank %d: First interpolated value: %f\n", 
           topo->cart_rank, test_val);
    fflush(stdout);

    // Si le test réussit, continuez avec la boucle complète
    MPI_Barrier(topo->cart_comm);
}

void boundary_source_condition(int n, int nx_glob, int ny_glob,
                             const parameters_t param,
                             all_data_t **all_data,
                             gather_data_t *gdata,
                             MPITopology *topo) {


    // Calcul du temps courant
    double t = n * param.dt;
    
    // Dimensions locales des champs u et v
    int nx_u = (*all_data)->u->nx;
    int ny_u = (*all_data)->u->ny;
    int nx_v = (*all_data)->v->nx;
    int ny_v = (*all_data)->v->ny;
    
    if (param.source_type == 1) {
        double A = 5.0;
        double f = 1.0 / 20.0;
        
        // Conditions aux limites pour u
        if (topo->neighbors[LEFT] == MPI_PROC_NULL) {
            for (int j = 0; j < ny_u; j++) {
                SET((*all_data)->u, 0, j, 0.0);
            }
        }
        if (topo->neighbors[RIGHT] == MPI_PROC_NULL) {
            for (int j = 0; j < ny_u; j++) {
                SET((*all_data)->u, nx_u-1, j, 0.0);
            }
        }
        
        // Conditions aux limites pour v
        if (topo->neighbors[DOWN] == MPI_PROC_NULL) {
            for (int i = 0; i < nx_v; i++) {
                SET((*all_data)->v, i, 0, 0.0);
            }
        }
        if (topo->neighbors[UP] == MPI_PROC_NULL) {
            for (int i = 0; i < nx_v; i++) {
                SET((*all_data)->v, i, ny_v-1, A * sin(2.0 * M_PI * f * t));
            }
        }
    }
    else if (param.source_type == 2) {
        // Coordonnées du point source au centre du domaine global
        int global_middle_i = nx_glob / 2;
        int global_middle_j = ny_glob / 2;
        
        // Conversion en coordonnées locales
        int local_i = global_middle_i - START_I(gdata, topo->cart_rank);
        int local_j = global_middle_j - START_J(gdata, topo->cart_rank);
        
        // Vérifier si le point source est dans ce sous-domaine
        if (local_i >= 0 && local_i < (*all_data)->eta->nx && 
            local_j >= 0 && local_j < (*all_data)->eta->ny) {
            double A = 5.0;
            double f = 1.0 / 20.0;
            SET((*all_data)->eta, local_i, local_j, A * sin(2.0 * M_PI * f * t));
        }
    }
    
    MPI_Barrier(topo->cart_comm);
}

all_data_t* init_all_data(const parameters_t *param, MPITopology *topo) {
    printf("Rank %d: Entering init_all_data\n", topo->cart_rank);
    fflush(stdout);

    all_data_t* all_data = malloc(sizeof(all_data_t));
    if (all_data == NULL) {
        fprintf(stderr, "Error: Failed to allocate all_data\n");
        return NULL;
    }

    // Initialize all pointers to NULL
    all_data->u = NULL;
    all_data->v = NULL;
    all_data->eta = NULL;
    all_data->h = NULL;
    all_data->h_interp = NULL;

    // Allocate and read bathymetry data
    all_data->h = malloc(sizeof(data_t));
    if (all_data->h == NULL) {
        fprintf(stderr, "Error: Failed to allocate h structure\n");
        free_all_data(all_data);
        return NULL;
    }

    printf("Rank %d: Reading bathymetry data from %s\n", 
           topo->cart_rank, param->input_h_filename);
    fflush(stdout);

    if (read_data(all_data->h, param->input_h_filename)) {
        fprintf(stderr, "Error: Failed to read bathymetry data\n");
        free_all_data(all_data);
        return NULL;
    }

    // Calculate global domain dimensions
    double hx = all_data->h->nx * all_data->h->dx;
    double hy = all_data->h->ny * all_data->h->dy;
    int nx_glob = floor(hx / param->dx);
    int ny_glob = floor(hy / param->dy);

    // Calculate local dimensions
    int local_nx = nx_glob / topo->dims[0];
    int local_ny = ny_glob / topo->dims[1];

    // Adjust for remainder
    if (topo->coords[0] < (nx_glob % topo->dims[0])) local_nx++;
    if (topo->coords[1] < (ny_glob % topo->dims[1])) local_ny++;

    printf("Rank %d: Global dimensions: %dx%d, Local dimensions: %dx%d\n",
           topo->cart_rank, nx_glob, ny_glob, local_nx, local_ny);
    fflush(stdout);

    // Allocate other fields with correct dimensions
    all_data->eta = malloc(sizeof(data_t));
    all_data->u = malloc(sizeof(data_t));
    all_data->v = malloc(sizeof(data_t));
    all_data->h_interp = malloc(sizeof(data_t));

    if (!all_data->eta || !all_data->u || !all_data->v || !all_data->h_interp) {
        fprintf(stderr, "Error: Failed to allocate data structures\n");
        free_all_data(all_data);
        return NULL;
    }

    // Initialize local fields with correct dimensions
    // eta field
    if (init_data(all_data->eta, local_nx, local_ny, 
                  param->dx, param->dy, 0.0, 1)) {
        fprintf(stderr, "Error: Failed to initialize eta\n");
        free_all_data(all_data);
        return NULL;
    }

    // u field (one more point in x direction)
    if (init_data(all_data->u, local_nx + 1, local_ny, 
                  param->dx, param->dy, 0.0, 1)) {
        fprintf(stderr, "Error: Failed to initialize u\n");
        free_all_data(all_data);
        return NULL;
    }

    // v field (one more point in y direction)
    if (init_data(all_data->v, local_nx, local_ny + 1, 
                  param->dx, param->dy, 0.0, 1)) {
        fprintf(stderr, "Error: Failed to initialize v\n");
        free_all_data(all_data);
        return NULL;
    }

    // h_interp field
    if (init_data(all_data->h_interp, local_nx, local_ny, 
                  param->dx, param->dy, 0.0, 0)) {
        fprintf(stderr, "Error: Failed to initialize h_interp\n");
        free_all_data(all_data);
        return NULL;
    }

    printf("Rank %d: Successfully initialized all fields\n", topo->cart_rank);
    fflush(stdout);

    return all_data;
}

int initialize_mpi_topology(int argc, char **argv, MPITopology *topo) {
    int err;
    char error_string[MPI_MAX_ERROR_STRING];
    int length;
    
    printf("Demarrage de l'initialisation MPI...\n");
    fflush(stdout);  
    
    int periods[2] = {0, 0};
    int reorder = 1;
    topo->dims[0] = 0;
    topo->dims[1] = 0;
    
    if ((err = MPI_Init(&argc, &argv)) != MPI_SUCCESS) {
        MPI_Error_string(err, error_string, &length);
        fprintf(stderr, "Erreur : echec de l'initialisation MPI - %s\n", error_string);
        fflush(stderr);
        return 1;
    }
    printf("MPI_Init reussi\n");
    fflush(stdout);
    
    if ((err = MPI_Comm_size(MPI_COMM_WORLD, &topo->nb_process)) != MPI_SUCCESS) {
        MPI_Error_string(err, error_string, &length);
        fprintf(stderr, "Erreur : Impossible d'obtenir le nombre de processus - %s\n", error_string);
        fflush(stderr);
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }
    printf("Nombre de processus: %d\n", topo->nb_process);
    fflush(stdout);
    
    if ((err = MPI_Comm_rank(MPI_COMM_WORLD, &topo->rank)) != MPI_SUCCESS) {
        MPI_Error_string(err, error_string, &length);
        fprintf(stderr, "Erreur : Impossible d'obtenir le rang du processus - %s\n", error_string);
        fflush(stderr);
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }
    printf("Process rank: %d\n", topo->rank);
    fflush(stdout);
    
    if ((err = MPI_Dims_create(topo->nb_process, 2, topo->dims)) != MPI_SUCCESS) {
        MPI_Error_string(err, error_string, &length);
        fprintf(stderr, "Erreur : echec de la creation des dimensions - %s\n", error_string);
        fflush(stderr);
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }
    printf("Process %d: Dimensions creees: [%d, %d]\n", topo->rank, topo->dims[0], topo->dims[1]);
    fflush(stdout);
    
    if (topo->dims[0] * topo->dims[1] != topo->nb_process) {
        fprintf(stderr, "Process %d: Erreur dimensions (%d x %d) != nb_process (%d)\n",
                topo->rank, topo->dims[0], topo->dims[1], topo->nb_process);
        fflush(stderr);
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }
    
    if ((err = MPI_Cart_create(MPI_COMM_WORLD, 2, topo->dims, periods, reorder, &topo->cart_comm)) != MPI_SUCCESS) {
        MPI_Error_string(err, error_string, &length);
        fprintf(stderr, "Process %d: Erreur creation communicateur cartesien - %s\n", topo->rank, error_string);
        fflush(stderr);
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }
    printf("Process %d: Communicateur cartesien cree\n", topo->rank);
    fflush(stdout);
    
    if ((err = MPI_Comm_rank(topo->cart_comm, &topo->cart_rank)) != MPI_SUCCESS) {
        MPI_Error_string(err, error_string, &length);
        fprintf(stderr, "Process %d: Erreur rang cartesien - %s\n", topo->rank, error_string);
        fflush(stderr);
        MPI_Comm_free(&topo->cart_comm);
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }
    printf("Process %d: Rang cartesien: %d\n", topo->rank, topo->cart_rank);
    fflush(stdout);
    
    if ((err = MPI_Cart_coords(topo->cart_comm, topo->cart_rank, 2, topo->coords)) != MPI_SUCCESS) {
        MPI_Error_string(err, error_string, &length);
        fprintf(stderr, "Process %d: Erreur coordonnees cartesiennes - %s\n", topo->rank, error_string);
        fflush(stderr);
        MPI_Comm_free(&topo->cart_comm);
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }
    printf("Process %d: Coordonnees: [%d, %d]\n", topo->rank, topo->coords[0], topo->coords[1]);
    fflush(stdout);
    
    if ((err = MPI_Cart_shift(topo->cart_comm, 0, 1, &topo->neighbors[LEFT], &topo->neighbors[RIGHT])) != MPI_SUCCESS) {
        MPI_Error_string(err, error_string, &length);
        fprintf(stderr, "Process %d: Erreur voisins gauche/droite - %s\n", topo->rank, error_string);
        fflush(stderr);
        MPI_Comm_free(&topo->cart_comm);
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }
    
    if ((err = MPI_Cart_shift(topo->cart_comm, 1, 1, &topo->neighbors[DOWN], &topo->neighbors[UP])) != MPI_SUCCESS) {
        MPI_Error_string(err, error_string, &length);
        fprintf(stderr, "Process %d: Erreur voisins haut/bas - %s\n", topo->rank, error_string);
        fflush(stderr);
        MPI_Comm_free(&topo->cart_comm);
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }
    
    printf("Process %d: Voisins - Gauche: %d, Droite: %d, Haut: %d, Bas: %d\n", 
           topo->rank, topo->neighbors[LEFT], topo->neighbors[RIGHT], 
           topo->neighbors[UP], topo->neighbors[DOWN]);
    fflush(stdout);
    
    MPI_Barrier(MPI_COMM_WORLD);
    
    if (topo->rank == 0) {
        printf("\nRecapitulatif de l'initialisation :\n");
        printf("Nombre total de processus : %d\n", topo->nb_process);
        printf("Dimensions de la grille : %d x %d\n", topo->dims[0], topo->dims[1]);
        fflush(stdout);
    }
    
    return 0;
}

int initialize_gather_structures(const MPITopology *topo, 
                               gather_data_t *gdata,
                               int nx_glob, int ny_glob,
                               double dx, double dy) {
    printf("Rank %d: Initializing gather structures\n", topo->cart_rank);
    fflush(stdout);
    
    // Initialiser tous les pointeurs à NULL
    memset(gdata, 0, sizeof(gather_data_t));
    
    // Allocation pour les tableaux de taille et déplacements
    gdata->recv_size_eta = calloc(topo->nb_process, sizeof(int));
    gdata->recv_size_u = calloc(topo->nb_process, sizeof(int));
    gdata->recv_size_v = calloc(topo->nb_process, sizeof(int));
    gdata->displacements_eta = calloc(topo->nb_process, sizeof(int));
    gdata->displacements_u = calloc(topo->nb_process, sizeof(int));
    gdata->displacements_v = calloc(topo->nb_process, sizeof(int));

    // Vérification des allocations
    if (!gdata->recv_size_eta || !gdata->recv_size_u || !gdata->recv_size_v ||
        !gdata->displacements_eta || !gdata->displacements_u || !gdata->displacements_v) {
        fprintf(stderr, "Rank %d: Failed to allocate size arrays\n", topo->cart_rank);
        return 1;
    }

    // Calcul des dimensions de base pour chaque processus
    int base_nx = nx_glob / topo->dims[0];
    int base_ny = ny_glob / topo->dims[1];
    int remainder_x = nx_glob % topo->dims[0];
    int remainder_y = ny_glob % topo->dims[1];

    // Allocation de rank_glob
    gdata->rank_glob = calloc(topo->nb_process, sizeof(limit_t*));
    if (!gdata->rank_glob) {
        fprintf(stderr, "Rank %d: Failed to allocate rank_glob\n", topo->cart_rank);
        return 1;
    }

    printf("Rank %d: Allocated rank_glob array\n", topo->cart_rank);
    fflush(stdout);

    // Allocation pour chaque processus
    for (int r = 0; r < topo->nb_process; r++) {
        gdata->rank_glob[r] = calloc(2, sizeof(limit_t));
        if (!gdata->rank_glob[r]) {
            fprintf(stderr, "Rank %d: Failed to allocate rank_glob[%d]\n", topo->cart_rank, r);
            return 1;
        }
    }

    printf("Rank %d: Allocated all rank_glob entries\n", topo->cart_rank);
    fflush(stdout);

    // Le rang 0 calcule toutes les tailles et limites
    if (topo->cart_rank == 0) {
        int current_x = 0;
        for (int r = 0; r < topo->nb_process; r++) {
            int coords[2];
            MPI_Cart_coords(topo->cart_comm, r, 2, coords);

            // Calcul des dimensions locales
            int local_nx = base_nx + (coords[0] < remainder_x ? 1 : 0);
            int local_ny = base_ny + (coords[1] < remainder_y ? 1 : 0);

            // X direction
            gdata->rank_glob[r][0].start = current_x;
            gdata->rank_glob[r][0].n = local_nx;
            gdata->rank_glob[r][0].end = current_x + local_nx;

            // Y direction
            gdata->rank_glob[r][1].start = coords[1] * base_ny + (coords[1] < remainder_y ? coords[1] : remainder_y);
            gdata->rank_glob[r][1].n = local_ny;
            gdata->rank_glob[r][1].end = gdata->rank_glob[r][1].start + local_ny;

            // Calcul des tailles pour MPI_Gatherv
            gdata->recv_size_eta[r] = local_nx * local_ny;
            gdata->recv_size_u[r] = (local_nx + 1) * local_ny;
            gdata->recv_size_v[r] = local_nx * (local_ny + 1);

            // Mise à jour de current_x
            if (coords[0] == topo->dims[0] - 1) {
                current_x = 0;
            } else {
                current_x += local_nx;
            }
        }

        // Calcul des déplacements
        int offset_eta = 0, offset_u = 0, offset_v = 0;
        for (int r = 0; r < topo->nb_process; r++) {
            gdata->displacements_eta[r] = offset_eta;
            gdata->displacements_u[r] = offset_u;
            gdata->displacements_v[r] = offset_v;
            
            offset_eta += gdata->recv_size_eta[r];
            offset_u += gdata->recv_size_u[r];
            offset_v += gdata->recv_size_v[r];
        }

        // Allocation des buffers de réception pour le rang 0
        gdata->receive_data_eta = calloc(offset_eta, sizeof(double));
        gdata->receive_data_u = calloc(offset_u, sizeof(double));
        gdata->receive_data_v = calloc(offset_v, sizeof(double));
        
        gdata->gathered_output = calloc(3, sizeof(data_t));
        
        if (!gdata->receive_data_eta || !gdata->receive_data_u || !gdata->receive_data_v || 
            !gdata->gathered_output) {
            return 1;
        }

        for (int i = 0; i < 3; i++) {
            gdata->gathered_output[i].nx = nx_glob;
            gdata->gathered_output[i].ny = ny_glob;
            gdata->gathered_output[i].dx = dx;
            gdata->gathered_output[i].dy = dy;
            gdata->gathered_output[i].vals = calloc(nx_glob * ny_glob, sizeof(double));
            if (!gdata->gathered_output[i].vals) return 1;
        }
    }

    printf("Rank %d: Starting broadcasts\n", topo->cart_rank);
    fflush(stdout);

    // Synchronisation avant les broadcasts
    MPI_Barrier(topo->cart_comm);

    // Broadcast des tailles et déplacements
    MPI_Bcast(gdata->recv_size_eta, topo->nb_process, MPI_INT, 0, topo->cart_comm);
    MPI_Bcast(gdata->recv_size_u, topo->nb_process, MPI_INT, 0, topo->cart_comm);
    MPI_Bcast(gdata->recv_size_v, topo->nb_process, MPI_INT, 0, topo->cart_comm);
    MPI_Bcast(gdata->displacements_eta, topo->nb_process, MPI_INT, 0, topo->cart_comm);
    MPI_Bcast(gdata->displacements_u, topo->nb_process, MPI_INT, 0, topo->cart_comm);
    MPI_Bcast(gdata->displacements_v, topo->nb_process, MPI_INT, 0, topo->cart_comm);

    // Broadcast de rank_glob
    for (int r = 0; r < topo->nb_process; r++) {
        MPI_Bcast(&(gdata->rank_glob[r][0]), sizeof(limit_t), MPI_BYTE, 0, topo->cart_comm);
        MPI_Bcast(&(gdata->rank_glob[r][1]), sizeof(limit_t), MPI_BYTE, 0, topo->cart_comm);
    }

    // Synchronisation finale
    MPI_Barrier(topo->cart_comm);

    printf("Rank %d: gather structures initialized successfully\n", topo->cart_rank);
    fflush(stdout);

    return 0;
}

void free_all_data(all_data_t *all_data) {
    if (all_data == NULL) return;

    // Libérer chaque composant avec vérification NULL
    if (all_data->u != NULL) {
        free_data(all_data->u, 1);
        all_data->u = NULL;
    }
    
    if (all_data->v != NULL) {
        free_data(all_data->v, 1);
        all_data->v = NULL;
    }
    
    if (all_data->eta != NULL) {
        free_data(all_data->eta, 1);
        all_data->eta = NULL;
    }
    
    if (all_data->h != NULL) {
        free_data(all_data->h, 0);
        all_data->h = NULL;
    }
    
    if (all_data->h_interp != NULL) {
        free_data(all_data->h_interp, 0);
        all_data->h_interp = NULL;
    }

    // Libérer la structure principale
    free(all_data);
}

void free_data(data_t *data, int has_edges) {
    if (data == NULL) return;

    // 1. Libérer les valeurs principales si elles existent
    if (data->vals != NULL) {
        free(data->vals);
        data->vals = NULL;
    }

    // 2. Libérer les edge values si nécessaire
    if (has_edges && data->edge_vals != NULL) {
        for (int i = 0; i < NEIGHBOR_NUM; i++) {
            if (data->edge_vals[i] != NULL) {
                free(data->edge_vals[i]);
                data->edge_vals[i] = NULL;
            }
        }
        free(data->edge_vals);
        data->edge_vals = NULL;
    }

    // 3. Finalement libérer la structure elle-même
    free(data);
}

void cleanup(parameters_t *param, MPITopology *topo, gather_data_t *gdata) {
    if (gdata == NULL || topo == NULL) return;

    // Une seule barrière au début
    MPI_Barrier(topo->cart_comm);

    // Libérer les tableaux communs à tous les rangs
    if (gdata->recv_size_eta) { free(gdata->recv_size_eta); gdata->recv_size_eta = NULL; }
    if (gdata->recv_size_u) { free(gdata->recv_size_u); gdata->recv_size_u = NULL; }
    if (gdata->recv_size_v) { free(gdata->recv_size_v); gdata->recv_size_v = NULL; }
    if (gdata->displacements_eta) { free(gdata->displacements_eta); gdata->displacements_eta = NULL; }
    if (gdata->displacements_u) { free(gdata->displacements_u); gdata->displacements_u = NULL; }
    if (gdata->displacements_v) { free(gdata->displacements_v); gdata->displacements_v = NULL; }

    // Libérer les éléments spécifiques au rang 0
    if (topo->cart_rank == 0) {
        if (gdata->gathered_output) {
            free(gdata->gathered_output);
            gdata->gathered_output = NULL;
        }
        
        if (gdata->receive_data_eta) { free(gdata->receive_data_eta); gdata->receive_data_eta = NULL; }
        if (gdata->receive_data_u) { free(gdata->receive_data_u); gdata->receive_data_u = NULL; }
        if (gdata->receive_data_v) { free(gdata->receive_data_v); gdata->receive_data_v = NULL; }

        if (gdata->rank_glob) {
            for (int r = 0; r < topo->nb_process; r++) {
                if (gdata->rank_glob[r]) {
                    free(gdata->rank_glob[r]);
                    gdata->rank_glob[r] = NULL;
                }
            }
            free(gdata->rank_glob);
            gdata->rank_glob = NULL;
        }
    }

    // Une seule barrière à la fin
    MPI_Barrier(topo->cart_comm);
    
    free(gdata);
}

void cleanup_mpi_topology(MPITopology *topo) {
    if (topo->cart_comm != MPI_COMM_NULL && topo->cart_comm != MPI_COMM_WORLD) {
        MPI_Comm_free(&topo->cart_comm);
    }
}

