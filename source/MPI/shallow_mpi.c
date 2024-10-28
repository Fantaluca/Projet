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
                MPITopology *topo,
                neighbour_t *direction) {

  MPI_Request request_recv[2];
  MPI_Request request_send[2];

  // Find process size (nx,ny) and every (i,j) within
  int nx = RANK_NX(gdata, topo->cart_rank);
  int i_start = START_I(gdata, topo->cart_rank);
  int i_end = END_I(gdata, topo->cart_rank);

  int ny = RANK_NY(gdata, topo->cart_rank);
  int j_start = START_J(gdata, topo->cart_rank);
  int j_end = END_J(gdata, topo->cart_rank);


  // Receive u(i+1,j) and v(i,j+1) from neighbouring processes
  MPI_Irecv((*all_data)->u->edge_vals[RIGHT], ny, MPI_DOUBLE, direction[RIGHT], 100, topo->cart_comm, &request_recv[0]);
  MPI_Irecv((*all_data)->v->edge_vals[UP], nx, MPI_DOUBLE, direction[UP], 101, topo->cart_comm, &request_recv[1]);


  // Prepare edge data to send
  double *send_u = malloc(nx * sizeof(double));;
  double *send_v =  malloc(ny * sizeof(double));;

  for (int j_rank = 0; j_rank < ny; j_rank++) 
    send_u[j_rank] = get_value_MPI((*all_data)->u, i_start, j_start+j_rank, gdata, topo);

  for (int i_rank = 0; i_rank < nx; i_rank++) 
    send_v[i_rank] = get_value_MPI((*all_data)->u, i_start+i_rank, j_end, gdata, topo);

  // Send u(i-1,j) and v(i,j-1) to neighbouring processes
  MPI_Isend(send_u, ny, MPI_DOUBLE, direction[LEFT], 100, topo->cart_comm, &request_recv[0]);
  MPI_Isend(send_v, nx, MPI_DOUBLE, direction[DOWN], 101, topo->cart_comm, &request_recv[0]);


  // Update eta
  double dx = param.dx;
  double dy = param.dy;
  
  for (int i = i_start; i <= i_end; i++) {
    for (int j = j_start; j <= j_end; j++) {

      double h_ij = get_value_MPI((*all_data)->h_interp, i, j, gdata, topo);
      double c1 = param.dt * h_ij;
      
      double du_dx = (get_value_MPI((*all_data)->u, i+1, j, gdata, topo) - get_value_MPI((*all_data)->u, i, j, gdata, topo))/dx;
      double dv_dy = (get_value_MPI((*all_data)->v, i, j+1, gdata, topo) - get_value_MPI((*all_data)->v , i, j, gdata, topo))/dy;
      double eta_ij = get_value_MPI((*all_data)->eta, i, j, gdata, topo) - c1 * (du_dx + dv_dy);
      
      set_value_MPI((*all_data)->eta, i, j, gdata, topo, eta_ij);
    }
  }

  // Wait for sent data to complete
  MPI_Waitall(2, request_send, MPI_STATUSES_IGNORE);

  free(send_u);
  free(send_v);

}

void update_velocities(const parameters_t param,
                       all_data_t **all_data,
                       gather_data_t *gdata,
                       MPITopology *topo,
                       neighbour_t *direction) {

    MPI_Request request_recv[2];
    MPI_Request request_send[2];

    // Find process size (nx,ny) and every (i,j) within
    int nx = RANK_NX(gdata, topo->cart_rank);
    int i_start = START_I(gdata, topo->cart_rank);
    int i_end = END_I(gdata, topo->cart_rank);

    int ny = RANK_NY(gdata, topo->cart_rank);
    int j_start = START_J(gdata, topo->cart_rank);
    int j_end = END_J(gdata, topo->cart_rank);


    // Receive eta(i-1,j) and eta(i,j-1) from neighbouring processes
    MPI_Irecv((*all_data)->eta->edge_vals[LEFT], ny, MPI_DOUBLE, direction[LEFT], 200, topo->cart_comm, &request_recv[0]);
    MPI_Irecv((*all_data)->eta->edge_vals[DOWN], nx, MPI_DOUBLE, direction[DOWN], 201, topo->cart_comm, &request_recv[1]);

    // Prepare data to send
    double *send_eta_right = malloc(ny * sizeof(double));
    double *send_eta_up = malloc(nx * sizeof(double));

    for (int j_rank = 0; j_rank < ny; j_rank++)
        send_eta_right[j_rank] = get_value_MPI((*all_data)->eta, i_end, j_start+j_rank, gdata, topo);
    for (int i_rank = 0; i_rank < nx; i_rank++)
        send_eta_up[i_rank] = get_value_MPI((*all_data)->eta, i_start+i_rank, j_end, gdata, topo);

    // Send eta(i+1,j) and eta(i,j+1) to neighbouring processes
    MPI_Isend(send_eta_right, ny, MPI_DOUBLE, direction[RIGHT], 200, topo->cart_comm, &request_send[0]);
    MPI_Isend(send_eta_up, nx, MPI_DOUBLE, direction[UP], 201, topo->cart_comm, &request_send[1]);


    double dx = param.dx;
    double dy = param.dy;
    double c1 = param.dt * param.g;
    double c2 = param.dt * param.gamma;

    // Update velocities 
    for (int i = i_start; i <= i_end; i++) {
        for (int j = j_start; j < j_end; j++) {

            double eta_ij = GET((*all_data)->eta, i, j);
            double eta_imj = GET((*all_data)->eta, i-1, j);
            double eta_ijm = GET((*all_data)->eta, i, j-1);

            double u_ij = (1. - c2) * GET((*all_data)->u, i, j)
                          - c1 / dx * (eta_ij - eta_imj);
            double v_ij = (1. - c2) * GET((*all_data)->v, i, j)
                          - c1 / dy * (eta_ij - eta_ijm);

            SET((*all_data)->u, i, j, u_ij);
            SET((*all_data)->v, i, j, v_ij);
        }
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
                  const parameters_t param,
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
                        const parameters_t param, 
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

all_data_t* init_all_data(const parameters_t *param, MPITopology *topo) {
    // 1. Allouer la structure principale
    all_data_t* all_data = malloc(sizeof(all_data_t));
    if (all_data == NULL) {
        fprintf(stderr, "Erreur d'allocation pour all_data\n");
        return NULL;
    }

    // 2. Initialiser tous les pointeurs à NULL
    all_data->u = NULL;
    all_data->v = NULL;
    all_data->eta = NULL;
    all_data->h = NULL;
    all_data->h_interp = NULL;

    // 3. Allouer chaque composant de data_t
    all_data->h = malloc(sizeof(data_t));
    if (all_data->h == NULL || read_data(all_data->h, param->input_h_filename)) {
        fprintf(stderr, "Erreur lors de la lecture des donnees bathymetriques\n");
        free_all_data(all_data);
        return NULL;
    }

    // 4. Maintenant on peut calculer les dimensions
    double hx = all_data->h->nx * all_data->h->dx;
    double hy = all_data->h->ny * all_data->h->dy;
    int nx = floor(hx / param->dx);
    int ny = floor(hy / param->dy);
    int local_nx = nx / topo->dims[0];
    int local_ny = ny / topo->dims[1];

    // 5. Allouer les autres structures
    all_data->eta = malloc(sizeof(data_t));
    all_data->u = malloc(sizeof(data_t));
    all_data->v = malloc(sizeof(data_t));
    all_data->h_interp = malloc(sizeof(data_t));

    if (all_data->eta == NULL || all_data->u == NULL || 
        all_data->v == NULL || all_data->h_interp == NULL) {
        fprintf(stderr, "Erreur d'allocation pour les structures de donnees\n");
        free_all_data(all_data);
        return NULL;
    }

    // 6. Initialiser les données
    if (init_data(all_data->eta, local_nx, local_ny, param->dx, param->dy, 0., 1) ||
        init_data(all_data->u, local_nx + 1, local_ny, param->dx, param->dy, 0., 1) ||
        init_data(all_data->v, local_nx, local_ny + 1, param->dx, param->dy, 0., 1) ||
        init_data(all_data->h_interp, local_nx, local_ny, param->dx, param->dy, 0., 0)) {
        fprintf(stderr, "Erreur lors de l'initialisation des donnees\n");
        free_all_data(all_data);
        return NULL;
    }

    return all_data;
}

int initialize_mpi_topology(int argc, char **argv, MPITopology *topo) {
    int err;
    char error_string[MPI_MAX_ERROR_STRING];
    int length;
    
    printf("Demarrage de l'initialisation MPI...\n");
    fflush(stdout);  // Force l'affichage
    
    // Initialisation des paramètres
    int periods[2] = {0, 0};
    int reorder = 1;
    topo->dims[0] = 0;
    topo->dims[1] = 0;
    
    // Initialisation MPI avec gestion d'erreur amelioree
    if ((err = MPI_Init(&argc, &argv)) != MPI_SUCCESS) {
        MPI_Error_string(err, error_string, &length);
        fprintf(stderr, "Erreur : echec de l'initialisation MPI - %s\n", error_string);
        fflush(stderr);
        return 1;
    }
    printf("MPI_Init reussi\n");
    fflush(stdout);
    
    // Obtention de la taille
    if ((err = MPI_Comm_size(MPI_COMM_WORLD, &topo->nb_process)) != MPI_SUCCESS) {
        MPI_Error_string(err, error_string, &length);
        fprintf(stderr, "Erreur : Impossible d'obtenir le nombre de processus - %s\n", error_string);
        fflush(stderr);
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }
    printf("Nombre de processus: %d\n", topo->nb_process);
    fflush(stdout);
    
    // Obtention du rang
    if ((err = MPI_Comm_rank(MPI_COMM_WORLD, &topo->rank)) != MPI_SUCCESS) {
        MPI_Error_string(err, error_string, &length);
        fprintf(stderr, "Erreur : Impossible d'obtenir le rang du processus - %s\n", error_string);
        fflush(stderr);
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }
    printf("Process rank: %d\n", topo->rank);
    fflush(stdout);
    
    // Creation de la topologie cartesienne 2D
    if ((err = MPI_Dims_create(topo->nb_process, 2, topo->dims)) != MPI_SUCCESS) {
        MPI_Error_string(err, error_string, &length);
        fprintf(stderr, "Erreur : echec de la creation des dimensions - %s\n", error_string);
        fflush(stderr);
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }
    printf("Process %d: Dimensions creees: [%d, %d]\n", topo->rank, topo->dims[0], topo->dims[1]);
    fflush(stdout);
    
    // Verification des dimensions
    if (topo->dims[0] * topo->dims[1] != topo->nb_process) {
        fprintf(stderr, "Process %d: Erreur dimensions (%d x %d) != nb_process (%d)\n",
                topo->rank, topo->dims[0], topo->dims[1], topo->nb_process);
        fflush(stderr);
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }
    
    // Creation du communicateur cartesien
    if ((err = MPI_Cart_create(MPI_COMM_WORLD, 2, topo->dims, periods, reorder, &topo->cart_comm)) != MPI_SUCCESS) {
        MPI_Error_string(err, error_string, &length);
        fprintf(stderr, "Process %d: Erreur creation communicateur cartesien - %s\n", topo->rank, error_string);
        fflush(stderr);
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }
    printf("Process %d: Communicateur cartesien cree\n", topo->rank);
    fflush(stdout);
    
    // Obtention du rang cartesien
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
    
    // Obtention des coordonnees cartesiennes
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
    
    // Determination des voisins
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
    
    // Barrière pour synchroniser tous les processus
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
                               int nx, int ny,
                               double dx, double dy) {

    // Allocation pour les trois types de données (eta, u, v)
    gdata->recv_size_eta = malloc(sizeof(int) * topo->nb_process);
    gdata->recv_size_u = malloc(sizeof(int) * topo->nb_process);
    gdata->recv_size_v = malloc(sizeof(int) * topo->nb_process);
    gdata->displacements_eta = malloc(sizeof(int) * topo->nb_process);
    gdata->displacements_u = malloc(sizeof(int) * topo->nb_process);
    gdata->displacements_v = malloc(sizeof(int) * topo->nb_process);

    if (gdata->recv_size_eta == NULL || gdata->recv_size_u == NULL || gdata->recv_size_v == NULL ||
        gdata->displacements_eta == NULL || gdata->displacements_u == NULL || gdata->displacements_v == NULL) {
        fprintf(stderr, "Rank %d: Error allocating recv_sizes or displacements\n", topo->cart_rank);
        return 1;
    }

    // Initialisation spécifique pour le rang maître (cart_rank 0)
    if (topo->cart_rank == 0) {
        int rank_coords[2];
        
        // Calcul des dimensions locales pour chaque processus
        int local_nx = nx / topo->dims[0];
        int local_ny = ny / topo->dims[1];
        
        // Allocations pour le rang maître
        gdata->gathered_output = malloc(sizeof(data_t));
        gdata->rank_glob = malloc(sizeof(limit_t*) * topo->nb_process);

        if (gdata->gathered_output == NULL || gdata->rank_glob == NULL) {
            fprintf(stderr, "Error when allocating memory for master rank\n");
            return 1;
        }

        // Pour chaque rang
        for (int r = 0; r < topo->nb_process; r++) {
            gdata->rank_glob[r] = malloc(2 * sizeof(limit_t));
            if (gdata->rank_glob[r] == NULL) {
                fprintf(stderr, "Error when allocating memory for rank limits\n");
                return 1;
            }

            MPI_Cart_coords(topo->cart_comm, r, 2, rank_coords);

            // Calcul des tailles pour chaque rang
            // Pour eta: dimensions régulières
            gdata->recv_size_eta[r] = local_nx * local_ny;

            // Pour u: une colonne de plus pour les processus à droite
            gdata->recv_size_u[r] = (local_nx + (rank_coords[0] == topo->dims[0]-1 ? 1 : 1)) * local_ny;

            // Pour v: une ligne de plus pour les processus en haut
            gdata->recv_size_v[r] = local_nx * (local_ny + (rank_coords[1] == topo->dims[1]-1 ? 1 : 1));

            // Calcul des déplacements
            gdata->displacements_eta[r] = (r == 0) ? 0 : gdata->displacements_eta[r-1] + gdata->recv_size_eta[r-1];
            gdata->displacements_u[r] = (r == 0) ? 0 : gdata->displacements_u[r-1] + gdata->recv_size_u[r-1];
            gdata->displacements_v[r] = (r == 0) ? 0 : gdata->displacements_v[r-1] + gdata->recv_size_v[r-1];

            // Stockage des limites pour reconstruction
            gdata->rank_glob[r][0].start = local_nx * rank_coords[0];
            gdata->rank_glob[r][0].end = local_nx * (rank_coords[0] + 1) - 1;
            gdata->rank_glob[r][0].n = local_nx;

            gdata->rank_glob[r][1].start = local_ny * rank_coords[1];
            gdata->rank_glob[r][1].end = local_ny * (rank_coords[1] + 1) - 1;
            gdata->rank_glob[r][1].n = local_ny;
        }

        // Calcul des tailles totales et allocation des buffers de réception
        int total_size_eta = gdata->displacements_eta[topo->nb_process-1] + gdata->recv_size_eta[topo->nb_process-1];
        int total_size_u = gdata->displacements_u[topo->nb_process-1] + gdata->recv_size_u[topo->nb_process-1];
        int total_size_v = gdata->displacements_v[topo->nb_process-1] + gdata->recv_size_v[topo->nb_process-1];

        gdata->receive_data_eta = malloc(sizeof(double) * total_size_eta);
        gdata->receive_data_u = malloc(sizeof(double) * total_size_u);
        gdata->receive_data_v = malloc(sizeof(double) * total_size_v);

        if (gdata->receive_data_eta == NULL || 
            gdata->receive_data_u == NULL || 
            gdata->receive_data_v == NULL) {
            fprintf(stderr, "Error when allocating receive buffers\n");
            return 1;
        }

        // Debug
        printf("Tailles et déplacements initialisés:\n");
        for (int r = 0; r < topo->nb_process; r++) {
            printf("Rank %d - eta: size=%d, disp=%d\n", r, gdata->recv_size_eta[r], gdata->displacements_eta[r]);
            printf("Rank %d - u: size=%d, disp=%d\n", r, gdata->recv_size_u[r], gdata->displacements_u[r]);
            printf("Rank %d - v: size=%d, disp=%d\n", r, gdata->recv_size_v[r], gdata->displacements_v[r]);
        }
        printf("Tailles totales des buffers de reception:\n");
        printf("eta: %d\n", total_size_eta);
        printf("u: %d\n", total_size_u);
        printf("v: %d\n", total_size_v);
    }

    // Broadcast des données nécessaires à tous les rangs
    MPI_Bcast(gdata->recv_size_eta, topo->nb_process, MPI_INT, 0, topo->cart_comm);
    MPI_Bcast(gdata->recv_size_u, topo->nb_process, MPI_INT, 0, topo->cart_comm);
    MPI_Bcast(gdata->recv_size_v, topo->nb_process, MPI_INT, 0, topo->cart_comm);
    MPI_Bcast(gdata->displacements_eta, topo->nb_process, MPI_INT, 0, topo->cart_comm);
    MPI_Bcast(gdata->displacements_u, topo->nb_process, MPI_INT, 0, topo->cart_comm);
    MPI_Bcast(gdata->displacements_v, topo->nb_process, MPI_INT, 0, topo->cart_comm);

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

