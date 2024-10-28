#include "shallow_mpi.h"

#undef MPI_STATUSES_IGNORE
#define MPI_STATUSES_IGNORE (MPI_Status *)0 // temporary to get rid of "warning: 'MPI_Waitall' accessing 20 bytes in a region of size 0 [-Wstringop-overflow=]"


void cleanup(all_data_t *all_data, parameters_t *param, MPITopology *topo, gather_data_t *gdata) {

    // Libération de la structure all_data
    if (all_data != NULL) {
        // Libérer u, v, eta avec leurs edge_vals
        free_data(all_data->u, 1);
        free_data(all_data->v, 1);
        free_data(all_data->eta, 1);
        
        // Libérer h et h_interp qui n'ont pas d'edge_vals
        free_data(all_data->h, 0);
        free_data(all_data->h_interp, 0);
        
        free(all_data);
    }

    // Libération des structures MPI (seulement pour le processus principal)
    if (topo != NULL && topo->cart_rank == 0 && gdata != NULL) {
        // Libérer gathered_output
        if (gdata->gathered_output != NULL) {
            free(gdata->gathered_output->vals);
            free(gdata->gathered_output);
        }
        
        // Libérer les autres données de gather
        free(gdata->receive_data);
        
        if (gdata->rank_glob != NULL) {
            for (int r = 0; r < topo->nb_process; r++) {
                free(gdata->rank_glob[r]);
            }
            free(gdata->rank_glob);
        }
        
        free(gdata->recv_size);
        free(gdata->displacements);
    }

    // Note: parameters_t n'est pas libéré car il n'est pas alloué dynamiquement
}


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


  int i_rank = i - START_I(&gdata, topo->cart_rank); 
  int j_rank = j - START_J(&gdata, topo->cart_rank); 
  int nx = RANK_NX(&gdata, topo->cart_rank);
  int ny = RANK_NY(&gdata, topo->cart_rank); 

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
                      
  int i_rank = i - START_I(&gdata, topo->cart_rank); 
  int j_rank = j - START_J(&gdata, topo->cart_rank); 
  int nx = RANK_NX(&gdata, topo->cart_rank);
  int ny = RANK_NY(&gdata, topo->cart_rank); 

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
  int nx = RANK_NX(&gdata, topo->cart_rank);
  int i_start = START_I(&gdata, topo->cart_rank);
  int i_end = END_I(&gdata, topo->cart_rank);

  int ny = RANK_NY(&gdata, topo->cart_rank);
  int j_start = START_J(&gdata, topo->cart_rank);
  int j_end = END_J(&gdata, topo->cart_rank);


  // Receive u(i+1,j) and v(i,j+1) from neighbouring processes
  MPI_Irecv((*all_data)->u->edge_vals[RIGHT], ny, MPI_DOUBLE, direction[RIGHT], 100, topo->cart_comm, &request_recv[0]);
  MPI_Irecv((*all_data)->v->edge_vals[UP], nx, MPI_DOUBLE, direction[UP], 101, topo->cart_comm, &request_recv[1]);


  // Prepare data to send
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
    int nx = RANK_NX(&gdata, topo->cart_rank);
    int i_start = START_I(&gdata, topo->cart_rank);
    int i_end = END_I(&gdata, topo->cart_rank);

    int ny = RANK_NY(&gdata, topo->cart_rank);
    int j_start = START_J(&gdata, topo->cart_rank);
    int j_end = END_J(&gdata, topo->cart_rank);


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


void free_all_data(all_data_t* all_data) {
    if (all_data != NULL) {
        free(all_data->u);
        free(all_data->v);
        free(all_data->eta);
        free(all_data->h);
        free(all_data->h_interp);
        free(all_data);
    }
}

all_data_t* init_all_data(const parameters_t *param) {

    all_data_t* all_data = malloc(sizeof(all_data_t));
    if (all_data == NULL) {
        fprintf(stderr, "Erreur d'allocation pour all_data\n");
        return NULL;
    }

    all_data->h = malloc(sizeof(data_t));
    if (all_data->h == NULL || read_data(all_data->h, param->input_h_filename)) {
        fprintf(stderr, "Erreur lors de la lecture des données bathymétriques\n");
        free_all_data(all_data);
        return NULL;
    }


    all_data->eta = malloc(sizeof(data_t));
    all_data->u = malloc(sizeof(data_t));
    all_data->v = malloc(sizeof(data_t));
    all_data->h_interp = malloc(sizeof(data_t));

    if (all_data->eta == NULL || all_data->u == NULL || 
        all_data->v == NULL || all_data->h_interp == NULL) {
        fprintf(stderr, "Erreur d'allocation pour les structures de données\n");
        free_all_data(all_data);
        return NULL;
    }

    double hx = all_data->h->nx * all_data->h->dx;
    double hy = all_data->h->ny * all_data->h->dy;
    int nx = floor(hx / param->dx);
    int ny = floor(hy / param->dy);

    init_data(all_data->eta, nx, ny, param->dx, param->dy, 0.);
    init_data(all_data->u, nx + 1, ny, param->dx, param->dy, 0.);
    init_data(all_data->v, nx, ny + 1, param->dx, param->dy, 0.);
    init_data(all_data->h_interp, nx, ny, param->dx, param->dy, 0.);

    return all_data;
}

int initialize_mpi_topology(int argc, char **argv, MPITopology *topo) {
    // Initialisation des paramètres
    int periods[2] = {0, 0};
    int reorder = 1;
    topo->dims[0] = 0;
    topo->dims[1] = 0;
    
    // Initialisation MPI
    if (MPI_Init(&argc, &argv) != MPI_SUCCESS) {
        fprintf(stderr, "Erreur : Échec de l'initialisation MPI\n");
        return 1;
    }
    
    // Obtention de la taille et du rang
    if (MPI_Comm_size(MPI_COMM_WORLD, &topo->nb_process) != MPI_SUCCESS) {
        fprintf(stderr, "Erreur : Impossible d'obtenir le nombre de processus\n");
        return 1;
    }
    
    if (MPI_Comm_rank(MPI_COMM_WORLD, &topo->rank) != MPI_SUCCESS) {
        fprintf(stderr, "Erreur : Impossible d'obtenir le rang du processus\n");
        return 1;
    }
    
    // Création de la topologie cartésienne 2D
    if (MPI_Dims_create(topo->nb_process, 2, topo->dims) != MPI_SUCCESS) {
        fprintf(stderr, "Erreur : Échec de la création des dimensions\n");
        return 1;
    }
    
    if (MPI_Cart_create(MPI_COMM_WORLD, 2, topo->dims, periods, reorder, &topo->cart_comm) != MPI_SUCCESS) {
        fprintf(stderr, "Erreur : Échec de la création du communicateur cartésien\n");
        return 1;
    }
    
    if (MPI_Comm_rank(topo->cart_comm, &topo->cart_rank) != MPI_SUCCESS) {
        fprintf(stderr, "Erreur : Impossible d'obtenir le rang cartésien\n");
        return 1;
    }
    
    // Obtention des coordonnées cartésiennes
    if (MPI_Cart_coords(topo->cart_comm, topo->cart_rank, 2, topo->coords) != MPI_SUCCESS) {
        fprintf(stderr, "Erreur : Impossible d'obtenir les coordonnées cartésiennes\n");
        return 1;
    }
    
    // Détermination des voisins
    if (MPI_Cart_shift(topo->cart_comm, 0, 1, &topo->neighbors[LEFT], &topo->neighbors[RIGHT]) != MPI_SUCCESS) {
        fprintf(stderr, "Erreur : Impossible d'obtenir les voisins gauche/droite\n");
        return 1;
    }
    
    if (MPI_Cart_shift(topo->cart_comm, 1, 1, &topo->neighbors[DOWN], &topo->neighbors[UP]) != MPI_SUCCESS) {
        fprintf(stderr, "Erreur : Impossible d'obtenir les voisins haut/bas\n");
        return 1;
    }
    
    return 0; // Succès
}
int initialize_gather_structures(const MPITopology *topo, 
                                 gather_data_t *gdata,
                                 int nx, int ny, 
                                 double dx, double dy) {

    // Allocation initiale pour tous les rangs
    gdata->recv_size = malloc(sizeof(int) * topo->nb_process);
    gdata->displacements = malloc(sizeof(int) * topo->nb_process);
    
    if (gdata->recv_size == NULL || gdata->displacements == NULL) {
        fprintf(stderr, "Rank %d: Error allocating recv_size or displacements\n", topo->cart_rank);
        return 1;
    }

    // Initialisation spécifique pour le rang maître (cart_rank 0)
    if (topo->cart_rank == 0) {
        int rank_coords[2];
        int num[2] = {nx, ny};
        
        // Allocations pour le rang maître
        gdata->gathered_output = malloc(sizeof(data_t));
        gdata->receive_data = malloc(sizeof(double) * nx * ny);
        gdata->rank_glob = malloc(sizeof(limit_t*) * topo->nb_process);
        
        if (gdata->gathered_output == NULL || 
            gdata->receive_data == NULL || 
            gdata->rank_glob == NULL) {
            fprintf(stderr, "Error when allocating memory for master rank\n");
            return 1;
        }
        
        // Initialisation de gathered_output
        gdata->gathered_output->vals = gdata->receive_data;
        gdata->gathered_output->nx = nx;
        gdata->gathered_output->ny = ny;
        gdata->gathered_output->dx = dx;
        gdata->gathered_output->dy = dy;
        gdata->gathered_output->edge_vals = NULL;
        
        // Allocation et initialisation pour chaque rang
        for (int r = 0; r < topo->nb_process; r++) {
            gdata->rank_glob[r] = malloc(2 * sizeof(limit_t));
            if (gdata->rank_glob[r] == NULL) {
                fprintf(stderr, "Error when allocating memory for rank limits\n");
                return 1;
            }
            
            MPI_Cart_coords(topo->cart_comm, r, 2, rank_coords);
            gdata->recv_size[r] = 1;
            
            for (int i = 0; i < 2; i++) {
                gdata->rank_glob[r][i].start = num[i] * rank_coords[i] / topo->dims[i];
                gdata->rank_glob[r][i].end = num[i] * (rank_coords[i] + 1) / topo->dims[i] - 1;
                gdata->rank_glob[r][i].n = (gdata->rank_glob[r][i].end - gdata->rank_glob[r][i].start + 1);
                
                gdata->recv_size[r] *= gdata->rank_glob[r][i].n;
            }
            
            // Calcul des déplacements
            gdata->displacements[r] = (r == 0) ? 0 : gdata->displacements[r - 1] + gdata->recv_size[r - 1];
        }
    }

    // Broadcast des données nécessaires à tous les rangs
    MPI_Bcast(gdata->recv_size, topo->nb_process, MPI_INT, 0, topo->cart_comm);
    MPI_Bcast(gdata->displacements, topo->nb_process, MPI_INT, 0, topo->cart_comm);
    
    return 0;
}

