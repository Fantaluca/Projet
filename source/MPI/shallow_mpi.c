#include "shallow_mpi.h"

#undef MPI_STATUSES_IGNORE
#define MPI_STATUSES_IGNORE (MPI_Status *)0 // temporary to get rid of "warning: 'MPI_Waitall' accessing 20 bytes in a region of size 0 [-Wstringop-overflow=]"


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

    // If neighbors[DIR] == NULL : global boundary -> 0.0 value
    if (topo->neighbors[RIGHT] != MPI_PROC_NULL) recv_buffer_u = calloc(ny, sizeof(double));
    if (topo->neighbors[UP] != MPI_PROC_NULL) recv_buffer_v = calloc(nx, sizeof(double));

    // Receive processes' neighbours (RIGHT/UP)
    int recv_count = 0;
    if (topo->neighbors[RIGHT] != MPI_PROC_NULL) 
        MPI_Irecv(recv_buffer_u, ny, MPI_DOUBLE, topo->neighbors[RIGHT], 100,
                  topo->cart_comm, &request_recv[recv_count++]);
    
    if (topo->neighbors[UP] != MPI_PROC_NULL)
        MPI_Irecv(recv_buffer_v, nx, MPI_DOUBLE, topo->neighbors[UP], 101, 
                  topo->cart_comm, &request_recv[recv_count++]);
    
    // Send processes' neighbours (LEFT/DOWN)
    for (int j = 0; j < ny; j++) send_u[j] = GET((*all_data)->u, 0, j);
    for (int i = 0; i < nx; i++) send_v[i] = GET((*all_data)->v, i, ny-1);

    int send_count = 0;
    if (topo->neighbors[LEFT] != MPI_PROC_NULL) 
        MPI_Isend(send_u, ny, MPI_DOUBLE, topo->neighbors[LEFT], 100,
                  topo->cart_comm, &request_send[send_count++]);
    
    if (topo->neighbors[DOWN] != MPI_PROC_NULL) 
        MPI_Isend(send_v, nx, MPI_DOUBLE, topo->neighbors[DOWN], 101,
                  topo->cart_comm, &request_send[send_count++]);
    
    MPI_Waitall(recv_count, request_recv, status);
    
    // Update eta
    for (int j = 0; j < ny; j++) {
        for (int i = 0; i < nx; i++) {
            double h_ij = GET((*all_data)->h_interp, i, j);
            if (h_ij <= 0) continue;

            double u_i = GET((*all_data)->u, i, j);
            double u_ip1 = (i < nx-1) ? GET((*all_data)->u, i+1, j) : 
                          (recv_buffer_u ? recv_buffer_u[j] : 0.0);
            
            double v_j = GET((*all_data)->v, i, j);
            double v_jp1 = (j < ny-1) ? GET((*all_data)->v, i, j+1) : 
                          (recv_buffer_v ? recv_buffer_v[i] : 0.0);

            double du_dx = (u_ip1 - u_i) / param.dx;
            double dv_dy = (v_jp1 - v_j) / param.dy;

            double eta_old = GET((*all_data)->eta, i, j);
            double eta_new = eta_old - param.dt * h_ij * (du_dx + dv_dy);
            SET((*all_data)->eta, i, j, eta_new);

        }
    }

    MPI_Waitall(send_count, request_send, status);

    // Clean buffers
    free(send_u);
    free(send_v);
    free(recv_buffer_u);
    free(recv_buffer_v);
}

void update_velocities(const parameters_t param,
                      all_data_t **all_data,
                      gather_data_t *gdata,
                      MPITopology *topo) {

    MPI_Request request_recv[2] = {MPI_REQUEST_NULL, MPI_REQUEST_NULL};
    MPI_Request request_send[2] = {MPI_REQUEST_NULL, MPI_REQUEST_NULL};
    MPI_Status status[2];

    int nx = (*all_data)->eta->nx;
    int ny = (*all_data)->eta->ny;
    
    // Allocate buffers
    double *send_eta_right = calloc(ny, sizeof(double));
    double *send_eta_up = calloc(nx, sizeof(double));
    double *recv_buffer_left = NULL;
    double *recv_buffer_down = NULL;

    // If neighbors[DIR] == NULL : global boundary -> 0.0 value
    if (topo->neighbors[LEFT] != MPI_PROC_NULL) recv_buffer_left = calloc(ny, sizeof(double));
    if (topo->neighbors[DOWN] != MPI_PROC_NULL) recv_buffer_down = calloc(nx, sizeof(double));
       
    for (int j = 0; j < ny; j++) send_eta_right[j] = GET((*all_data)->eta, nx-1, j);
    for (int i = 0; i < nx; i++) send_eta_up[i] = GET((*all_data)->eta, i, ny-1);
    
    // Receive processes' neighbours (LEFT/DOWN)
    int recv_count = 0;
    if (topo->neighbors[LEFT] != MPI_PROC_NULL)  
        MPI_Irecv(recv_buffer_left, ny, MPI_DOUBLE, topo->neighbors[LEFT], 200,
                  topo->cart_comm, &request_recv[recv_count++]);
    
    if (topo->neighbors[DOWN] != MPI_PROC_NULL) 
        MPI_Irecv(recv_buffer_down, nx, MPI_DOUBLE, topo->neighbors[DOWN], 201, 
                  topo->cart_comm, &request_recv[recv_count++]);
    
    // Send processes' neighbours (RIGHT/UP)
    int send_count = 0;
    if (topo->neighbors[RIGHT] != MPI_PROC_NULL) 
        MPI_Isend(send_eta_right, ny, MPI_DOUBLE, topo->neighbors[RIGHT], 200,
                  topo->cart_comm, &request_send[send_count++]);
    

    if (topo->neighbors[UP] != MPI_PROC_NULL) 
        MPI_Isend(send_eta_up, nx, MPI_DOUBLE, topo->neighbors[UP], 201,
                  topo->cart_comm, &request_send[send_count++]);
    
    MPI_Waitall(recv_count, request_recv, status);
    
    // Update velocities
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

            double u_ij = GET((*all_data)->u, i, j);
            double v_ij = GET((*all_data)->v, i, j);

            double new_u = (1.0 - c2) * u_ij - c1 / dx * (eta_ij - eta_imj);
            double new_v = (1.0 - c2) * v_ij - c1 / dy * (eta_ij - eta_ijm);

            SET((*all_data)->u, i, j, new_u);
            SET((*all_data)->v, i, j, new_v);
        }
    }

    MPI_Waitall(send_count, request_send, status);
    
    // Cleanup buffers
    free(send_eta_right);
    free(send_eta_up);
    free(recv_buffer_left);
    free(recv_buffer_down);
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
    

    int start_i = START_I(gdata, topo->cart_rank);
    int start_j = START_J(gdata, topo->cart_rank);
    int local_nx = (*all_data)->h_interp->nx;
    int local_ny = (*all_data)->h_interp->ny;


    #pragma omp parallel for collapse(2)
    for(int i = 0; i < local_nx; i++) {
        for(int j = 0; j < local_ny; j++) {

            double x = (i + start_i) * param.dx;
            double y = (j + start_j) * param.dy;
            double val = interpolate_data((*all_data)->h, nx_glob, ny_glob, x, y);
            SET((*all_data)->h_interp, i, j, val);
        }
    }
}

void boundary_source_condition(int n, int nx_glob, int ny_glob,
                             const parameters_t param,
                             all_data_t **all_data,
                             gather_data_t *gdata,
                             MPITopology *topo) {
    
    double t = n * param.dt;
    
    if (param.source_type == 1) {
        double A = 5.0;
        double f = 1.0 / 20.0;
        
        if (topo->neighbors[LEFT] == MPI_PROC_NULL) {
            for (int j = 0; j < (*all_data)->u->ny; j++) SET((*all_data)->u, 0, j, 0.0);
            
        }
        if (topo->neighbors[RIGHT] == MPI_PROC_NULL) {
            for (int j = 0; j < (*all_data)->u->ny; j++) SET((*all_data)->u, (*all_data)->u->nx-1, j, 0.0);
        }
        if (topo->neighbors[DOWN] == MPI_PROC_NULL) {
            for (int i = 0; i < (*all_data)->v->nx; i++) SET((*all_data)->v, i, 0, 0.0);
            
        }
        if (topo->neighbors[UP] == MPI_PROC_NULL) {
            double source = A * sin(2.0 * M_PI * f * t);
            for (int i = 0; i < (*all_data)->v->nx; i++) SET((*all_data)->v, i, (*all_data)->v->ny-1, source);
        }
    }
    else if (param.source_type == 2) {

        int global_middle_i = nx_glob / 2;
        int global_middle_j = ny_glob / 2;
        
        int local_i = global_middle_i - START_I(gdata, topo->cart_rank);
        int local_j = global_middle_j - START_J(gdata, topo->cart_rank);
        
        if (local_i >= 0 && local_i < (*all_data)->eta->nx && 
            local_j >= 0 && local_j < (*all_data)->eta->ny) {
            
            double A = 5.0;
            double f = 1.0 / 20.0;
            double source = A * sin(2.0 * M_PI * f * t);
            SET((*all_data)->eta, local_i, local_j, source);
        }
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
  
    memset(gdata, 0, sizeof(gather_data_t));
    gdata->recv_size_eta = calloc(topo->nb_process, sizeof(int));
    gdata->recv_size_u = calloc(topo->nb_process, sizeof(int));
    gdata->recv_size_v = calloc(topo->nb_process, sizeof(int));
    gdata->displacements_eta = calloc(topo->nb_process, sizeof(int));
    gdata->displacements_u = calloc(topo->nb_process, sizeof(int));
    gdata->displacements_v = calloc(topo->nb_process, sizeof(int));
    
    if (!gdata->recv_size_eta || !gdata->recv_size_u || !gdata->recv_size_v ||
        !gdata->displacements_eta || !gdata->displacements_u || !gdata->displacements_v) {
        fprintf(stderr, "Rank %d: Failed to allocate size arrays\n", topo->cart_rank);
        fflush(stdout);
        return 1;
    }

    gdata->rank_glob = calloc(topo->nb_process, sizeof(limit_t*));
    if (!gdata->rank_glob) {
        fprintf(stderr, "Rank %d: Failed to allocate rank_glob\n", topo->cart_rank);
        fflush(stdout);
        return 1;
    }

    for (int r = 0; r < topo->nb_process; r++) {
        gdata->rank_glob[r] = calloc(2, sizeof(limit_t));
        if (!gdata->rank_glob[r]) {
            fprintf(stderr, "Rank %d: Failed to allocate rank_glob[%d]\n", topo->cart_rank, r);
            fflush(stdout);
            return 1;
        }
    }

    int base_nx = nx_glob / topo->dims[0];
    int base_ny = ny_glob / topo->dims[1];
    int remainder_x = nx_glob % topo->dims[0];
    int remainder_y = ny_glob % topo->dims[1];

    if (topo->cart_rank == 0) {
        int total_offset = 0;
        
        // Important: browse processes in the order of the cartesian grid
        for (int j = 0; j < topo->dims[1]; j++) {
            for (int i = 0; i < topo->dims[0]; i++) {
                int coords[2] = {i, j};
                int rank;
                MPI_Cart_rank(topo->cart_comm, coords, &rank);

                int local_nx = base_nx + (i < remainder_x ? 1 : 0);
                int local_ny = base_ny + (j < remainder_y ? 1 : 0);

                // X direction 
                gdata->rank_glob[rank][0].start = i * base_nx + (i < remainder_x ? i : remainder_x);
                gdata->rank_glob[rank][0].n = local_nx;
                gdata->rank_glob[rank][0].end = gdata->rank_glob[rank][0].start + local_nx;

                // Y direction
                gdata->rank_glob[rank][1].start = j * base_ny + (j < remainder_y ? j : remainder_y);
                gdata->rank_glob[rank][1].n = local_ny;
                gdata->rank_glob[rank][1].end = gdata->rank_glob[rank][1].start + local_ny;

                // Size for MPI_Gatherv
                gdata->recv_size_eta[rank] = local_nx * local_ny;
                gdata->recv_size_u[rank] = (local_nx + 1) * local_ny;
                gdata->recv_size_v[rank] = local_nx * (local_ny + 1);
                
                gdata->displacements_eta[rank] = total_offset;
                gdata->displacements_u[rank] = total_offset;
                gdata->displacements_v[rank] = total_offset;
                
                total_offset += gdata->recv_size_eta[rank];
            }
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

        for (int i = 0; i < 3; i++) {
            gdata->gathered_output[i].nx = nx_glob;
            gdata->gathered_output[i].ny = ny_glob;
            gdata->gathered_output[i].dx = dx;
            gdata->gathered_output[i].dy = dy;
            gdata->gathered_output[i].vals = calloc(nx_glob * ny_glob, sizeof(double));
            if (!gdata->gathered_output[i].vals) return 1;
        }
    }

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
    
    // Structures de données pour le gathering
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

    // Gather et assemblage pour chaque champ
    for (int field = 0; field < 3; field++) {
        // Gathering des données
        int send_size = output_data[field]->nx * output_data[field]->ny;
        MPI_Gatherv(output_data[field]->vals, send_size, MPI_DOUBLE,
                    receive_buffers[field], recv_sizes[field],
                    displacements[field], MPI_DOUBLE, 0, topo->cart_comm);

        // Assemblage sur le rang 0
        if (topo->cart_rank == 0) {
            const int nx_gathered = nx_glob;
            const int ny_gathered = ny_glob;

            // Réinitialisation du tableau de sortie
            memset(gdata->gathered_output[field].vals, 0,
                  nx_gathered * ny_gathered * sizeof(double));

            // Assemblage des données de chaque processus
            for (int r = 0; r < topo->nb_process; r++) {
                int coords[2];
                MPI_Cart_coords(topo->cart_comm, r, 2, coords);
                
                int local_nx = RANK_NX(gdata, r);
                int local_ny = RANK_NY(gdata, r);
                int global_start_i = START_I(gdata, r);
                int global_start_j = START_J(gdata, r);

                // Copie des données locales dans la position globale
                for (int j = 0; j < local_ny; j++) {
                    for (int i = 0; i < local_nx; i++) {
                        int global_i = global_start_i + i;
                        int global_j = global_start_j + j;

                        int src_idx = j * local_nx + i;
                        int src_with_displacement = displacements[field][r] + src_idx;
                        int dst_idx = global_j * nx_gathered + global_i;

                        gdata->gathered_output[field].vals[dst_idx] =
                            receive_buffers[field][src_with_displacement];
                    }
                }
            }

            // Mise à jour des métadonnées
            gdata->gathered_output[field].nx = nx_gathered;
            gdata->gathered_output[field].ny = ny_gathered;
            gdata->gathered_output[field].dx = param.dx;
            gdata->gathered_output[field].dy = param.dy;
        }
    }

    // Écriture des données sur le rang 0
    if (topo->cart_rank == 0) {
        write_data_vtk(&(gdata->gathered_output), "water elevation",
                      param.output_eta_filename, timestep);
    }
}
