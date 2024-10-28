#include "shallow_MPI.h"

#undef MPI_STATUSES_IGNORE
#define MPI_STATUSES_IGNORE (MPI_Status *)0 // temporary to get rid of "warning: 'MPI_Waitall' accessing 20 bytes in a region of size 0 [-Wstringop-overflow=]"


int main(int argc, char **argv) {


    if (argc != 2) {
        printf("Usage: %s parameter_file\n", argv[0]);
        return 1;
    }

    //---------------------------------//
    // INITIALIZE MPI, PARAMS AND DATA //
    //---------------------------------//
    MPITopology topo;
    if (initialize_mpi_topology(argc, argv, &topo)) {
      MPI_Finalize();
      return 1;
    }

    parameters_t param;
    if (read_parameters(&param, argv[1])) return 1;
    if (topo.cart_rank == 0) print_parameters(&param);

    all_data_t* all_data = init_all_data(&param, &topo);
    if (all_data == NULL) {
        fprintf(stderr, "Failed to initialize all_data\n");
        return 1;
    }

    // Infer size of domain from input bathymetric data
    double hx = all_data->h->nx * all_data->h->dx;
    double hy = all_data->h->ny * all_data->h->dy;
    int nx_glob = floor(hx / param.dx);
    int ny_glob = floor(hy / param.dy);

    if(nx_glob <= 0) nx_glob = 1;
    if(ny_glob <= 0) ny_glob = 1;
    int nt = floor(param.max_t / param.dt);



    //------------------------------------------------------//
    // INITIALIZE COMMUNICATING VARIABLES FOR MPI PROCESSES //
    //------------------------------------------------------//
    gather_data_t *gdata = malloc(sizeof(gather_data_t));
    if (gdata == NULL) {
        fprintf(stderr, "Rank %d: Failed to allocate gdata\n", topo.cart_rank);
        free_all_data(all_data);
        cleanup_mpi_topology(&topo);  
        MPI_Finalize();              
        return 1;
    }
    memset(gdata, 0, sizeof(gather_data_t));
    MPI_Barrier(topo.cart_comm);

    if(initialize_gather_structures(&topo, gdata, nx_glob, ny_glob, param.dx, param.dy)) {
        fprintf(stderr, "Rank %d: Failed to initialize gather structures\n", topo.cart_rank);
        cleanup(&param, &topo, gdata); 
        free_all_data(all_data);
        cleanup_mpi_topology(&topo);
        MPI_Finalize();
        return 1;
    }

    // Interpolate bathymetry
    interp_bathy(param, nx_glob, ny_glob, &all_data, gdata, &topo);

        
    double start = GET_TIME(); 
    // Loop over timestep
    for (int n = 0; n < nt; n++) {

        boundary_source_condition(n, nx_glob, ny_glob, param, &all_data, gdata, &topo);
        update_eta(param, &all_data, gdata, &topo);
        update_velocities(param, &all_data, gdata, &topo);

        // Gather data from processes
        if (param.sampling_rate && !(n % param.sampling_rate)) {
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

            for (int field = 0; field < 3; field++) {

              int send_size = output_data[field]->nx * output_data[field]->ny;
              MPI_Gatherv(output_data[field]->vals, send_size, MPI_DOUBLE, receive_buffers[field],
                          recv_sizes[field], displacements[field], MPI_DOUBLE, 0, topo.cart_comm);


              // Reorganize data on rank 0
              if (topo.cart_rank == 0) {
                  const int nx_gathered = nx_glob;
                  const int ny_gathered = ny_glob;

                  // Reinit output array for this field
                  memset(gdata->gathered_output[field].vals, 0, 
                        nx_gathered * ny_gathered * sizeof(double));

                  for (int r = 0; r < topo.nb_process; r++) {

                      int coords[2];
                      MPI_Cart_coords(topo.cart_comm, r, 2, coords);
                      int local_nx = RANK_NX(gdata, r);
                      int local_ny = RANK_NY(gdata, r);
                      int global_start_i = START_I(gdata, r);
                      int global_start_j = START_J(gdata, r);

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

                  gdata->gathered_output[field].nx = nx_gathered;
                  gdata->gathered_output[field].ny = ny_gathered;
                  gdata->gathered_output[field].dx = param.dx;
                  gdata->gathered_output[field].dy = param.dy;
              }
          }

            if (topo.cart_rank == 0) { 
                write_data_vtk(&(gdata->gathered_output), "water elevation", 
                              param.output_eta_filename, n);
            }
        }
    }
    
  // Clean up all variables
  MPI_Barrier(topo.cart_comm);

  free_all_data(all_data);
  cleanup(&param, &topo, gdata);
  cleanup_mpi_topology(&topo);

  MPI_Finalize();

  return 0;
}
