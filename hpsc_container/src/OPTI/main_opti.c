#include "shallow_opti.h"

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

    //----------------------//
    // Simulation algorithm //
    //----------------------//

    // Interpolate bathymetry
    interp_bathy(param, nx_glob, ny_glob, all_data, gdata, &topo);

// Vérification des tailles avant le transfert
printf("Sizes check before transfer:\n");
printf("h total_size: %zu\n", all_data->h->total_size);
printf("h_interp total_size: %zu\n", all_data->h_interp->total_size);
printf("eta total_size: %zu\n", all_data->eta->total_size);
printf("u total_size: %zu\n", all_data->u->total_size);
printf("v total_size: %zu\n", all_data->v->total_size);

// Vérification des pointeurs
printf("Pointer check before transfer:\n");
printf("h vals: %p\n", (void*)all_data->h->vals);
printf("h_interp vals: %p\n", (void*)all_data->h_interp->vals);
printf("eta vals: %p\n", (void*)all_data->eta->vals);
printf("u vals: %p\n", (void*)all_data->u->vals);
printf("v vals: %p\n", (void*)all_data->v->vals);

// Vérification des allocations
if (!all_data->h->vals || !all_data->h_interp->vals || 
    !all_data->eta->vals || !all_data->u->vals || !all_data->v->vals) {
    fprintf(stderr, "Error: One or more arrays not properly allocated\n");
    cleanup(&param, &topo, gdata);
    free_all_data(all_data);
    cleanup_mpi_topology(&topo);
    MPI_Finalize();
    return 1;
}

if (all_data->h->nx * all_data->h->ny != all_data->h->total_size) {
    fprintf(stderr, "Error: h dimensions mismatch: nx=%d, ny=%d, total_size=%zu\n",
            all_data->h->nx, all_data->h->ny, all_data->h->total_size);
    return 1;
}


// Calculer les tailles avant le transfert
size_t h_size = all_data->h->total_size;
size_t h_interp_size = all_data->h_interp->total_size;
size_t eta_size = all_data->eta->total_size;
size_t u_size = all_data->u->total_size;
size_t v_size = all_data->v->total_size;

// Vérifier que les tailles sont valides
if (h_size == 0 || h_interp_size == 0 || eta_size == 0 || 
    u_size == 0 || v_size == 0) {
    fprintf(stderr, "Error: Invalid sizes detected\n");
    return 1;
}

#pragma omp target enter data \
    map(to: all_data->h_interp->vals[0:h_interp_size], \
        all_data->eta->vals[0:eta_size], \
        all_data->u->vals[0:u_size], \
        all_data->v->vals[0:v_size])

printf("Starting main loop with dimensions:\n");
printf("nx=%d, ny=%d, nt=%d\n", nx_glob, ny_glob, nt);
printf("Memory requirements:\n");
printf("Total memory for h: %zu bytes\n", h_size * sizeof(double));
printf("Total memory for eta: %zu bytes\n", eta_size * sizeof(double));

    // Loop over timestep
    double start = GET_TIME(); 
    for (int n = 0; n < nt; n++) {
      
      boundary_conditions(param , all_data, &topo);
      apply_source(n, nx_glob, ny_glob, param, all_data, gdata, &topo);

      if (param.sampling_rate && !(n % param.sampling_rate)) 
        gather_and_assemble_data(param, all_data, gdata, &topo, nx_glob, ny_glob, n);

        // output solution
        if(param.sampling_rate && !(n % param.sampling_rate)){
            #pragma omp target update from(all_data->eta->vals[0:all_data->eta->total_size], \
                              all_data->u->vals[0:all_data->u->total_size], \
                              all_data->v->vals[0:all_data->v->total_size])
            write_data_vtk(&all_data->eta, "water elevation", param.output_eta_filename, n);
        }
      
      update_eta(param, all_data, gdata, &topo);
      update_velocities(param, all_data, gdata, &topo);

      print_progress(n, nt, start, &topo);
      
    }

  if (topo.rank ==0){
  double time = GET_TIME() - start;
    printf("\nDone: %g seconds (%g MUpdates/s)\n", time,
           1e-6 * (double)nx_glob * (double)ny_glob * (double)nt / time);
  }

  #pragma omp target exit data \
    map(from: all_data->eta->vals[0:all_data->eta->total_size], \
        all_data->u->vals[0:all_data->u->total_size], \
        all_data->v->vals[0:all_data->v->total_size]) \
    map(release: all_data->h_interp->vals[0:all_data->h_interp->total_size])
      
  // Clean up all variables
  MPI_Barrier(topo.cart_comm);
  free_all_data(all_data);
  cleanup(&param, &topo, gdata);
  cleanup_mpi_topology(&topo);

  MPI_Finalize();

  return 0;
}
