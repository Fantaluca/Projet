#include "shallow_mpi.h"
#include <stdlib.h> 


#undef MPI_STATUSES_IGNORE
#define MPI_STATUSES_IGNORE (MPI_Status *)0


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
		MPI_Abort(MPI_COMM_WORLD, 1);
		return 1;
    }

    parameters_t param;
    if (read_parameters(&param, argv[1])) {
        MPI_Abort(topo.cart_comm, 1); 
        return 1;
    }
    if (topo.cart_rank == 0) print_parameters(&param);

    all_data_t* all_data = init_all_data(&param, &topo);
    if (all_data == NULL) {
        fprintf(stderr, "Failed to initialize all_data\n");
        MPI_Abort(topo.cart_comm, 1); 
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
        MPI_Abort(topo.cart_comm, 1);           
        return 1;
    }
    memset(gdata, 0, sizeof(gather_data_t));

    if(initialize_gather_structures(&topo, gdata, nx_glob, ny_glob, param.dx, param.dy)) {
        fprintf(stderr, "Rank %d: Failed to initialize gather structures\n", topo.cart_rank);
        cleanup(&param, &topo, gdata); 
        free_all_data(all_data);
        cleanup_mpi_topology(&topo);
        MPI_Abort(topo.cart_comm, 1);
        return 1;
    }

    //----------------------//
    // Simulation algorithm //
    //----------------------//

    // Interpolate bathymetry
    interp_bathy(param, nx_glob, ny_glob, all_data, gdata, &topo);
	  check_cfl(param, all_data, &topo);

    // Loop over timestep
    double start = GET_TIME(); 
    for (int n = 0; n < nt; n++) {

		if (param.sampling_rate && !(n % param.sampling_rate)) 
			gather_and_assemble_data(param, all_data, gdata, &topo, nx_glob, ny_glob, n);

		if (topo.cart_rank == 0 && param.sampling_rate && !(n % param.sampling_rate)){
			write_data_vtk((gdata->gathered_output), "water elevation", param.output_eta_filename, n);
			
		}

		boundary_conditions(param, all_data, &topo);
		apply_source(n, nx_glob, ny_glob, param, all_data, gdata, &topo);
		
		update_eta(param, all_data, gdata, &topo);
		update_velocities(param, all_data, gdata, &topo);

		if (topo.rank ==0) print_progress(n, nt, start, &topo);

    }

  if (topo.rank ==0){
  double time = GET_TIME() - start;
    printf("\nDone: %g seconds (%g MUpdates/s)\n", time,
           1e-6 * (double)nx_glob * (double)ny_glob * (double)nt / time);
  }
        
  // Clean up all variables
  MPI_Barrier(topo.cart_comm);
  free_all_data(all_data);
  cleanup(&param, &topo, gdata);
  cleanup_mpi_topology(&topo);

  MPI_Finalize();

  return 0;
}
