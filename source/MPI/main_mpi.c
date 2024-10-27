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

    // Allocate params
    struct parameters param;
    if (read_parameters(&param, argv[1])) return 1;

    // Allocate all_data
    all_data_t* all_data = init_all_data(&param);
    if (all_data == NULL) return 1;

    // Allocate bathymetry
    if (read_data(all_data->h, param.input_h_filename)) return 1;
    if (topo.cart_rank == 0) print_parameters(&param);

    // Infer size of domain from input bathymetric data
    double hx = all_data->h->nx * all_data->h->dx;
    double hy = all_data->h->ny * all_data->h->dy;
    int nx = floor(hx / param.dx);
    int ny = floor(hy / param.dy);
    if(nx <= 0) nx = 1;
    if(ny <= 0) ny = 1;
    int nt = floor(param.max_t / param.dt);


    //------------------------------------------------------//
    // INITIALIZE COMMUNICATING VARIABLES FOR MPI PROCESSES //
    //------------------------------------------------------//
    neighbour_t direction[NEIGHBOR_NUM];
    gather_data_t *gdata = {0};  
    if(initialize_gather_structures(&topo, &gdata, nx, ny, param.dx, param.dy)) {
        MPI_Abort(topo.cart_comm, MPI_ERR_NO_MEM);
        return 1;
    }

    double start = GET_TIME(); 
    // Loop over timestep
    for (int n = 0; n < nt; n++) {

      // print elapsed time
      if (n && (n % (nt / 10)) == 0 && topo.cart_rank == 0) {
        double time_sofar = GET_TIME() - start;
        double eta = (nt - n) * time_sofar / n;
        printf("Computing step %d/%d (ETA: %g seconds) \r", n, nt, eta);
        fflush(stdout);
      }

      // impose boundary conditions
      boundary_source_condition(n, nx, ny, param, &all_data);

      // Gather all output from processes
      if (param.sampling_rate && !(n % param.sampling_rate)) {

        int send_size = RANK_NX(&gdata, topo.cart_rank) * RANK_NY(&gdata, topo.cart_rank);
        data_t *output_data[] = {all_data->eta, all_data->u, all_data->v};
        const char *output_files[] = {param.output_eta_filename, param.output_u_filename, param.output_v_filename};
        
        for (int i = 0; i < 3; i++) {

          
          /*
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
        
        */
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
      // free_all_data(all_data_t* all_data)

    if (topo.cart_rank == 0) {
        printf("End MPI_finalized\n");
        printf("finished\n");
    }

  return 0;
}
