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
    parameters_t param;
    if (read_parameters(&param, argv[1])) return 1;

    if (topo.cart_rank == 0) print_parameters(&param);

    // Allocate all_data avec une seule allocation
    all_data_t* all_data = init_all_data(&param, &topo);
    if (all_data == NULL) {
        fprintf(stderr, "Failed to initialize all_data\n");
        return 1;
    }

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

    // Allocation de gdata
    gather_data_t *gdata = malloc(sizeof(gather_data_t));
    if (gdata == NULL) {
        fprintf(stderr, "Rank %d: Failed to allocate gdata\n", topo.cart_rank);
        free_all_data(all_data);
        cleanup_mpi_topology(&topo);  // Ajouté
        MPI_Finalize();              // Ajouté
        return 1;
    }
    memset(gdata, 0, sizeof(gather_data_t));
    MPI_Barrier(topo.cart_comm);

    // Initialisation des structures gather
    if(initialize_gather_structures(&topo, gdata, nx, ny, param.dx, param.dy)) {
        fprintf(stderr, "Rank %d: Failed to initialize gather structures\n", topo.cart_rank);
        cleanup(&param, &topo, gdata);  // Utiliser cleanup au lieu de free individuels
        free_all_data(all_data);
        cleanup_mpi_topology(&topo);
        MPI_Finalize();
        return 1;
    }

    // Pour le test, initialisons les données avec des valeurs simples
    int send_size_eta = all_data->eta->nx * all_data->eta->ny;
    for(int j = 0; j < all_data->eta->ny; j++) {
        for(int i = 0; i < all_data->eta->nx; i++) {
            int idx = j * all_data->eta->nx + i;
            all_data->eta->vals[idx] = topo.cart_rank + 1.0;
        }
    }
    int send_size_u = all_data->u->nx * all_data->u->ny;
    for(int j = 0; j < all_data->u->ny; j++) {
        for(int i = 0; i < all_data->u->nx; i++) {
            int idx = j * all_data->u->nx + i;
            all_data->u->vals[idx] = topo.cart_rank + 2.0;
        }
    }

    int send_size_v = all_data->v->nx * all_data->v->ny;
    for(int j = 0; j < all_data->v->ny; j++) {
        for(int i = 0; i < all_data->v->nx; i++) {
            int idx = j * all_data->v->nx + i;
            all_data->v->vals[idx] = topo.cart_rank + 3.0;
        }
    }

    // Debug pour vérifier les dimensions
    if (topo.cart_rank == 0) {
        printf("Dimensions locales pour rank %d:\n", topo.cart_rank);
        printf("eta: %d x %d (total: %d)\n", all_data->eta->nx, all_data->eta->ny, send_size_eta);
        printf("u  : %d x %d (total: %d)\n", all_data->u->nx, all_data->u->ny, send_size_u);
        printf("v  : %d x %d (total: %d)\n", all_data->v->nx, all_data->v->ny, send_size_v);
        fflush(stdout);
    }
        
    double start = GET_TIME(); 
    // Loop over timestep
    int test_iterations = 3;  // Réduit pour le test
    for (int n = 0; n < test_iterations; n++) {

      /*
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

        int send_size = RANK_NX(gdata, topo.cart_rank) * RANK_NY(gdata, topo.cart_rank);
        data_t *output_data[] = {all_data->eta, all_data->u, all_data->v};
        const char *output_files[] = {param.output_eta_filename, param.output_u_filename, param.output_v_filename};
        
        for (int i = 0; i < 3; i++) {


          int gatherv_result = MPI_Gatherv(output_data[i]->vals, send_size, MPI_DOUBLE, 
                                          gdata->receive_data, gdata->recv_size, gdata->displacements, MPI_DOUBLE, 
                                          0, topo.cart_comm);
          
          
          if (gatherv_result != MPI_SUCCESS) {
              char error_string[MPI_MAX_ERROR_STRING];
              int length_of_error_string;
              MPI_Error_string(gatherv_result, error_string, &length_of_error_string);
              printf("Rank %d: MPI_Gatherv failed: %s\n", topo.cart_rank, error_string);
              MPI_Abort(topo.cart_comm, gatherv_result);
          }
          
   
          if (topo.cart_rank == 0) {
            for (int r = 0; r < topo.nb_process; r++) {
              for (int j = 0; j < RANK_NY(gdata, r); j++) {
                for (int i = 0; i < RANK_NX(gdata, r); i++) {
                  int global_i = START_I(gdata, r) + i;
                  int global_j = START_J(gdata, r) + j;
                  int local_index = j * RANK_NX(gdata, r) + i;
                  int global_index = global_j * nx + global_i;

                  gdata->gathered_output->vals[global_index] = 
                  gdata->receive_data[gdata->displacements[r] + local_index];
                }
              }
            }
            //write_data_vtk(&all_data->eta, "water elevation", param.output_eta_filename, n);
            //write_data_vtk(&u, "x velocity", param.output_u_filename, n);
            //write_data_vtk(&v, "y velocity", param.output_v_filename, n);
          }
        }
        //printf("Exiting sampling block\n");
      }

      // Update variables
      //printf("Before update_eta\n");
      //update_eta(param, &all_data, gdata, &topo, direction);
      //printf("After update_eta\n");
      
      //printf("Before update_velocities\n");
      //update_velocities(param, &all_data, gdata, &topo, direction);
      //printf("After update_velocities\n");
      
      //printf("Iteration %d end\n", n);

      */

      if (topo.cart_rank == 0) {
            printf("Test iteration %d/%d\n", n+1, test_iterations);
        }

      // Synchronisation explicite avant le gather
      MPI_Barrier(topo.cart_comm);

      // Test gather
      double* receive_buffers[] = {gdata->receive_data_eta,
                                   gdata->receive_data_u,
                                   gdata->receive_data_v};

      int* recv_sizes[] = {gdata->recv_size_eta,
                           gdata->recv_size_u,
                           gdata->recv_size_v};

      int* displacements[] = {gdata->displacements_eta,
                              gdata->displacements_u,
                              gdata->displacements_v};

        // Les données à envoyer
        data_t *output_data[] = {all_data->eta, all_data->u, all_data->v};

        // Gather pour chaque champ
        for (int i = 0; i < 3; i++) {
            int send_size = output_data[i]->nx * output_data[i]->ny;
            
            if (topo.cart_rank == 0) {
                printf("Starting gather for field %d with send_size=%d\n", i, send_size);
                fflush(stdout);
            }

            int gatherv_result = MPI_Gatherv(
                output_data[i]->vals,        // données à envoyer
                send_size,                   // nombre d'éléments à envoyer
                MPI_DOUBLE,                  // type des données
                receive_buffers[i],          // buffer de réception spécifique au champ
                recv_sizes[i],              // tailles de réception spécifiques au champ
                displacements[i],           // déplacements spécifiques au champ
                MPI_DOUBLE,                  // type des données
                0,                          // rang root
                topo.cart_comm              // communicateur
            );

            if (gatherv_result != MPI_SUCCESS) {
                char error_string[MPI_MAX_ERROR_STRING];
                int length_of_error_string;
                MPI_Error_string(gatherv_result, error_string, &length_of_error_string);
                fprintf(stderr, "Rank %d: MPI_Gatherv failed for field %d: %s\n", 
                        topo.cart_rank, i, error_string);
                fflush(stderr);
                MPI_Abort(topo.cart_comm, gatherv_result);
                return 1;
            }

            if (topo.cart_rank == 0) {
                printf("Gather %d completed successfully\n", i);
                printf("Field %d sizes - send: %d, total receive: %d\n", 
                      i, send_size, recv_sizes[i][topo.nb_process-1] + displacements[i][topo.nb_process-1]);
                fflush(stdout);
            }
        }

        // Synchronisation après le gather
        MPI_Barrier(topo.cart_comm);

      }
    
  // Synchronisation avant de commencer le nettoyage
  MPI_Barrier(topo.cart_comm);

  if (topo.cart_rank == 0) {
      printf("Starting cleanup sequence...\n");
      fflush(stdout);
  }

  // 1. D'abord libérer les données qui ne dépendent pas de MPI
  free_all_data(all_data);
  all_data = NULL;  // Éviter tout accès ultérieur

  MPI_Barrier(topo.cart_comm);  // S'assurer que tous les processus ont libéré leurs données

  if (topo.cart_rank == 0) {
      printf("free_all_data completed\n");
      fflush(stdout);
  }

  // 2. Ensuite nettoyer les structures gather
  cleanup(&param, &topo, gdata);
  gdata = NULL;  // Éviter tout accès ultérieur

  MPI_Barrier(topo.cart_comm);  // S'assurer que tous les processus ont terminé cleanup

  if (topo.cart_rank == 0) {
      printf("Cleanup completed\n");
      fflush(stdout);
  }

  // 3. Enfin, nettoyer la topologie MPI
  // Important : s'assurer que plus aucune communication MPI ne sera nécessaire après ce point
  MPI_Barrier(topo.cart_comm);  // Dernière synchronisation avant de libérer le communicateur
  cleanup_mpi_topology(&topo);

  // Ne pas utiliser cart_rank après ce point car le communicateur a été libéré
  // Utiliser MPI_COMM_WORLD si nécessaire pour les derniers messages

  int world_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  if (world_rank == 0) {
      printf("cleanup_mpi_topology completed\n");
      fflush(stdout);
  }

  // Synchronisation finale sur COMM_WORLD avant MPI_Finalize
  MPI_Barrier(MPI_COMM_WORLD);

  MPI_Finalize();
  return 0;
}
