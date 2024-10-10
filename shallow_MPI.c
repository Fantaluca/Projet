#include <mpi.h>
#include <math.h>
#include "shallow.h"

// ... other includes and function definitions ...

int main(int argc, char **argv) {

    if(argc != 2){
        printf("Usage: %s parameter_file\n", argv[0]);
        return 1;
    }

    int rank, // actual rank process
    int size; // total nb of process
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // ... parameter initialization ...

    // Calculate 2D decomposition
    int dims[2] = {0, 0};
    MPI_Dims_create(size, 2, dims);
    int px = dims[0]; // Number of processes in x direction
    int py = dims[1]; // Number of processes in y direction

    // Create 2D Cartesian communicator
    MPI_Comm cart_comm;
    int periods[2] = {0, 0};
    int reorder = 1; //  MPI auto-organizes ranks of process (1) or not (0)
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, reorder, &cart_comm);

    // Get coordinates in the 2D process grid
    int coords[2];
    MPI_Cart_coords(cart_comm, rank, 2, coords);

    // Calculate local domain size
    int local_nx = nx / px;
    int local_ny = ny / py;
    int start_x = coords[0] * local_nx;
    int start_y = coords[1] * local_ny;
    int end_x = (coords[0] == px - 1) ? nx : start_x + local_nx;
    int end_y = (coords[1] == py - 1) ? ny : start_y + local_ny;
    local_nx = end_x - start_x;
    local_ny = end_y - start_y;

    int neighbors[4]; 
    MPI_Cart_shift(cart_comm, 0, 1, &neighbors[LEFT], &neighbors[RIGHT]);  
    MPI_Cart_shift(cart_comm, 1, 1, &neighbors[UP], &neighbors[DOWN]);  

    // Initialize local data structures
    struct data local_eta, local_u, local_v, local_h_interp;
    init_data(&local_eta, local_nx, local_ny, param.dx, param.dy, 0.);
    init_data(&local_u, local_nx + 1, local_ny, param.dx, param.dy, 0.);
    init_data(&local_v, local_nx, local_ny + 1, param.dx, param.dy, 0.);
    init_data(&local_h_interp, local_nx, local_ny, param.dx, param.dy, 0.);

    // ... interpolate bathymetry for local domain ...

    for(int n = 0; n < nt; n++){
        // ... timing and output code ...

        // Exchange ghost cells with neighbouring processes
        exchange_ghost_cells(&local_eta, &local_u, &local_v, cart_comm);

        // Update variables for local domain
        update_eta(local_nx, local_ny, param, &local_u, &local_v, &local_eta, &local_h_interp);
        update_velocities(local_nx, local_ny, param, &local_u, &local_v, &local_eta);

        // ... boundary conditions and other updates ...
    }

    // Gather results from all processes
    if (rank == 0){
        // Allocate memory for full domain data
        init_data(&eta, nx, ny, param.dx, param.dy, 0.);
        // ... similar for u and v ...
    }
    gather_data(&local_eta, &eta, local_nx, local_ny, nx, ny, cart_comm);

    // ... output results, free memory, etc. ...

    MPI_Finalize();
    return 0;
}

void exchange_ghost_cells(struct data *eta, struct data *u, struct data *v, MPI_Comm cart_comm){

    int rank, coords[2], nbr_coords[2];
    MPI_Comm_rank(cart_comm, &rank);
    MPI_Cart_coords(cart_comm, rank, 2, coords);

    // Exchange in x direction
    for (int dir = 0; dir < 2; dir++) {
        nbr_coords[0] = coords[0] + (dir * 2 - 1);
        nbr_coords[1] = coords[1];
        int nbr_rank;
        MPI_Cart_rank(cart_comm, nbr_coords, &nbr_rank);

        // Exchange eta
        MPI_Sendrecv(&eta->data[dir * (eta->nx - 1)], eta->ny, MPI_DOUBLE, nbr_rank, 0,
                     &eta->data[(1 - dir) * (eta->nx - 1)], eta->ny, MPI_DOUBLE, nbr_rank, 0,
                     cart_comm, MPI_STATUS_IGNORE);

        // Similar exchanges for u and v...
    }

    // Exchange in y direction
    for (int dir = 0; dir < 2; dir++) {
        nbr_coords[0] = coords[0];
        nbr_coords[1] = coords[1] + (dir * 2 - 1);
        int nbr_rank;
        MPI_Cart_rank(cart_comm, nbr_coords, &nbr_rank);

        // Exchange eta
        MPI_Sendrecv(&eta->data[dir * (eta->ny - 1) * eta->nx], eta->nx, MPI_DOUBLE, nbr_rank, 0,
                     &eta->data[(1 - dir) * (eta->ny - 1) * eta->nx], eta->nx, MPI_DOUBLE, nbr_rank, 0,
                     cart_comm, MPI_STATUS_IGNORE);

        // Similar exchanges for u and v...
    }
}

void gather_data(struct data *local_data, struct data *global_data, 
                    int local_nx, int local_ny, int nx, int ny, MPI_Comm cart_comm) {
    int rank, size, coords[2], dims[2];
    MPI_Comm_rank(cart_comm, &rank);
    MPI_Comm_size(cart_comm, &size);
    MPI_Cart_get(cart_comm, 2, dims, NULL, coords);

    // Create derived datatype for local block
    MPI_Datatype block_type;
    int sizes[2] = {ny, nx};
    int subsizes[2] = {local_ny, local_nx};
    int starts[2] = {coords[1] * local_ny, coords[0] * local_nx};
    MPI_Type_create_subarray(2, sizes, subsizes, starts, MPI_ORDER_C, MPI_DOUBLE, &block_type);
    MPI_Type_commit(&block_type);

    // Gather data to rank 0
    MPI_Gather(local_data->data, local_nx * local_ny, MPI_DOUBLE,
               global_data->data, 1, block_type, 0, cart_comm);

    MPI_Type_free(&block_type);
}