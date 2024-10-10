#include <mpi.h>
#include <math.h>
#include "shallow.h"

// ... other includes and function definitions ...


typedef struct {
    int start;
    int end;
    int n;
} limit_t;

typedef struct {
    double *data;
    int nx, ny;
    double dx, dy;
} data_t;

int main(int argc, char **argv) {

    // Initialize paramters and h
    struct parameters param;
    if(read_parameters(&param, argv[1])) return 1;
    print_parameters(&param);

    struct data h;
    if(read_data(&h, param.input_h_filename)) return 1;


    // Infer size of domain from input bathymetric data
    double hx = h.nx * h.dx;
    double hy = h.ny * h.dy;
    double dx = param.dx;
    double dy = param.dy;
    int nx = floor(hx / param.dx);
    int ny = floor(hy / param.dy);
    if(nx <= 0) nx = 1;
    if(ny <= 0) ny = 1;
    int nt = floor(param.max_t / param.dt);

    int world_size, rank, cart_rank;
    int dims[2] = {0, 0};
    int periods[2] = {0, 0};
    int reorder = 1;
    MPI_Comm cart_comm;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Create 2D Cartesian communicator
    MPI_Dims_create(world_size, 2, dims);
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, reorder, &cart_comm);
    MPI_Comm_rank(cart_comm, &cart_rank);

    int coords[2];
    MPI_Cart_coords(cart_comm, cart_rank, 2, coords);

    int neighbors[4];
    MPI_Cart_shift(cart_comm, 0, 1, &neighbors[LEFT], &neighbors[RIGHT]);
    MPI_Cart_shift(cart_comm, 1, 1, &neighbors[UP], &neighbors[DOWN]);

    data_t *gathered_output;
    double *receive_data;
    limit_t **rank_limit;
    int *recv_size, *displacements;

    if (cart_rank == 0) {
        int rank_coords[2];
        int num[2] = {nx, ny};

        if ((gathered_output = malloc(sizeof(data_t))) == NULL ||
            (gathered_output->data = malloc(sizeof(double) * nx * ny)) == NULL ||
            (displacements = malloc(sizeof(int) * world_size)) == NULL ||
            (receive_data = malloc(sizeof(double) * nx * ny)) == NULL ||
            (rank_limit = malloc(sizeof(limit_t*) * world_size)) == NULL ||
            (recv_size = malloc(sizeof(int) * world_size)) == NULL) {
            printf("Error when allocating memory");
            MPI_Abort(cart_comm, MPI_ERR_NO_MEM);
        }

        int total_size = 0;
        for (int r = 0; r < world_size; r++) {
            if ((rank_limit[r] = malloc(2 * sizeof(limit_t))) == NULL) {
                printf("Error when allocating memory");
                MPI_Abort(cart_comm, MPI_ERR_NO_MEM);
            }

            MPI_Cart_coords(cart_comm, r, 2, rank_coords);
            recv_size[r] = 1;
            for (int i = 0; i < 2; i++) {
                rank_limit[r][i].start = num[i] * rank_coords[i] / dims[i];
                rank_limit[r][i].end = num[i] * (rank_coords[i] + 1) / dims[i] - 1;
                rank_limit[r][i].n = (rank_limit[r][i].end - rank_limit[r][i].start + 1);

                recv_size[r] *= rank_limit[r][i].n;
            }

            total_size += recv_size[r];
            if (r == 0) {
                displacements[r] = 0;
            } else {
                displacements[r] = displacements[r - 1] + recv_size[r - 1];
            }
        }
    }

    // Broadcast the dimensions to all processes
    MPI_Bcast(&nx, 1, MPI_INT, 0, cart_comm);
    MPI_Bcast(&ny, 1, MPI_INT, 0, cart_comm);

    // Calculate local domain size
    limit_t local_limit[2];
    for (int i = 0; i < 2; i++) {
        local_limit[i].start = nx * coords[i] / dims[i];
        local_limit[i].end = nx * (coords[i] + 1) / dims[i] - 1;
        local_limit[i].n = local_limit[i].end - local_limit[i].start + 1;
    }

    int local_nx = local_limit[0].n;
    int local_ny = local_limit[1].n;

    // Initialize local data structures
    data_t local_data;
    local_data.nx = local_nx;
    local_data.ny = local_ny;
    local_data.dx = dx;
    local_data.dy = dy;
    local_data.data = malloc(sizeof(double) * local_nx * local_ny);

    // Initialize local data (placeholder)
    for (int i = 0; i < local_nx * local_ny; i++) {
        local_data.data[i] = 0.0;
    }

    // Clean up
    if (cart_rank == 0) {
        free(gathered_output->data);
        free(gathered_output);
        free(displacements);
        free(receive_data);
        for (int r = 0; r < world_size; r++) {
            free(rank_limit[r]);
        }
        free(rank_limit);
        free(recv_size);
    }
    free(local_data.data);

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