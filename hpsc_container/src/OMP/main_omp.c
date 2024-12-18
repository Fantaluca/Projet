#include "shallow_omp.h"


int main(int argc, char **argv) {
    if(argc != 2) {
        printf("Usage: %s parameter_file\n", argv[0]);
        return 1;
    }

    // Initialize parameters and h
    parameters_t param;
    if(read_parameters(&param, argv[1])) return 1;
    print_parameters(&param);

    all_data_t *all_data = init_all_data(&param);
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

    printf(" - grid size: %g m x %g m (%d x %d = %d grid points)\n",
           hx, hy, nx, ny, nx * ny);
    printf(" - number of time steps: %d\n", nt);


    // Interpolate bathymetry
    interp_bathy(nx, ny, param, all_data);

    // Loop over timestep
    double start = GET_TIME();
    for(int n = 0; n < nt; n++) {
       
        // output solution
        if(param.sampling_rate && !(n % param.sampling_rate)) 
            write_data_vtk(all_data->eta, "water elevation", param.output_eta_filename, n);

        boundary_conditions(nx, ny, param, all_data);
        apply_source(n, nx, ny, param, all_data);

        update_eta(nx, ny, param, all_data);
        update_velocities(nx, ny, param, all_data);

        print_progress(n, nt, start);
    }

    write_manifest_vtk(param.output_eta_filename, param.dt, nt, param.sampling_rate);

    double time = GET_TIME() - start;
    printf("\nDone: %g seconds (%g MUpdates/s)\n", time,
           1e-6 * (double)all_data->eta->nx * (double)all_data->eta->ny * (double)nt / time);

    free_all_data(all_data);


    return 0;
}