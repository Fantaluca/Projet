/*===========================================================
 * SHALLOW WATER EQUATIONS SOLVER - SERIAL VERSION
 * Implementation File
 * Contains core computation routines and main program
 ===========================================================*/

#include "shallow_serial.h"

/*===========================================================
 * INTERPOLATION FUNCTIONS
 ===========================================================*/

/**
 * Performs bilinear interpolation of data at given coordinates
 * 
 * @param data Source data structure
 * @param x X-coordinate for interpolation
 * @param y Y-coordinate for interpolation
 * @return Interpolated value at (x,y)
 */
double interpolate_data(const data_t *data, double x, double y) {
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

    // Four values of data surrounding (i,j)
    double Q11 = GET(data, i, j);
    double Q12 = GET(data, i, j + 1);
    double Q21 = GET(data, i + 1, j);
    double Q22 = GET(data, i + 1, j + 1);

    // Weighted coefficients
    double wx = (x2 - x) / (x2 - x1);
    double wy = (y2 - y) / (y2 - y1);

    return wx * wy * Q11 +
           (1 - wx) * wy * Q21 +
           wx * (1 - wy) * Q12 +
           (1 - wx) * (1 - wy) * Q22;
}

/*===========================================================
 * MAIN COMPUTATION FUNCTIONS
 ===========================================================*/

/**
 * Updates water height (eta) using shallow water equations
 * 
 * @param nx, ny Grid dimensions
 * @param param Simulation parameters
 * @param u X-velocity field
 * @param v Y-velocity field
 * @param eta Water elevation field
 * @param h_interp Interpolated bathymetry field
 */
double update_eta(int nx, int ny, parameters_t param, 
                 data_t *u, data_t *v, data_t *eta, data_t *h_interp) {
    for (int i = 0; i < nx; i++) {
        for (int j = 0; j < ny; j++) {
            // Get bathymetry values with boundary handling
            double h_ui_plus_1_j = (i < nx - 1) ? GET(h_interp, i + 1, j) : GET(h_interp, i, j);
            double h_ui_j = GET(h_interp, i, j);
            double h_vi_j_plus_1 = (j < ny - 1) ? GET(h_interp, i, j + 1) : GET(h_interp, i, j);
            double h_vi_j = GET(h_interp, i, j);

            // Get velocity values with boundary handling
            double u_ip1_j = (i < nx - 1) ? GET(u, i + 1, j) : GET(u, i, j);
            double u_i_j = GET(u, i, j);
            double v_i_jp1 = (j < ny - 1) ? GET(v, i, j + 1) : GET(v, i, j);
            double v_i_j = GET(v, i, j);

            // Compute spatial derivatives
            double c1_x = param.dt / param.dx;
            double c1_y = param.dt / param.dy;

            // Update eta value
            double eta_ij = GET(eta, i, j)
                - c1_x * (h_ui_plus_1_j * u_ip1_j - h_ui_j * u_i_j)
                - c1_y * (h_vi_j_plus_1 * v_i_jp1 - h_vi_j * v_i_j);

            SET(eta, i, j, eta_ij);
        }
    }
}

/**
 * Updates velocity fields (u,v) using shallow water equations
 * 
 * @param nx, ny Grid dimensions
 * @param param Simulation parameters
 * @param u X-velocity field
 * @param v Y-velocity field
 * @param eta Water elevation field
 */
double update_velocities(int nx, int ny, parameters_t param,
                        data_t *u, data_t *v, data_t *eta) {
    for(int i = 0; i < nx; i++) {
        for(int j = 0; j < ny; j++) {
            // Compute coefficients
            double c1 = param.dt * param.g;
            double c2 = param.dt * param.gamma;

            // Get eta values with boundary handling
            double eta_ij = GET(eta, i, j);
            double eta_imj = GET(eta, (i == 0) ? 0 : i - 1, j);
            double eta_ijm = GET(eta, i, (j == 0) ? 0 : j - 1);

            // Update velocities
            double u_ij = (1. - c2) * GET(u, i, j)
                - c1 / param.dx * (eta_ij - eta_imj);
            double v_ij = (1. - c2) * GET(v, i, j)
                - c1 / param.dy * (eta_ij - eta_ijm);

            SET(u, i, j, u_ij);
            SET(v, i, j, v_ij);
        }
    }
}

/*===========================================================
 * BOUNDARY CONDITIONS AND SOURCE TERMS
 ===========================================================*/

/**
 * Apply boundary conditions and source terms
 * Sets appropriate values at domain boundaries and applies source terms
 * 
 * @param n Current timestep
 * @param nx, ny Grid dimensions
 * @param param Simulation parameters
 * @param u X-velocity field
 * @param v Y-velocity field
 * @param eta Water elevation field
 * @param h_interp Interpolated bathymetry field
 */
void boundary_condition(int n, int nx, int ny, parameters_t param,
                       data_t *u, data_t *v, data_t *eta, const data_t *h_interp) {
    double t = n * param.dt;

    if(param.source_type == 1) {
        // Sinusoidal velocity on top boundary
        double A = 5;
        double f = 1. / 20.;
        for(int i = 0; i < nx; i++) {
            for(int j = 0; j < ny; j++) {
                SET(u, 0, j, 0.);
                SET(u, nx, j, 0.);
                SET(v, i, 0, 0.);
                SET(v, i, ny, A * sin(2 * M_PI * f * t));
            }
        }
    }
    else if(param.source_type == 2) {
        // Sinusoidal elevation in the middle of the domain
        double A = 5;
        double f = 1. / 20.;
        SET(eta, nx / 2, ny / 2, A * sin(2 * M_PI * f * t));
    }
    else {
        printf("Error: Unknown source type %d\n", param.source_type);
        exit(0);
    }
}

/**
 * Interpolates bathymetry data onto computation grid
 * 
 * @param nx, ny Grid dimensions
 * @param param Simulation parameters
 * @param h_interp Output interpolated bathymetry field
 * @param h Input bathymetry field
 */
void interp_bathy(int nx, int ny, parameters_t param,
                  data_t *h_interp, data_t *h) {
    for(int i = 0; i < nx; i++) {
        for(int j = 0; j < ny; j++) {
            double x = i * param.dx;
            double y = j * param.dy;
            double val = interpolate_data(h, x, y);
            SET(h_interp, i, j, val);
        }
    }
}

/*===========================================================
 * MAIN PROGRAM
 ===========================================================*/

/**
 * Main program for shallow water equations solver
 * Handles initialization, time stepping, and cleanup
 * 
 * @param argc Number of command line arguments
 * @param argv Array of command line arguments
 * @return Exit status (0 for success)
 */
int main(int argc, char **argv) {
    if(argc != 2) {
        printf("Usage: %s parameter_file\n", argv[0]);
        return 1;
    }

    // Initialize parameters and bathymetry
    parameters_t param;
    if(read_parameters(&param, argv[1])) return 1;
    print_parameters(&param);

    data_t h;
    if(read_data(&h, param.input_h_filename)) return 1;

    // Infer size of domain from input bathymetric data
    double hx = h.nx * h.dx;
    double hy = h.ny * h.dy;
    int nx = floor(hx / param.dx);
    int ny = floor(hy / param.dy);
    if(nx <= 0) nx = 1;
    if(ny <= 0) ny = 1;
    int nt = floor(param.max_t / param.dt);

    printf(" - grid size: %g m x %g m (%d x %d = %d grid points)\n",
           hx, hy, nx, ny, nx * ny);
    printf(" - number of time steps: %d\n", nt);

    // Initialize variables
    data_t eta, u, v;
    init_data(&eta, nx, ny, param.dx, param.dy, 0.);
    init_data(&u, nx + 1, ny, param.dx, param.dy, 0.);
    init_data(&v, nx, ny + 1, param.dx, param.dy, 0.);

    // Interpolate bathymetry
    data_t h_interp;
    init_data(&h_interp, nx, ny, param.dx, param.dy, 0.);
    interp_bathy(nx, ny, param, &h_interp, &h);

    double start = GET_TIME();

    // Main time stepping loop
    for(int n = 0; n < nt; n++) {
        // Progress indicator
        if(n && (n % (nt / 10)) == 0) {
            double time_sofar = GET_TIME() - start;
            double eta = (nt - n) * time_sofar / n;
            printf("Computing step %d/%d (ETA: %g seconds)     \r", 
                   n, nt, eta);
            fflush(stdout);
        }

        // Output solution
        if(param.sampling_rate && !(n % param.sampling_rate)) {
            write_data_vtk(&eta, "water elevation", 
                          param.output_eta_filename, n);
        }

        // Impose boundary conditions
        boundary_condition(n, nx, ny, param, &u, &v, &eta, &h_interp);

        // Update variables
        update_eta(nx, ny, param, &u, &v, &eta, &h_interp);
        update_velocities(nx, ny, param, &u, &v, &eta);
    }

    // Write final output manifest
    write_manifest_vtk(param.output_eta_filename, param.dt, nt,
                      param.sampling_rate);

    // Print performance statistics
    double time = GET_TIME() - start;
    printf("\nDone: %g seconds (%g MUpdates/s)\n", time,
           1e-6 * (double)eta.nx * (double)eta.ny * (double)nt / time);

    // Cleanup
    free_data(&h_interp);
    free_data(&eta);
    free_data(&u);
    free_data(&v);

    return 0;
}