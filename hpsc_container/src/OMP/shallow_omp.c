/*===========================================================
 * SHALLOW WATER EQUATIONS SOLVER - SERIAL VERSION
 * Implementation File
 * Contains core computation routines and algorithms
 ===========================================================*/

#include "shallow_omp.h"

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

/**
 * Interpolates bathymetry data onto computation grid
 * 
 * @param nx, ny Grid dimensions
 * @param param Simulation parameters
 * @param all_data Data structures containing fields
 */
void interp_bathy(int nx, int ny, const parameters_t param, all_data_t *all_data) {
    #pragma omp parallel for
    for(int i = 0; i < nx; i++) {
        for(int j = 0; j < ny; j++) {
            double x = i * param.dx;
            double y = j * param.dy;
            double val = interpolate_data(all_data->h, x, y);
            SET(all_data->h_interp, i, j, val);
        }
    }
}

/*===========================================================
 * MAIN COMPUTATION FUNCTIONS
 ===========================================================*/

/**
 * Updates water height (eta) using shallow water equations
 * 
 * @param nx, ny Grid dimensions
 * @param param Simulation parameters
 * @param all_data Data structures containing fields
 */
void update_eta(int nx, int ny, const parameters_t param, all_data_t *all_data) {
    #pragma omp parallel for
    for (int i = 0; i < nx; i++) {
        for (int j = 0; j < ny; j++) {
            // Get bathymetry values with boundary handling
            double h_ui_plus_1_j = (i < nx - 1) ? GET(all_data->h_interp, i + 1, j) : GET(all_data->h_interp, i, j);
            double h_ui_j = GET(all_data->h_interp, i, j);
            double h_vi_j_plus_1 = (j < ny - 1) ? GET(all_data->h_interp, i, j + 1) : GET(all_data->h_interp, i, j);
            double h_vi_j = GET(all_data->h_interp, i, j);

            // Get velocity values with boundary handling
            double u_ip1_j = (i < nx - 1) ? GET(all_data->u, i + 1, j) : GET(all_data->u, i, j);
            double u_i_j = GET(all_data->u, i, j);
            double v_i_jp1 = (j < ny - 1) ? GET(all_data->v, i, j + 1) : GET(all_data->v, i, j);
            double v_i_j = GET(all_data->v, i, j);

            // Compute spatial derivatives
            double c1_x = param.dt / param.dx;
            double c1_y = param.dt / param.dy;

            // Update eta value
            double eta_ij = GET(all_data->eta, i, j)
                - c1_x * (h_ui_plus_1_j * u_ip1_j - h_ui_j * u_i_j)
                - c1_y * (h_vi_j_plus_1 * v_i_jp1 - h_vi_j * v_i_j);

            SET(all_data->eta, i, j, eta_ij);
        }
    }
}

/**
 * Updates velocity fields (u,v) using shallow water equations
 * 
 * @param nx, ny Grid dimensions
 * @param param Simulation parameters
 * @param all_data Data structures containing fields
 */
void update_velocities(int nx, int ny, const parameters_t param, all_data_t *all_data) {
    #pragma omp parallel for
    for(int i = 0; i < nx; i++) {
        for(int j = 0; j < ny; j++) {
            // Compute coefficients
            double c1 = param.dt * param.g;
            double c2 = param.dt * param.gamma;

            // Get eta values with boundary handling
            double eta_ij = GET(all_data->eta, i, j);
            double eta_imj = GET(all_data->eta, (i == 0) ? 0 : i - 1, j);
            double eta_ijm = GET(all_data->eta, i, (j == 0) ? 0 : j - 1);

            // Update velocities
            double u_ij = (1. - c2) * GET(all_data->u, i, j)
                - c1 / param.dx * (eta_ij - eta_imj);
            double v_ij = (1. - c2) * GET(all_data->v, i, j)
                - c1 / param.dy * (eta_ij - eta_ijm);

            SET(all_data->u, i, j, u_ij);
            SET(all_data->v, i, j, v_ij);
        }
    }
}

/*===========================================================
 * BOUNDARY CONDITIONS AND SOURCE TERMS
 ===========================================================*/

/**
 * Apply boundary conditions to velocity fields
 * Sets appropriate values at domain boundaries
 * 
 * @param nx, ny Grid dimensions
 * @param param Simulation parameters
 * @param all_data Data structures containing fields
 */
void boundary_conditions(int nx, int ny, const parameters_t param, all_data_t *all_data) {
    // Apply boundary condition on velocities
    #pragma omp parallel for
    for (int j = 0; j < ny; j++) {
        // Left and right boundaries for u
        SET(all_data->u, 0, j, 0.0);
        SET(all_data->u, nx, j, 0.0);
    }

    #pragma omp parallel for
    for (int i = 0; i < nx; i++) {
        // Top and bottom boundaries for v
        SET(all_data->v, i, 0, 0.0);
        SET(all_data->v, i, ny, 0.0);
    }
}

/**
 * Apply source terms according to simulation type
 * 
 * @param timestep Current simulation timestep
 * @param nx, ny Grid dimensions
 * @param param Simulation parameters
 * @param all_data Data structures containing fields
 */
void apply_source(int timestep, int nx, int ny, const parameters_t param, all_data_t *all_data) {
    double t = timestep * param.dt;
    const double A = 5.0;        // Amplitude 
    const double f = 1.0 / 20.0; // Frequency
    double source = A * sin(2.0 * M_PI * f * t);
    
    switch(param.source_type) {
        case 1: {  // Top boundary wave maker
            #pragma omp parallel for
            for(int i = 0; i < nx; i++) {
                double x_pos = i * param.dx;
                double spatial_mod = sin(2.0 * M_PI * x_pos / (nx * param.dx) * 2);
                SET(all_data->v, i, ny, source * (1.0 + 0.3 * spatial_mod));
            }
            break;
        }
        
        case 2: {  // Central point source
            SET(all_data->eta, nx / 2, ny / 2, source);
            break;
        }

        default:
            printf("Error: Unknown source type %d\n", param.source_type);
            exit(1);
    }
}
