/*===========================================================
 * SHALLOW WATER EQUATIONS SOLVER - PARALLEL GPU
 * Implementation File
 * Contains core computation routines for GPU execution
 ===========================================================*/

#include "shallow_gpu.h"

/*===========================================================
 * INTERPOLATION AND PREPROCESSING FUNCTIONS 
 ===========================================================*/

/**
 * Performs bilinear interpolation on structured grid data
 * 
 * @param data Source data structure
 * @param x, y Target coordinates
 * @return Interpolated value
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

    double val = wx * wy * Q11 +
                 (1 - wx) * wy * Q21 +
                 wx * (1 - wy) * Q12 +
                 (1 - wx) * (1 - wy) * Q22;

    return val;
}

void interp_bathy(int nx, int ny, const parameters_t param, all_data_t *all_data) {
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
 * CORE COMPUTATION FUNCTIONS
 ===========================================================*/

/**
 * Updates water elevation (eta) using GPU acceleration
 * 
 * @param nx, ny Grid dimensions
 * @param param Simulation parameters
 * @param all_data Simulation data structures
 */
void update_eta(int nx, int ny, const parameters_t param, all_data_t *all_data) {
    double* eta_gpu = all_data->eta->values;
    double* h_interp_gpu = all_data->h_interp->values;
    double* u_gpu = all_data->u->values;
    double* v_gpu = all_data->v->values;

    #pragma omp target teams distribute parallel for collapse(2) \
        map(tofrom: eta_gpu[0:nx*ny]) \
        map(to: h_interp_gpu[0:nx*ny], u_gpu[0:(nx+1)*ny], v_gpu[0:nx*(ny+1)])
    for(int i = 0; i < nx; i++) {
        for(int j = 0; j < ny; j++) {
            double h_ij = h_interp_gpu[nx * j + i];
            double c1 = param.dt * h_ij;
            
            // u on staggered grid (nx+1) x ny
            double u_i = u_gpu[(nx+1) * j + i];
            double u_ip1 = u_gpu[(nx+1) * j + (i+1)];
            
            // v on staggered grid nx x (ny+1)
            double v_j = v_gpu[nx * j + i];
            double v_jp1 = v_gpu[nx * (j+1) + i];
            
            eta_gpu[nx * j + i] = eta_gpu[nx * j + i] 
                - c1 * ((u_ip1 - u_i) / param.dx 
                     + (v_jp1 - v_j) / param.dy);
        }
    }
}

/**
 * Updates water velocities (u,v) using GPU acceleration
 * 
 * @param nx, ny Grid dimensions
 * @param param Simulation parameters
 * @param all_data Simulation data structures
 */
void update_velocities(int nx, int ny, const parameters_t param, all_data_t *all_data) {
    double* u_gpu = all_data->u->values;
    double* v_gpu = all_data->v->values;
    double* eta_gpu = all_data->eta->values;

    // For u: grid (nx+1) x ny
    #pragma omp target teams distribute parallel for collapse(2) \
        map(tofrom: u_gpu[0:(nx+1)*ny]) \
        map(to: eta_gpu[0:nx*ny])
    for(int i = 0; i < nx + 1; i++) {
        for(int j = 0; j < ny; j++) {
            double c1 = param.dt * param.g;
            double c2 = param.dt * param.gamma;
            
            double eta_ij, eta_im1j;
            
            if (i < nx) {
                eta_ij = eta_gpu[nx * j + i];
            } else {
                eta_ij = eta_gpu[nx * j + (nx-1)];  // last valid point
            }
            
            if (i > 0) {
                eta_im1j = eta_gpu[nx * j + (i-1)];
            } else {
                eta_im1j = eta_gpu[nx * j + 0];     // first valid point
            }
            
            double u_old = u_gpu[(nx+1) * j + i];
            double new_u = (1. - c2) * u_old 
                - c1 / param.dx * (eta_ij - eta_im1j);
            
            u_gpu[(nx+1) * j + i] = new_u;
        }
    }

    // For v: grid nx x (ny+1)
    #pragma omp target teams distribute parallel for collapse(2) \
        map(tofrom: v_gpu[0:nx*(ny+1)]) \
        map(to: eta_gpu[0:nx*ny])
    for(int i = 0; i < nx; i++) {
        for(int j = 0; j < ny + 1; j++) {
            double c1 = param.dt * param.g;
            double c2 = param.dt * param.gamma;
            
            double eta_ij, eta_ijm1;
            
            if (j < ny) {
                eta_ij = eta_gpu[nx * j + i];
            } else {
                eta_ij = eta_gpu[nx * (ny-1) + i];  // last valid point
            }
            
            if (j > 0) {
                eta_ijm1 = eta_gpu[nx * (j-1) + i];
            } else {
                eta_ijm1 = eta_gpu[nx * 0 + i];     // first valid point
            }
            
            double v_old = v_gpu[nx * j + i];
            double new_v = (1. - c2) * v_old
                - c1 / param.dy * (eta_ij - eta_ijm1);
            
            v_gpu[nx * j + i] = new_v;
        }
    }
}

/*===========================================================
 * BOUNDARY CONDITIONS AND SOURCE TERMS
 ===========================================================*/

/**
 * Applies boundary conditions using GPU acceleration
 * 
 * @param nx, ny Grid dimensions
 * @param param Simulation parameters
 * @param all_data Simulation data structures
 */
void boundary_conditions(int nx, int ny, const parameters_t param, all_data_t *all_data) {
    double* u_gpu = all_data->u->values;
    double* v_gpu = all_data->v->values;
    double* eta_gpu = all_data->eta->values;

    #pragma omp target teams distribute parallel for \
        map(tofrom: u_gpu[0:ny*(nx+1)])
    for(int j = 0; j < ny; j++) {
        u_gpu[(nx+1)*j + 0] = 0.0;     // Left boundary
        u_gpu[(nx+1)*j + (nx)] = 0.0;  // Right boundary
    }

    #pragma omp target teams distribute parallel for \
        map(tofrom: v_gpu[0:nx*(ny+1)])
    for(int i = 0; i < nx; i++) {
        v_gpu[i] = 0.0;            // Bottom boundary
        v_gpu[nx*(ny) + i] = 0.0;  // Top boundary
    }
}

/**
 * Applies source terms using GPU acceleration
 * 
 * @param n Current time step
 * @param nx, ny Grid dimensions
 * @param param Simulation parameters
 * @param all_data Simulation data structures
 */
void apply_source(int n, int nx, int ny, const parameters_t param, all_data_t *all_data) {
    double* v_gpu = all_data->v->values;
    double* eta_gpu = all_data->eta->values;
    double t = n * param.dt;
    double A = 5.0;
    double f = 1.0 / 20.0;
    
    double t_start = 5.0 / f;
    double envelope = 1.0 - exp(-(t/t_start) * (t/t_start));
    double source = A * sin(2 * M_PI * f * t) * envelope;

    switch(param.source_type) {
        case 1:
            #pragma omp target teams distribute parallel for \
                map(tofrom: v_gpu[0:nx*(ny+1)])
            for(int i = 0; i < nx; i++) {
                for(int j = 0; j < ny + 1; j++) {
                    if(j == ny) {
                        double x_pos = i * param.dx;
                        double spatial_mod = sin(2.0 * M_PI * x_pos / (nx * param.dx) * 2);
                        v_gpu[nx * j + i] = source * (1.0 + 0.3 * spatial_mod);
                    }
                }
            }
            break;

        case 2:
            #pragma omp target \
                map(tofrom: eta_gpu[0:nx*ny])
            {
                eta_gpu[nx * (ny/2) + nx/2] = source;
            }
            break;

        case 3:
            {
                const int num_sources = 3;
                int source_positions[3][2] = {
                    {nx/4, ny/4},
                    {nx/2, ny/2},
                    {3*nx/4, 3*ny/4}
                };
                double phase_shifts[3] = {0.0, 2.0*M_PI/3.0, 4.0*M_PI/3.0};

                #pragma omp target \
                    map(tofrom: eta_gpu[0:nx*ny])
                {
                    for (int s = 0; s < num_sources; s++) {
                        int i = source_positions[s][0];
                        int j = source_positions[s][1];
                        if (i >= 0 && i < nx && j >= 0 && j < ny) {
                            double phase_shifted_source = A * sin(2.0 * M_PI * f * t + phase_shifts[s]) * envelope;
                            eta_gpu[nx * j + i] = phase_shifted_source;
                        }
                    }
                }
            }
            break;

        case 4:
            {
                double speed = 0.004;
                int source_i = (int)(nx/4 + (nx/2) * sin(speed * t));
                int source_j = (int)(ny/2 + (ny/4) * cos(speed * t));

                #pragma omp target \
                    map(tofrom: eta_gpu[0:nx*ny])
                {
                    if (source_i >= 0 && source_i < nx && source_j >= 0 && source_j < ny) {
                        eta_gpu[nx * source_j + source_i] = source;
                    }
                }
            }
            break;

        default:
            printf("Warning: Unknown source type %d\n", param.source_type);
            break;
    }
}