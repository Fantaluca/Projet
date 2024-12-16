#include "shallow_gpu.h"

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

    // Weighted coef
    double wx = (x2 - x) / (x2 - x1);
    double wy = (y2 - y) / (y2 - y1);

    double val = wx * wy * Q11 +
                 (1 - wx) * wy * Q21 +
                 wx * (1 - wy) * Q12 +
                 (1 - wx) * (1 - wy) * Q22;

    return val;
}

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
            
            // u est sur une grille décalée (nx+1) x ny
            double u_i = u_gpu[(nx+1) * j + i];
            double u_ip1 = u_gpu[(nx+1) * j + (i+1)];
            
            // v est sur une grille décalée nx x (ny+1)
            double v_j = v_gpu[nx * j + i];
            double v_jp1 = v_gpu[nx * (j+1) + i];
            
            eta_gpu[nx * j + i] = eta_gpu[nx * j + i] 
                - c1 * ((u_ip1 - u_i) / param.dx 
                     + (v_jp1 - v_j) / param.dy);
        }
    }
}

void update_velocities(int nx, int ny, const parameters_t param, all_data_t *all_data) {
    double* u_gpu = all_data->u->values;
    double* v_gpu = all_data->v->values;
    double* eta_gpu = all_data->eta->values;

    // Pour u: grille (nx+1) x ny
    #pragma omp target teams distribute parallel for collapse(2) \
        map(tofrom: u_gpu[0:(nx+1)*ny]) \
        map(to: eta_gpu[0:nx*ny])
    for(int i = 0; i < nx + 1; i++) {
        for(int j = 0; j < ny; j++) {
            double c1 = param.dt * param.g;
            double c2 = param.dt * param.gamma;
            
            // Gestion des conditions aux limites pour eta
            double eta_ij = (i < nx) ? eta_gpu[nx * j + i] 
                                   : eta_gpu[nx * j + (nx-1)];
            double eta_im1j = (i > 0) ? eta_gpu[nx * j + (i-1)]
                                     : eta_gpu[nx * j + 0];
            
            double u_old = u_gpu[(nx+1) * j + i];
            double new_u = (1. - c2) * u_old 
                - c1 / param.dx * (eta_ij - eta_im1j);
            
            u_gpu[(nx+1) * j + i] = new_u;
        }
    }

    // Pour v: grille nx x (ny+1)
    #pragma omp target teams distribute parallel for collapse(2) \
        map(tofrom: v_gpu[0:nx*(ny+1)]) \
        map(to: eta_gpu[0:nx*ny])
    for(int i = 0; i < nx; i++) {
        for(int j = 0; j < ny + 1; j++) {
            double c1 = param.dt * param.g;
            double c2 = param.dt * param.gamma;
            
            // Gestion des conditions aux limites pour eta
            double eta_ij = (j < ny) ? eta_gpu[nx * j + i]
                                   : eta_gpu[nx * (ny-1) + i];
            double eta_ijm1 = (j > 0) ? eta_gpu[nx * (j-1) + i]
                                     : eta_gpu[nx * 0 + i];
            
            double v_old = v_gpu[nx * j + i];
            double new_v = (1. - c2) * v_old
                - c1 / param.dy * (eta_ij - eta_ijm1);
            
            v_gpu[nx * j + i] = new_v;
        }
    }
}


void boundary_condition(int nx, int ny, const parameters_t param, all_data_t *all_data) {
    #pragma omp target teams distribute parallel for \
        map(tofrom: *all_data->eta, *all_data->u, *all_data->v) \
        map(to: *all_data->h_interp)
    for(int i = 0; i < nx; i++) {
        double h_bottom = GET(all_data->h_interp, i, 0);
        double h_top = GET(all_data->h_interp, i, ny-1);
        double c_bottom = sqrt(param.g * h_bottom);
        double c_top = sqrt(param.g * h_top);
        
        if(i > 0 && i < nx-1) {
            if(param.boundary_type == 0) {
                // Conditions réflexives
                SET(all_data->eta, i, 0, GET(all_data->eta, i, 1));
                SET(all_data->eta, i, ny-1, GET(all_data->eta, i, ny-2));
                SET(all_data->u, i, 0, GET(all_data->u, i, 1));
                SET(all_data->u, i, ny-1, GET(all_data->u, i, ny-2));
                SET(all_data->v, i, 0, -GET(all_data->v, i, 1));
                SET(all_data->v, i, ny-1, -GET(all_data->v, i, ny-2));
            } else {
                // Conditions transparentes
                SET(all_data->eta, i, 0, GET(all_data->eta, i, 1) - 
                    (c_bottom * param.dt / param.dy) * 
                    (GET(all_data->eta, i, 1) - GET(all_data->eta, i, 0)));
                SET(all_data->u, i, 0, GET(all_data->u, i, 1) - 
                    (c_bottom * param.dt / param.dy) * 
                    (GET(all_data->u, i, 1) - GET(all_data->u, i, 0)));
                SET(all_data->v, i, 0, GET(all_data->v, i, 1) - 
                    (c_bottom * param.dt / param.dy) * 
                    (GET(all_data->v, i, 1) - GET(all_data->v, i, 0)));

                SET(all_data->eta, i, ny-1, GET(all_data->eta, i, ny-2) - 
                    (c_top * param.dt / param.dy) * 
                    (GET(all_data->eta, i, ny-1) - GET(all_data->eta, i, ny-2)));
                SET(all_data->u, i, ny-1, GET(all_data->u, i, ny-2) - 
                    (c_top * param.dt / param.dy) * 
                    (GET(all_data->u, i, ny-1) - GET(all_data->u, i, ny-2)));
                SET(all_data->v, i, ny-1, GET(all_data->v, i, ny-2) - 
                    (c_top * param.dt / param.dy) * 
                    (GET(all_data->v, i, ny-1) - GET(all_data->v, i, ny-2)));
            }
        }
    }

    #pragma omp target teams distribute parallel for \
        map(tofrom: *all_data->eta, *all_data->u, *all_data->v) \
        map(to: *all_data->h_interp)
    for(int j = 0; j < ny; j++) {
        double h_left = GET(all_data->h_interp, 0, j);
        double h_right = GET(all_data->h_interp, nx-1, j);
        double c_left = sqrt(param.g * h_left);
        double c_right = sqrt(param.g * h_right);
        
        if(j > 0 && j < ny-1) {
            if(param.boundary_type == 0) {
                // Conditions réflexives
                SET(all_data->eta, 0, j, GET(all_data->eta, 1, j));
                SET(all_data->eta, nx-1, j, GET(all_data->eta, nx-2, j));
                SET(all_data->u, 0, j, -GET(all_data->u, 1, j));
                SET(all_data->u, nx-1, j, -GET(all_data->u, nx-2, j));
                SET(all_data->v, 0, j, GET(all_data->v, 1, j));
                SET(all_data->v, nx-1, j, GET(all_data->v, nx-2, j));
            } else {
                // Conditions transparentes
                SET(all_data->eta, 0, j, GET(all_data->eta, 1, j) - 
                    (c_left * param.dt / param.dx) * 
                    (GET(all_data->eta, 1, j) - GET(all_data->eta, 0, j)));
                SET(all_data->u, 0, j, GET(all_data->u, 1, j) - 
                    (c_left * param.dt / param.dx) * 
                    (GET(all_data->u, 1, j) - GET(all_data->u, 0, j)));
                SET(all_data->v, 0, j, GET(all_data->v, 1, j) - 
                    (c_left * param.dt / param.dx) * 
                    (GET(all_data->v, 1, j) - GET(all_data->v, 0, j)));

                SET(all_data->eta, nx-1, j, GET(all_data->eta, nx-2, j) - 
                    (c_right * param.dt / param.dx) * 
                    (GET(all_data->eta, nx-1, j) - GET(all_data->eta, nx-2, j)));
                SET(all_data->u, nx-1, j, GET(all_data->u, nx-2, j) - 
                    (c_right * param.dt / param.dx) * 
                    (GET(all_data->u, nx-1, j) - GET(all_data->u, nx-2, j)));
                SET(all_data->v, nx-1, j, GET(all_data->v, nx-2, j) - 
                    (c_right * param.dt / param.dx) * 
                    (GET(all_data->v, nx-1, j) - GET(all_data->v, nx-2, j)));
            }
        }
    }
}

// Fonction pour appliquer la source
void apply_source(int n, int nx, int ny, const parameters_t param, all_data_t *all_data) {
    double t = n * param.dt;
    double A = 5.0;
    double f = 1.0 / 20.0;
    
    if(param.source_type == 1) {
        // Source sinusoïdale sur le bord supérieur
        #pragma omp target teams distribute parallel for collapse(2) \
            map(tofrom: *all_data->v)
        for(int i = 0; i < nx; i++) {
            for(int j = 0; j < ny + 1; j++) {
                if(j == ny) {
                    SET(all_data->v, i, j, A * sin(2 * M_PI * f * t));
                }
            }
        }
    }
    else if(param.source_type == 2) {
        // Source ponctuelle au centre
        #pragma omp target teams distribute parallel for collapse(2) \
            map(tofrom: *all_data->eta)
        for(int i = 0; i < nx; i++) {
            for(int j = 0; j < ny; j++) {
                if(i == nx/2 && j == ny/2) {
                    SET(all_data->eta, i, j, A * sin(2 * M_PI * f * t));
                }
            }
        }
    }
}

// Dans interp_bathy, ajout du mapping correct
void interp_bathy(int nx, int ny, const parameters_t param, all_data_t *all_data) {
    #pragma omp target teams distribute parallel for collapse(2) \
        map(to: *all_data->h) \
        map(tofrom: *all_data->h_interp)
    for(int i = 0; i < nx; i++) {
        for(int j = 0; j < ny; j++) {
            double x = i * param.dx;
            double y = j * param.dy;
            double val = interpolate_data(all_data->h, x, y);
            SET(all_data->h_interp, i, j, val);
        }
    }
}
