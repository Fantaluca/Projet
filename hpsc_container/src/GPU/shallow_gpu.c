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
            
            // Équivalent à GET(&eta, i, j) et GET(&eta, i-1, j)
            double eta_ij, eta_im1j;
            
            if (i < nx) {
                eta_ij = eta_gpu[nx * j + i];
            } else {
                eta_ij = eta_gpu[nx * j + (nx-1)];  // dernier point valide
            }
            
            if (i > 0) {
                eta_im1j = eta_gpu[nx * j + (i-1)];
            } else {
                eta_im1j = eta_gpu[nx * j + 0];     // premier point valide
            }
            
            // Équivalent à GET(&u, i, j)
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
            
            // Équivalent à GET(&eta, i, j) et GET(&eta, i, j-1)
            double eta_ij, eta_ijm1;
            
            if (j < ny) {
                eta_ij = eta_gpu[nx * j + i];
            } else {
                eta_ij = eta_gpu[nx * (ny-1) + i];  // dernier point valide
            }
            
            if (j > 0) {
                eta_ijm1 = eta_gpu[nx * (j-1) + i];
            } else {
                eta_ijm1 = eta_gpu[nx * 0 + i];     // premier point valide
            }
            
            // Équivalent à GET(&v, i, j)
            double v_old = v_gpu[nx * j + i];
            
            double new_v = (1. - c2) * v_old
                - c1 / param.dy * (eta_ij - eta_ijm1);
            
            v_gpu[nx * j + i] = new_v;
        }
    }
}


void boundary_conditions(int nx, int ny, const parameters_t param, all_data_t *all_data) {
    double* u_gpu = all_data->u->values;
    double* v_gpu = all_data->v->values;
    double* eta_gpu = all_data->eta->values;

    #pragma omp target teams distribute parallel for \
        map(tofrom: u_gpu[0:ny*(nx+1)])
    for(int j = 0; j < ny; j++) {
        u_gpu[(nx+1)*j + 0] = 0.0;     // Bord gauche
        u_gpu[(nx+1)*j + (nx)] = 0.0;    // Bord droit
    }

    #pragma omp target teams distribute parallel for \
        map(tofrom: v_gpu[0:nx*(ny+1)])
    for(int i = 0; i < nx; i++) {
        v_gpu[i] = 0.0;            // Bord bas
        v_gpu[nx*(ny) + i] = 0.0;  // Bord haut
    }
}


// Fonction pour appliquer la source
void apply_source(int n, int nx, int ny, const parameters_t param, all_data_t *all_data) {

    double* v_gpu = all_data->v->values;
    double* eta_gpu = all_data->eta->values;
    double t = n * param.dt;
    double A = 5.0;
    double f = 1.0 / 20.0;
    
    if(param.source_type == 1) {
        #pragma omp target teams distribute parallel for collapse(2) \
            map(tofrom: v_gpu[0:nx*(ny+1)])
        for(int i = 0; i < nx; i++) {
            for(int j = 0; j < ny + 1; j++) {
                if(j == ny) {
                    v_gpu[nx * j + i] = A * sin(2 * M_PI * f * t);
                }
            }
        }
    }
    else if(param.source_type == 2) {
        #pragma omp target \
            map(tofrom: eta_gpu[0:nx*ny])
        {
            eta_gpu[nx * (ny/2) + nx/2] = A * sin(2 * M_PI * f * t);
        }
    }
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
