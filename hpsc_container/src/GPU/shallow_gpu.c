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

    #pragma omp target teams distribute parallel for collapse(2) \
        map(tofrom: *all_data->eta) \
        map(to: *all_data->h_interp, *all_data->u, *all_data->v)
    for(int i = 0; i < nx; i++) {
        for(int j = 0; j < ny; j++) {
            double h_ij = GET(all_data->h_interp, i, j);
            double c1 = param.dt * h_ij;
            double eta_ij = GET(all_data->eta, i, j)
                - c1 / param.dx * (GET(all_data->u, i + 1, j) - GET(all_data->u, i, j))
                - c1 / param.dy * (GET(all_data->v, i, j + 1) - GET(all_data->v, i, j));
            SET(all_data->eta, i, j, eta_ij);
        }
    }
}

void update_velocities(int nx, int ny, const parameters_t param, all_data_t *all_data) {
    #pragma omp target teams distribute parallel for collapse(2) \
        map(tofrom: *all_data->u, *all_data->v) \
        map(to: *all_data->eta)
    for(int i = 0; i < nx; i++) {
        for(int j = 0; j < ny; j++) {
            double c1 = param.dt * param.g;
            double c2 = param.dt * param.gamma;
            double eta_ij = GET(all_data->eta, i, j);
            double eta_imj = GET(all_data->eta, (i == 0) ? 0 : i - 1, j);
            double eta_ijm = GET(all_data->eta, i, (j == 0) ? 0 : j - 1);
            
            // Récupération des anciennes vitesses
            double u_old = GET(all_data->u, i, j);
            double v_old = GET(all_data->v, i, j);
            
            // Calcul des nouvelles vitesses avec Coriolis
            double u_ij = (1. - c2) * u_old 
                - c1 / param.dx * (eta_ij - eta_imj)
                + param.dt * param.f * v_old;  // Terme de Coriolis
                
            double v_ij = (1. - c2) * v_old
                - c1 / param.dy * (eta_ij - eta_ijm)
                - param.dt * param.f * u_old;  // Terme de Coriolis
            
            SET(all_data->u, i, j, u_ij);
            SET(all_data->v, i, j, v_ij);
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
