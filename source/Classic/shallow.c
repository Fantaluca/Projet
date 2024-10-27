#include "shallow.h"


double interpolate_data(const data_t *data, 
                        double x, 
                        double y){

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


double update_eta(int nx, 
                  int ny, 
                  const struct parameters param, 
                  data_t *u, 
                  data_t *v, 
                  data_t *eta, 
                  data_t *h_interp){

  for(int i = 0; i < nx; i++) {
    for(int j = 0; j < ny ; j++) {
      // TODO: this does not evaluate h at the correct locations
      double h_ij = GET(h_interp, i, j);
      double c1 = param.dt * h_ij;
      double eta_ij = GET(eta, i, j)
        - c1 / param.dx * (GET(u, i + 1, j) - GET(u, i, j))
        - c1 / param.dy * (GET(v, i, j + 1) - GET(v, i, j));
      SET(eta, i, j, eta_ij);
    }
  }
}

double update_velocities(int nx, 
                         int ny, 
                         const struct parameters param, 
                         data_t *u, 
                         data_t *v, 
                         data_t *eta){

  for(int i = 0; i < nx; i++){
    for(int j = 0; j < ny; j++){

      double c1 = param.dt * param.g;
      double c2 = param.dt * param.gamma;
      double eta_ij = GET(eta, i, j);
      double eta_imj = GET(eta, (i == 0) ? 0 : i - 1, j);
      double eta_ijm = GET(eta, i, (j == 0) ? 0 : j - 1);
      double u_ij = (1. - c2) * GET(u, i, j)
        - c1 / param.dx * (eta_ij - eta_imj);
      double v_ij = (1. - c2) * GET(v, i, j)
        - c1 / param.dy * (eta_ij - eta_ijm);
      SET(u, i, j, u_ij);
      SET(v, i, j, v_ij);
    }
  }
}

void boundary_condition(int n,
                        int nx, 
                        int ny, 
                        const struct parameters param, 
                        data_t *u, 
                        data_t *v, 
                        data_t *eta,
                        const data_t *h_interp)
{
    double t = n * param.dt;
    if(param.source_type == 1){
      // sinusoidal velocity on top boundary
      double A = 5;
      double f = 1. / 20.;
      for(int i = 0; i < nx; i++) {
        for(int j = 0; j < ny; j++){

          SET(u, 0, j, 0.);
          SET(u, nx, j, 0.);
          SET(v, i, 0, 0.);
          SET(v, i, ny, A * sin(2 * M_PI * f * t));
        }
      }
    }
    else if(param.source_type == 2){
        // sinusoidal elevation in the middle of the domain
        double A = 5;
        double f = 1. / 20.;
        SET(eta, nx / 2, ny / 2, A * sin(2 * M_PI * f * t));

        // Apply boundary condition on (eta, u, v) for each corners of the domain
        for(int i = 0; i < nx; i++) {
            double h_bottom = GET(h_interp, i, 0);
            double h_top = GET(h_interp, i, ny-1);
            double c_bottom = sqrt(param.g * h_bottom);
            double c_top = sqrt(param.g * h_top);

            // Bottom corner
            SET(eta, i, 0, GET(eta, i, 1) - (c_bottom * param.dt / param.dy) * (GET(eta, i, 1) - GET(eta, i, 0)));
            SET(u, i, 0, GET(u, i, 1) - (c_bottom * param.dt / param.dy) * (GET(u, i, 1) - GET(u, i, 0)));
            SET(v, i, 0, GET(v, i, 1) - (c_bottom * param.dt / param.dy) * (GET(v, i, 1) - GET(v, i, 0)));

            // Top corner
            SET(eta, i, ny-1, GET(eta, i, ny-2) - (c_top * param.dt / param.dy) * (GET(eta, i, ny-1) - GET(eta, i, ny-2)));
            SET(u, i, ny-1, GET(u, i, ny-2) - (c_top * param.dt / param.dy) * (GET(u, i, ny-1) - GET(u, i, ny-2)));
            SET(v, i, ny-1, GET(v, i, ny-2) - (c_top * param.dt / param.dy) * (GET(v, i, ny-1) - GET(v, i, ny-2)));
        }

        for(int j = 0; j < ny; j++) {
            double h_left = GET(h_interp, 0, j);
            double h_right = GET(h_interp, nx-1, j);
            double c_left = sqrt(param.g * h_left);
            double c_right = sqrt(param.g * h_right);

            // Left corner
            SET(eta, 0, j, GET(eta, 1, j) - (c_left * param.dt / param.dx) * (GET(eta, 1, j) - GET(eta, 0, j)));
            SET(u, 0, j, GET(u, 1, j) - (c_left * param.dt / param.dx) * (GET(u, 1, j) - GET(u, 0, j)));
            SET(v, 0, j, GET(v, 1, j) - (c_left * param.dt / param.dx) * (GET(v, 1, j) - GET(v, 0, j)));

            // Right corner
            SET(eta, nx-1, j, GET(eta, nx-2, j) - (c_right * param.dt / param.dx) * (GET(eta, nx-1, j) - GET(eta, nx-2, j)));
            SET(u, nx-1, j, GET(u, nx-2, j) - (c_right * param.dt / param.dx) * (GET(u, nx-1, j) - GET(u, nx-2, j)));
            SET(v, nx-1, j, GET(v, nx-2, j) - (c_right * param.dt / param.dx) * (GET(v, nx-1, j) - GET(v, nx-2, j)));
        }
    }
    else{
        // TODO: add other sources
        printf("Error: Unknown source type %d\n", param.source_type);
        exit(0);
    }
}

void interp_bathy(int nx,
                  int ny, 
                  const struct parameters param,
                  data_t *h_interp, 
                  data_t *h){

  for(int i = 0; i < nx; i++){
    for(int j = 0; j < ny; j++){

      double x = i * param.dx;
      double y = j * param.dy;
      double val = interpolate_data(h, x, y);
      SET(h_interp, i, j, val);
    }
  }
}


int main(int argc, char **argv){

  if(argc != 2){
    printf("Usage: %s parameter_file\n", argv[0]);
    return 1;
  }

  // Initialize paramters and h
  struct parameters param;
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


  double start = GET_TIME(); // time init

  // Loop over timestep
  for(int n = 0; n < nt; n++) {

    if(n && (n % (nt / 10)) == 0) {
      double time_sofar = GET_TIME() - start;
      double eta = (nt - n) * time_sofar / n;
      printf("Computing step %d/%d (ETA: %g seconds)     \r", n, nt, eta);
      fflush(stdout);
    }

    // output solution
    if(param.sampling_rate && !(n % param.sampling_rate)){
      write_data_vtk(&eta, "water elevation", param.output_eta_filename, n);
      //write_data_vtk(&u, "x velocity", param.output_u_filename, n);
      //write_data_vtk(&v, "y velocity", param.output_v_filename, n);
    }

    // impose boundary conditions
    boundary_condition(n, nx, ny, param, &u, &v, &eta, &h_interp);

    // Update variables
    update_eta(nx, ny, param, &u, &v, &eta, &h_interp);
    update_velocities(nx, ny, param, &u, &v, &eta);


    
  }

  write_manifest_vtk(param.output_eta_filename, param.dt, nt,
                     param.sampling_rate);
  //write_manifest_vtk(param.output_u_filename, param.dt, nt,
  //                   param.sampling_rate);
  //write_manifest_vtk(param.output_v_filename, param.dt, nt,
  //                   param.sampling_rate);

  double time = GET_TIME() - start;
  printf("\nDone: %g seconds (%g MUpdates/s)\n", time,
         1e-6 * (double)eta.nx * (double)eta.ny * (double)nt / time);

  free_data(&h_interp);
  free_data(&eta);
  free_data(&u);
  free_data(&v);

  return 0;
}
