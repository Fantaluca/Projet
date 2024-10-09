#include "tools.c"


double interpolate_data(const struct data *data, 
                        double x, 
                        double y)
{
  // TODO: this returns the nearest neighbor, should implement actual
  // interpolation instead
  int i = (int)(x / data->dx);
  int j = (int)(y / data->dy);
  if(i < 0) i = 0;
  else if(i > data->nx - 1) i = data->nx - 1;
  if(j < 0) j = 0;
  else if(j > data->ny - 1) j = data->ny - 1;

  int a = 1;

  double val = GET(data, i, j);
  return val;
}


double update_eta(int nx, 
                  int ny, 
                  const struct parameters param, 
                  struct data *u, 
                  struct data *v, 
                  struct data *eta, 
                  struct data *h_interp)
{
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
                         struct data *u, 
                         struct data *v, 
                         struct data *eta)
{
  for(int i = 0; i < nx; i++) {
    for(int j = 0; j < ny; j++) {
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


int main(int argc, char **argv)
{
  if(argc != 2) {
    printf("Usage: %s parameter_file\n", argv[0]);
    return 1;
  }

  struct parameters param;
  if(read_parameters(&param, argv[1])) return 1;
  print_parameters(&param);

  struct data h;
  if(read_data(&h, param.input_h_filename)) return 1;

  // infer size of domain from input bathymetric data
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

  struct data eta, u, v;
  init_data(&eta, nx, ny, param.dx, param.dy, 0.);
  init_data(&u, nx + 1, ny, param.dx, param.dy, 0.);
  init_data(&v, nx, ny + 1, param.dx, param.dy, 0.);

  // interpolate bathymetry
  struct data h_interp;
  init_data(&h_interp, nx, ny, param.dx, param.dy, 0.);
  for(int j = 0; j < ny; j++) {
    for(int i = 0; i < nx; i++) {
      double x = i * param.dx;
      double y = j * param.dy;
      double val = interpolate_data(&h, x, y);
      SET(&h_interp, i, j, val);
    }
  }

  double start = GET_TIME();

  for(int n = 0; n < nt; n++) {

    if(n && (n % (nt / 10)) == 0) {
      double time_sofar = GET_TIME() - start;
      double eta = (nt - n) * time_sofar / n;
      printf("Computing step %d/%d (ETA: %g seconds)     \r", n, nt, eta);
      fflush(stdout);
    }

    // output solution
    if(param.sampling_rate && !(n % param.sampling_rate)) {
      write_data_vtk(&eta, "water elevation", param.output_eta_filename, n);
      //write_data_vtk(&u, "x velocity", param.output_u_filename, n);
      //write_data_vtk(&v, "y velocity", param.output_v_filename, n);
    }

    // impose boundary conditions
    double t = n * param.dt;
    if(param.source_type == 1) {
      // sinusoidal velocity on top boundary
      double A = 5;
      double f = 1. / 20.;
      for(int i = 0; i < nx; i++) {
        for(int j = 0; j < ny; j++) {
          SET(&u, 0, j, 0.);
          SET(&u, nx, j, 0.);
          SET(&v, i, 0, 0.);
          SET(&v, i, ny, A * sin(2 * M_PI * f * t));
        }
      }
    }
    else if(param.source_type == 2) {
      // sinusoidal elevation in the middle of the domain
      double A = 5;
      double f = 1. / 20.;
      SET(&eta, nx / 2, ny / 2, A * sin(2 * M_PI * f * t));
    }
    else {
      // TODO: add other sources
      printf("Error: Unknown source type %d\n", param.source_type);
      exit(0);
    }

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
