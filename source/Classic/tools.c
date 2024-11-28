#include "shallow.h"
#include <string.h>
#include <ctype.h>

int read_parameters(struct parameters *param, const char *filename) {
    char full_path[MAX_PATH_LENGTH];
    snprintf(full_path, sizeof(full_path), "../../%s%s", INPUT_DIR, filename);

    FILE *fp = fopen(full_path, "r");
    if(!fp) {
        printf("Error: Could not open parameter file '%s'\n", full_path);
        return 1;
    }

    char line[1024];
    char *token;
    int param_count = 0;
    int ok = 1;

    while(fgets(line, sizeof(line), fp) && ok) {

        char *start = line;
        while(*start && isspace(*start)) start++;

        // ignore line if "#" symbol
        if(*start == '\0' || *start == '#') continue;


        switch(param_count) {
            case 0:
                if(sscanf(start, "%lf", &param->dx) != 1) ok = 0;
                break;
            case 1:
                if(sscanf(start, "%lf", &param->dy) != 1) ok = 0;
                break;
            case 2:
                if(sscanf(start, "%lf", &param->dt) != 1) ok = 0;
                break;
            case 3:
                if(sscanf(start, "%lf", &param->max_t) != 1) ok = 0;
                break;
            case 4:
                if(sscanf(start, "%lf", &param->g) != 1) ok = 0;
                break;
            case 5:
                if(sscanf(start, "%lf", &param->gamma) != 1) ok = 0;
                break;
            case 6:
                if(sscanf(start, "%d", &param->source_type) != 1) ok = 0;
                break;
            case 7:
                if(sscanf(start, "%d", &param->boundary_type) != 1) ok = 0;
                break;
            case 8:
                if(sscanf(start, "%d", &param->sampling_rate) != 1) ok = 0;
                break;
            case 9: {
                char temp[MAX_PATH_LENGTH];
                // Extraire le premier mot non-commenté
                token = strtok(start, " \t\n#");
                if(!token || strlen(token) >= MAX_PATH_LENGTH) {
                    ok = 0;
                    break;
                }
                strncpy(temp, token, MAX_PATH_LENGTH-1);
                temp[MAX_PATH_LENGTH-1] = '\0';
                
                if (strlen(INPUT_DIR) + strlen(temp) + 1 > MAX_PATH_LENGTH) {
                    printf("Error: Path too long for input_h_filename\n");
                    ok = 0;
                } else {
                    strcpy(param->input_h_filename, "../../"INPUT_DIR);
                    strcat(param->input_h_filename, temp);
                }
                break;
            }
            case 10: {
                token = strtok(start, " \t\n#");
                if(!token || strlen(token) >= 256) {
                    ok = 0;
                    break;
                }
                strncpy(param->output_eta_filename, token, 255);
                param->output_eta_filename[255] = '\0';
                break;
            }
            case 11: {
                token = strtok(start, " \t\n#");
                if(!token || strlen(token) >= 256) {
                    ok = 0;
                    break;
                }
                strncpy(param->output_u_filename, token, 255);
                param->output_u_filename[255] = '\0';
                break;
            }
            case 12: {
                token = strtok(start, " \t\n#");
                if(!token || strlen(token) >= 256) {
                    ok = 0;
                    break;
                }
                strncpy(param->output_v_filename, token, 255);
                param->output_v_filename[255] = '\0';
                break;
            }
        }
        param_count++;
    }

    fclose(fp);

    // Check all parameters read
    if(!ok || param_count != 13) {
        printf("Error: Could not read one or more parameters in '%s'\n", full_path);
        printf("Expected 13 parameters, got %d\n", param_count);
        return 1;
    }

    return 0;
}


void print_parameters(const struct parameters *param)
{
  printf("Parameters:\n");
  printf(" - grid spacing (dx, dy): %g m, %g m\n", param->dx, param->dy);
  printf(" - time step (dt): %g s\n", param->dt);
  printf(" - maximum time (max_t): %g s\n", param->max_t);
  printf(" - gravitational acceleration (g): %g m/s^2\n", param->g);
  printf(" - dissipation coefficient (gamma): %g 1/s\n", param->gamma);
  printf(" - source type: %d\n", param->source_type);
  printf(" - sampling rate: %d\n", param->sampling_rate);
  printf(" - input bathymetry (h) file: '%s'\n", param->input_h_filename);
  printf(" - output elevation (eta) file: '%s'\n", param->output_eta_filename);
  printf(" - output velocity (u, v) files: '%s', '%s'\n",
         param->output_u_filename, param->output_v_filename);
}

int read_data(data_t *data, const char *filename)
{
  printf("read_data function\n");
  FILE *fp = fopen(filename, "rb");
  if(!fp) {
    printf("Error: Could not open input data file '%s'\n", filename);
    return 1;
  }
  int ok = 1;
  if(ok) ok = (fread(&data->nx, sizeof(int), 1, fp) == 1);
  if(ok) ok = (fread(&data->ny, sizeof(int), 1, fp) == 1);
  if(ok) ok = (fread(&data->dx, sizeof(double), 1, fp) == 1);
  if(ok) ok = (fread(&data->dy, sizeof(double), 1, fp) == 1);
  if(ok) {
    int N = data->nx * data->ny;
    if(N <= 0) {
      printf("Error: Invalid number of data points %d\n", N);
      ok = 0;
    }
    else {
      data->values = (double*)malloc(N * sizeof(double));
      if(!data->values) {
        printf("Error: Could not allocate data (%d doubles)\n", N);
        ok = 0;
      }
      else {
        ok = (fread(data->values, sizeof(double), N, fp) == N);
      }
    }
  }
  fclose(fp);
  if(!ok) {
    printf("Error reading input data file '%s'\n", filename);
    return 1;
  }
  return 0;
}


int write_data(const data_t *data, const char *filename, int step)
{
  char out[MAX_PATH_LENGTH];
  if(step < 0)
    sprintf(out, "../../output/%s.dat", filename);
  else
    sprintf(out, "../../output/%s_%d.dat", filename, step);
  FILE *fp = fopen(out, "wb");
  if(!fp) {
    printf("Error: Could not open output data file '%s'\n", out);
    return 1;
  }
  int ok = 1;
  if(ok) ok = (fwrite(&data->nx, sizeof(int), 1, fp) == 1);
  if(ok) ok = (fwrite(&data->ny, sizeof(int), 1, fp) == 1);
  if(ok) ok = (fwrite(&data->dx, sizeof(double), 1, fp) == 1);
  if(ok) ok = (fwrite(&data->dy, sizeof(double), 1, fp) == 1);
  int N = data->nx * data->ny;
  if(ok) ok = (fwrite(data->values, sizeof(double), N, fp) == N);
  fclose(fp);
  if(!ok) {
    printf("Error writing data file '%s'\n", out);
    return 1;
  }
  return 0;
}



int write_data_vtk(const data_t *data, const char *name,
                   const char *filename, int step) {

  char out[MAX_PATH_LENGTH];
 if(step < 0)
    sprintf(out, "../../output/%s.vti", filename);
  else
    sprintf(out, "../../output/%s_%d.vti", filename, step);

  FILE *fp = fopen(out, "wb");
  if(!fp) {
    printf("Error: Could not open output VTK file '%s'\n", out);
    return 1;
  }

  uint64_t num_points = data->nx * data->ny;
  uint64_t num_bytes = num_points * sizeof(double);

  fprintf(fp, "<?xml version=\"1.0\"?>\n");
  fprintf(fp, "<VTKFile type=\"ImageData\" version=\"1.0\" "
          "byte_order=\"LittleEndian\" header_type=\"UInt64\">\n");
  fprintf(fp, "  <ImageData WholeExtent=\"0 %d 0 %d 0 0\" "
          "Spacing=\"%lf %lf 0.0\">\n",
          data->nx - 1, data->ny - 1, data->dx, data->dy);
  fprintf(fp, "    <Piece Extent=\"0 %d 0 %d 0 0\">\n",
          data->nx - 1, data->ny - 1);

  fprintf(fp, "      <PointData Scalars=\"scalar_data\">\n");
  fprintf(fp, "        <DataArray type=\"Float64\" Name=\"%s\" "
          "format=\"appended\" offset=\"0\">\n", name);
  fprintf(fp, "        </DataArray>\n");
  fprintf(fp, "      </PointData>\n");

  fprintf(fp, "    </Piece>\n");
  fprintf(fp, "  </ImageData>\n");

  fprintf(fp, "  <AppendedData encoding=\"raw\">\n_");

  fwrite(&num_bytes, sizeof(uint64_t), 1, fp);
  fwrite(data->values, sizeof(double), num_points, fp);

  fprintf(fp, "  </AppendedData>\n");
  fprintf(fp, "</VTKFile>\n");

  fclose(fp);
  return 0;
}

int write_manifest_vtk(const char *filename, double dt, int nt,
                       int sampling_rate)
{
  char out[MAX_PATH_LENGTH];
  sprintf(out, "../../output/%s.pvd", filename);

   FILE *fp = fopen(out, "wb");
  if(!fp) {
    printf("Error: Could not open output VTK manifest file '%s'\n", out);
    return 1;
  }

  fprintf(fp, "<VTKFile type=\"Collection\" version=\"0.1\" "
          "byte_order=\"LittleEndian\">\n");
  fprintf(fp, "  <Collection>\n");
  for(int n = 0; n < nt; n++) {
    if(sampling_rate && !(n % sampling_rate)) {
      double t = n * dt;
      fprintf(fp, "    <DataSet timestep=\"%g\" file='%s_%d.vti'/>\n", t,
              filename, n);
    }
  }
  fprintf(fp, "  </Collection>\n");
  fprintf(fp, "</VTKFile>\n");
  fclose(fp);
  return 0;
}

int init_data(data_t *data, int nx, int ny, double dx, double dy,
              double val)
{
  data->nx = nx;
  data->ny = ny;
  data->dx = dx;
  data->dy = dy;
  data->values = (double*)malloc(nx * ny * sizeof(double));
  if(!data->values){
    printf("Error: Could not allocate data\n");
    return 1;
  }
  for(int i = 0; i < nx * ny; i++) data->values[i] = val;
  return 0;
}

void free_data(data_t *data)
{
  free(data->values);
}