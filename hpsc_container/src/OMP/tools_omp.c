/*===========================================================
 * SHALLOW WATER EQUATIONS SOLVER - SERIAL VERSION
 * I/O and Initialization Functions
 * Contains file handling, setup and cleanup routines
 ===========================================================*/

#include "shallow_omp.h"
#include <ctype.h>

/*===========================================================
 * FILE I/O FUNCTIONS 
 ===========================================================*/

/**
 * Reads simulation parameters from input file
 * 
 * @param param Pointer to parameters structure to fill
 * @param filename Name of input parameter file
 * @return 0 on success, 1 on failure
 */
int read_parameters(parameters_t *param, const char *filename) {
    char full_path[MAX_PATH_LENGTH];
    snprintf(full_path, sizeof(full_path), "%s%s", INPUT_DIR, filename);

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

        if(*start == '\0' || *start == '#') continue;

        switch(param_count) {
            case 0: if(sscanf(start, "%lf", &param->dx) != 1) ok = 0; break;
            case 1: if(sscanf(start, "%lf", &param->dy) != 1) ok = 0; break;
            case 2: if(sscanf(start, "%lf", &param->dt) != 1) ok = 0; break;
            case 3: if(sscanf(start, "%lf", &param->max_t) != 1) ok = 0; break;
            case 4: if(sscanf(start, "%lf", &param->g) != 1) ok = 0; break;
            case 5: if(sscanf(start, "%lf", &param->gamma) != 1) ok = 0; break;
            case 6: if(sscanf(start, "%d", &param->source_type) != 1) ok = 0; break;
            case 7: if(sscanf(start, "%d", &param->sampling_rate) != 1) ok = 0; break;
            case 8: { // Input bathymetry filename
                token = strtok(start, " \t\n#");
                if(!token || strlen(token) >= MAX_PATH_LENGTH) {
                    ok = 0;
                    break;
                }
                if (strlen(INPUT_DIR) + strlen(token) + 1 > MAX_PATH_LENGTH) {
                    printf("Error: Path too long for input_h_filename\n");
                    ok = 0;
                } else {
                    strcpy(param->input_h_filename, INPUT_DIR);
                    strcat(param->input_h_filename, token);
                }
                break;
            }
            case 9: case 10: case 11: { // Output filenames for eta, u, v
                token = strtok(start, " \t\n#");
                if(!token || strlen(token) >= 256) {
                    ok = 0;
                    break;
                }
                char *dest = (param_count == 9) ? param->output_eta_filename :
                            (param_count == 10) ? param->output_u_filename :
                                                param->output_v_filename;
                strncpy(dest, token, 255);
                dest[255] = '\0';
                break;
            }
        }
        param_count++;
    }

    fclose(fp);

    if(!ok || param_count != 12) {
        printf("Error: Could not read parameters in '%s'. Expected 12, got %d\n", 
               full_path, param_count);
        return 1;
    }

    return 0;
}

/**
 * Reads simulation data from binary file
 * 
 * @param data Data structure to fill
 * @param filename Input file path
 * @return 0 on success, 1 on failure
 */
int read_data(data_t *data, const char *filename) {
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
        } else {
            data->values = (double*)malloc(N * sizeof(double));
            if(!data->values) {
                printf("Error: Could not allocate data (%d doubles)\n", N);
                ok = 0;
            } else {
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

/*===========================================================
 * OUTPUT FUNCTIONS
 ===========================================================*/

/**
 * Writes simulation data to binary file
 * 
 * @param data Data structure to write
 * @param filename Output file base name
 * @param step Time step number (-1 for single output)
 * @return 0 on success, 1 on failure
 */
int write_data(const data_t *data, const char *filename, int step) {
    char out[MAX_PATH_LENGTH];
    sprintf(out, "output/%s%s.dat", filename, (step < 0) ? "" : "_");
    if(step >= 0) sprintf(out + strlen(out), "%d", step);

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

/**
 * Writes data to VTK image format file
 * 
 * @param data Data structure to write
 * @param name Field name for VTK file
 * @param filename Output base filename
 * @param step Time step number (-1 for single output)
 * @return 0 on success, 1 on failure
 */
int write_data_vtk(const data_t *data, const char *name,
                   const char *filename, int step) {
    char out[MAX_PATH_LENGTH];
    sprintf(out, "../../output/%s%s.vti", filename,
            (step < 0) ? "" : "_");
    if(step >= 0) sprintf(out + strlen(out), "%d", step);

    FILE *fp = fopen(out, "wb");
    if(!fp) {
        printf("Error: Could not open output VTK file '%s'\n", out);
        return 1;
    }

    uint64_t num_points = data->nx * data->ny;
    uint64_t num_bytes = num_points * sizeof(double);

    // Write VTK XML header and structure
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

    // Write binary data
    fwrite(&num_bytes, sizeof(uint64_t), 1, fp);
    fwrite(data->values, sizeof(double), num_points, fp);

    fprintf(fp, "  </AppendedData>\n");
    fprintf(fp, "</VTKFile>\n");

    fclose(fp);
    return 0;
}

/**
 * Creates VTK manifest file for time series visualization
 * 
 * @param filename Base filename for VTK files
 * @param dt Time step size
 * @param nt Number of time steps
 * @param sampling_rate Output frequency
 * @return 0 on success, 1 on failure
 */
int write_manifest_vtk(const char *filename, double dt, int nt,
                       int sampling_rate) {
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
            fprintf(fp, "    <DataSet timestep=\"%g\" file='%s_%d.vti'/>\n", 
                    t, filename, n);
        }
    }
    
    fprintf(fp, "  </Collection>\n");
    fprintf(fp, "</VTKFile>\n");
    fclose(fp);
    return 0;
}

/*===========================================================
 * INITIALIZATION AND CLEANUP FUNCTIONS
 ===========================================================*/

/**
 * Initializes data structure with given dimensions and value
 * 
 * @param data Data structure to initialize
 * @param nx, ny Grid dimensions
 * @param dx, dy Grid spacing
 * @param val Initial value for all points
 * @return 0 on success, 1 on failure
 */
int init_data(data_t *data, int nx, int ny, double dx, double dy, double val) {
    data->nx = nx;
    data->ny = ny;
    data->dx = dx;
    data->dy = dy;
    
    data->values = (double*)malloc(nx * ny * sizeof(double));
    if(!data->values) {
        printf("Error: Could not allocate data\n");
        return 1;
    }
    
    for(int i = 0; i < nx * ny; i++) data->values[i] = val;
    return 0;
}

/**
 * Initializes all simulation data structures
 * 
 * @param param Simulation parameters
 * @return Initialized all_data_t structure or NULL on failure
 */
all_data_t* init_all_data(const parameters_t *param) {
    all_data_t* all_data = malloc(sizeof(all_data_t));
    if (all_data == NULL) {
        fprintf(stderr, "Error: Failed to allocate all_data\n");
        return NULL;
    }

    // Allocate and read bathymetry data
    all_data->h = malloc(sizeof(data_t));
    if (all_data->h == NULL) {
        fprintf(stderr, "Error: Failed to allocate h structure\n");
        free_all_data(all_data);
        return NULL;
    }

    if (read_data(all_data->h, param->input_h_filename)) {
        fprintf(stderr, "Error: Failed to read bathymetry data\n");
        free_all_data(all_data);
        return NULL;
    }

    // Calculate domain dimensions
    double hx = all_data->h->nx * all_data->h->dx;
    double hy = all_data->h->ny * all_data->h->dy;
    int nx = floor(hx / param->dx);
    int ny = floor(hy / param->dy);

    // Initialize all fields
    all_data->eta = malloc(sizeof(data_t));
    all_data->u = malloc(sizeof(data_t));
    all_data->v = malloc(sizeof(data_t));
    all_data->h_interp = malloc(sizeof(data_t));

    init_data(all_data->eta, nx, ny, param->dx, param->dy, 0.);
    init_data(all_data->u, nx + 1, ny, param->dx, param->dy, 0.);
    init_data(all_data->v, nx, ny + 1, param->dx, param->dy, 0.);
    init_data(all_data->h_interp, nx, ny, param->dx, param->dy, 0.);

    return all_data;
}

/**
 * Prints current simulation parameters
 * 
 * @param param Pointer to parameters structure
 */
void print_parameters(const parameters_t *param) {
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

/**
 * Displays simulation progress information
 * 
 * @param current_step Current simulation step
 * @param total_steps Total number of steps
 * @param start_time Simulation start time
 */
void print_progress(int current_step, int total_steps, double start_time) {
    if (current_step > 0 && (current_step % (total_steps / 10)) == 0) {
        double time_elapsed = GET_TIME() - start_time;
        double estimated_remaining = (total_steps - current_step) * 
                                   time_elapsed / current_step;
        
        printf("Computing step %d/%d (ETA: %.2f seconds)     \r", 
               current_step, total_steps, estimated_remaining);
        fflush(stdout);
    }
}

/**
 * Frees memory for single data structure
 * 
 * @param data Data structure to free
 */
void free_data(data_t *data) {
    free(data->values);
}

/**
 * Frees all allocated simulation data
 * 
 * @param all_data Main data structure to free
 */
void free_all_data(all_data_t* all_data) {
    free_data(all_data->h_interp);
    free_data(all_data->eta);
    free_data(all_data->u);
    free_data(all_data->v);
    free_data(all_data->h);
    free(all_data);
}