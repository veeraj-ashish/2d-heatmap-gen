#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <math.h>

#define MAX_CSV_LINE_SIZE 100
#define CSV_FILENAME "sampled_points.csv"

typedef struct {
    double x;
    double y;
    double temperature;
} SampledPoint;

double calculate_distance(double x1, double y1, double x2, double y2) {
    return sqrt(pow(x1 - x2, 2) + pow(y1 - y2, 2));
}

double inverse_distance_weighting(double target_x, double target_y, SampledPoint sampled_points[], int num_sampled_points, double p) {
    double numerator = 0;
    double denominator = 0;

    for (int i = 0; i < num_sampled_points; i++) {
        double distance = calculate_distance(target_x, target_y, sampled_points[i].x, sampled_points[i].y);
        if (distance == 0) {
            return sampled_points[i].temperature;
        }
        double weight = 1.0 / pow(distance, p);
        numerator += weight * sampled_points[i].temperature;
        denominator += weight;
    }

    if (denominator == 0) {
        return 0;
    } 
    else {
        return numerator / denominator;
    }
}

void print_heat_map(double target_x, double target_y, double estimated_temperature) {
    int color;

    if (estimated_temperature < 15.0) {
        color = 44 + (int)((estimated_temperature - (int) estimated_temperature) * 4);
    } else if (estimated_temperature < 25.0) {
        color = 42 + (int)((estimated_temperature - (int) estimated_temperature) * 4);
    } else {
        color = 41 + (int)((estimated_temperature - (int) estimated_temperature) * 4);
    }

    printf("\033[%dm[%2.1f]\033[0m", color, estimated_temperature);
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int world_size, world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    int GRID_SIZE;
    if (world_rank == 0) {
        printf("Enter the grid size: ");
        scanf("%d", &GRID_SIZE);
    }

    MPI_Bcast(&GRID_SIZE, 1, MPI_INT, 0, MPI_COMM_WORLD);

    FILE *csv_file = fopen(CSV_FILENAME, "r");
    if (csv_file == NULL) {
        fprintf(stderr, "Could not open CSV file.\n");
        MPI_Finalize();
        return 1;
    }

    int num_sampled_points = 0;
    SampledPoint sampled_points[GRID_SIZE * GRID_SIZE];
    char line[MAX_CSV_LINE_SIZE];
    while (fgets(line, sizeof(line), csv_file) != NULL) {
        sscanf(line, "%lf,%lf,%lf", &sampled_points[num_sampled_points].x, &sampled_points[num_sampled_points].y, &sampled_points[num_sampled_points].temperature);
        num_sampled_points++;
    }
    fclose(csv_file);

    int num_segments = world_size;
    int segment_size = GRID_SIZE / num_segments;

    int start_row = world_rank * segment_size;
    int end_row = start_row + segment_size;

    printf("Process %d generating segment for rows %d to %d.\n", world_rank, start_row, end_row);

    for (int i = start_row; i < end_row; i++) {
        for (int j = 0; j < GRID_SIZE; j++) {
            double estimated_temperature = inverse_distance_weighting(i, j, sampled_points, num_sampled_points, 2.0);
            print_heat_map(i, j, estimated_temperature);
        }
        printf("\n");
    }

    MPI_Finalize();
    return 0;
}