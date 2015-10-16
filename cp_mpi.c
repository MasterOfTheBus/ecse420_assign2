#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

void get_rows_cols(char *filename, int* cols_rows);
double** allocate_matrix(int rowss, int columns);
void fill_matrix_portion(char *filename, double **matrix, int start_row, int end_row, int columns);

int main(int argc, char* argv[]) {

  if (argc != 2) {
    printf("Usage: mpirun -np <number of processes> ./cp_mpi <path to text file of matrix>\n");
    return -1;
  }

  int size, rank;
  int columns, rows;
  double **matrix; 
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (rank == 0) {
    int cols_rows[2];
    get_rows_cols(argv[1], cols_rows);
    columns = cols_rows[0];
    rows = cols_rows[1];
  }

  // broadcast the column and row info
  MPI_Bcast(&columns, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&rows, 1, MPI_INT, 0, MPI_COMM_WORLD);

  // determine the portion of the matrix to take
  int start_row = rows / size * rank;
  int end_row = rows / size * (rank + 1); 
  if (rank == size - 1) end_row += rows % size;

  matrix = allocate_matrix(end_row - start_row, columns);
  fill_matrix_portion(argv[1], matrix, start_row, end_row, columns);

  MPI_Finalize();

  return 0;
}

// cols_rows has element 0 as columns and element 1 as rows
void get_rows_cols(char *filename, int* cols_rows) {
    FILE *file;
    file = fopen(filename, "r");

    // get number of rows and columns
    cols_rows[1] = 1;
    cols_rows[0] = 1;
    int columns_known = 0;
    char c;
    while (!feof(file)) {
      c = fgetc(file);
      if (c == ' ') {
	if (!columns_known) (cols_rows[0])++;
      }

      if (c == '\n') {
	(cols_rows[1])++;
	columns_known = 1;
      }
    }

    fclose(file);

    return;
}

double ** allocate_matrix(int rows, int cols)
{
  int i = 0;
  double ** matrix = (double **) malloc(rows * sizeof(double *));

  for (i = 0; i < rows; i++) {
    matrix[i] = (double *) malloc(cols * sizeof(double));
  }

  return matrix;
}

void fill_matrix_portion(char *filename, double **matrix, int start_row, int end_row, int columns) {
  FILE *file;
  file = fopen(filename, "r");

  // position the file pointer first to the right row
  int i = 0;
  char c;
  while (i != start_row) {
    c = fgetc(file);
    if (c == '\n') i++;
  }

  // read in the data
  i = 0;
  int rows = end_row - start_row;
  for (; i < rows; i++) {
    int j = 0;
    for (; j < columns; j++) {
      fscanf(file, "%lf", &matrix[i][j]);
    }
  }

  fclose(file);

  return;
}
