#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

void get_start_end_for_rank(int rank, int size, int rows, int* start_row, int* end_row);
double** allocate_contiguous_2d_double(int rows, int columns);
void free_contiguous_2d_double(double** array);
void gaussian_elimination(double** matrix_portion, int start_row, int end_row, int columns, double* pivot_row,
			  int pivot_row_num);
void RREF(double** matrix, int rows, int columns, int rank, int size, int *start_row, int *end_row);
void print_matrix(double** matrix, int rows, int columns);
double** read_user_matrix_from_file(char* filename, int* rows, int* columns);

int main(int argc, char* argv[]) {

  if (argc != 2) {
    printf("Usage: mpirun -np <number of processes> ./cp_mpi <path to text file of matrix>\n");
    return -1;
  }

  int size, rank;
  int columns, rows;
  int start_row, end_row;
  double **matrix;

  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (rank == 0) {
    matrix = read_user_matrix_from_file(argv[1], &rows, &columns);
  }

  // broadcast the column info
  MPI_Bcast(&columns, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&rows, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (rank != 0) {
    matrix = allocate_contiguous_2d_double(rows, columns);
  }

  RREF(matrix, rows, columns, rank, size, &start_row, &end_row);

  if (rank == 0) {
    printf("\n");
    print_matrix(matrix, rows, columns);
  }


  if (rank == 0)
    free_contiguous_2d_double(matrix);

  MPI_Finalize();

  return 0;
}

void get_start_end_for_rank(int rank, int size, int rows, int* start_row, int* end_row) {
  *start_row = rows / size * rank;
  *end_row = rows / size * (rank + 1); 
  if (rank == size - 1) *end_row += rows % size;
}

// passing multidimensional arrays through mpi requires the data to be contiguous
double** allocate_contiguous_2d_double(int rows, int columns) {
  double *data = (double*)malloc(rows * columns * sizeof(double));
  if (!data) printf("failed to malloc\n");
  double **array = (double**)malloc(rows * sizeof(double*));
  if (!array) printf("failed to malloc\n");
  int i;
  for (i = 0; i < rows; i++)
    array[i] = &(data[columns * i]);

  return array;
}

void free_contiguous_2d_double(double** matrix) {
  free(&(matrix[0][0]));
  free(matrix);
}

/*
    The strategy is to use a broadcast to inform the other processes of the rows
*/
void RREF(double** matrix, int rows, int columns, int rank, int size, int *start_row, int *end_row) {
  int scatter_array[size];
  int displ_array[size];

  // determine the portion of the matrix to take
  int i;
  for (i = 0; i < size; i++) {
    int start_temp, end_temp;
    get_start_end_for_rank(i, size, rows, &start_temp, &end_temp);
    scatter_array[i] = (end_temp - start_temp) * columns;
    displ_array[i] = i * (end_temp - start_temp) * columns;
    if (i == rank) {
      *start_row = start_temp;
      *end_row = end_temp;
    }
  }

  // for each row, broadcast it, then scatter the rest of the matrix
  double** reduce_rows = allocate_contiguous_2d_double(*end_row - *start_row, columns);

  for (i = 0; i < rows; i++) {
    int j;
    double bcast_row[columns];
    //    double** gather_recv = allocate_contiguous_2d_double(rows, columns);
    if (rank == 0) {
      for (j = 0; j < columns; j++) {
        bcast_row[j] = matrix[i][j];
      }
    }
    MPI_Bcast(&bcast_row, columns, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    MPI_Scatterv(&(matrix[0][0]), scatter_array, displ_array, MPI_DOUBLE,
                 &(reduce_rows[0][0]), scatter_array[rank], MPI_DOUBLE, 0,
                 MPI_COMM_WORLD);

    gaussian_elimination(reduce_rows, *start_row, *end_row, columns, bcast_row, i);

    MPI_Gatherv(&(reduce_rows[0][0]), scatter_array[rank], MPI_DOUBLE, &(matrix[0][0]),
                scatter_array, displ_array, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  }

  free_contiguous_2d_double(reduce_rows);
}

void gaussian_elimination(double** matrix_portion, int start_row, int end_row, int columns, double* pivot_row,
			  int pivot_row_num) {
  int dest_row, column;
  double pivot;
  for (dest_row = start_row; dest_row < end_row; dest_row++) {
    if (dest_row == pivot_row_num) continue;

    pivot = matrix_portion[dest_row - start_row][pivot_row_num] / pivot_row[pivot_row_num];

    for (column = 0; column < columns; column++) {
      matrix_portion[dest_row - start_row][column] = matrix_portion[dest_row - start_row][column] - pivot * pivot_row[column];
    }
  }
}

void print_matrix(double** matrix, int rows, int columns) {
  int i;
  for (i = 0; i < rows; i++) {
    int j;
    for (j = 0; j < columns; j++) {
      printf("%lf ", matrix[i][j]);
    }
    printf("\n");
  }
}

double ** read_user_matrix_from_file(char *filename, int *rows, int *columns) {
  FILE *file;
  file = fopen(filename, "r");

  /* get number of rows and columns*/
  *rows = 1;
  *columns = 1;
  char c;
  int columns_known = 0;
  while(!feof(file)) {
    c = fgetc(file);
    if (c == ' ') {
      if (!columns_known) (*columns)++;
    } 

    if (c == '\n') {
      (*rows)++;
      columns_known = 1;
      continue;
    }
  }

  /* read values into array */
  rewind(file);
  int i, j;
  double **matrix = allocate_contiguous_2d_double(*rows, *columns); // MPI passes assuming contiguous data
  for (i = 0; i < *rows; i++) {
    for (j = 0; j < *columns; j++) {
    fscanf(file,"%lf",&matrix[i][j]);
    }
  } 
  fclose(file);
  return matrix;
}
