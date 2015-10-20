#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

void get_rows_cols(char *filename, int* cols_rows);
double** allocate_matrix(int rows, int columns);
void get_start_end_for_rank(int rank, int size, int rows, int* start_row, int* end_row);
void fill_matrix_portion(char *filename, double **matrix, int start_row, int end_row, int columns);
double** allocate_contiguous_2d_double(int rows, int columns);
void free_contiguous_2d_double(double** array);
void gaussian_elimination(double** src_matrix, double** dest_matrix, int src_row_start, int src_row_end,
			  int dest_row_start, int dest_row_end, int columns);
void RREF(double** matrix, int start_row, int end_row, int rows, int columns, int rank, int size);
void print_matrix(double** matrix, int rows, int columns, int rank, int size);
void free_matrix(double** matrix, int rows);

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
    int cols_rows[2];
    get_rows_cols(argv[1], cols_rows);
    columns = cols_rows[0];
    rows = cols_rows[1];
  }

  // broadcast the column and row info
  MPI_Bcast(&columns, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&rows, 1, MPI_INT, 0, MPI_COMM_WORLD);

  // determine the portion of the matrix to take
  get_start_end_for_rank(rank, size, rows, &start_row, &end_row);

  matrix = allocate_matrix(end_row - start_row, columns);
  fill_matrix_portion(argv[1], matrix, start_row, end_row, columns);

  print_matrix(matrix, end_row - start_row, columns, rank, size);

  RREF(matrix, start_row, end_row, rows, columns, rank, size);

  MPI_Barrier(MPI_COMM_WORLD);

  if (rank == 0) printf("\n\n");
  print_matrix(matrix, end_row - start_row, columns, rank, size);

  free_matrix(matrix, end_row - start_row);

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

void get_start_end_for_rank(int rank, int size, int rows, int* start_row, int* end_row) {
  *start_row = rows / size * rank;
  *end_row = rows / size * (rank + 1); 
  if (rank == size - 1) *end_row += rows % size;
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
void RREF(double** matrix, int start_row, int end_row, int rows, int columns, int rank, int size) {
    /*int src_row, dest_row, row, column, i;
    double pivot;
    for (src_row = start_row; src_row < end_row; src_row++) {
      for (dest_row = start_row; dest_row < end_row; dest_row++) {
	if (dest_row == src_row) continue;

	pivot = matrix[dest_row-start_row][src_row] / matrix[src_row-start_row][src_row];
	for (column = src_row; column < columns; column++) {
	  matrix[dest_row-start_row][column] = matrix[dest_row-start_row][column] - pivot * matrix[src_row-start_row][column];
	}
      }
    }*/

  //gaussian_elimination(matrix, matrix, start_row, end_row, start_row, end_row, columns);

//printf("rank %d with %d ranks\n", rank, size);

  // broadcast rows to send to other processes
  int i;
  for (i = 0; i < size; i++) {
//printf("iteration %d of %d\n", i, size);

    int start_i, end_i;
    get_start_end_for_rank(i, size, rows, &start_i, &end_i);
    //printf("%d, %d, for %d and size %d\n", start_i, end_i, i, size);
    double **matrix_portion = allocate_contiguous_2d_double(end_i - start_i, columns);
    //printf("allocated for rank %d from rank %d\n", i, rank);
    int matrix_size = (end_i - start_i) * columns;
    if (rank == i) {
      // copy in the matrix portion
      int index;
      //printf("\n\nassigning the values\n");
      for (index = 0; index < end_i - start_i; index++) {
        int j;
	for (j = 0; j < columns; j++) {
          matrix_portion[index][j] = matrix[index][j];
	  //printf("%lf ", matrix_portion[index][j]);
	}
	//printf("\n");
      }
      //printf("\n\n");
    }

    MPI_Bcast(&(matrix_portion[0][0]), end_i - start_i, MPI_DOUBLE, i, MPI_COMM_WORLD);

   // if (rank != i)
      gaussian_elimination(matrix_portion, matrix, start_i, end_i, start_row, end_row, columns);

    free_contiguous_2d_double(matrix_portion);
  }

}

void gaussian_elimination(double** src_matrix, double** dest_matrix, int src_row_start, int src_row_end,
			  int dest_row_start, int dest_row_end, int columns) {
  int src_row, dest_row, column;
  double pivot;
  for (src_row = src_row_start; src_row < src_row_end; src_row++) {
    for (dest_row = dest_row_start; dest_row < dest_row_end; dest_row++) {
      if (src_matrix == dest_matrix && dest_row == src_row) continue;

//    printf("src %d, dest %d\n", src_row, dest_row);

      double numerator = dest_matrix[dest_row - dest_row_start][src_row];
      double denominator = src_matrix[src_row - src_row_start][src_row];
 
      // check that numerator and denominator != 0, but need to have a delta for floats
      if (((numerator <= 0.0000001) && (numerator >= -0.0000001)) ||
	  ((denominator <= 0.0000001) && (denominator >= -0.0000001)))
	  continue;
      
      pivot = numerator / denominator;
      //printf("%lf / %lf = %lf\n", numerator, denominator, pivot);
      for (column = src_row; column < columns; column++) {
        dest_matrix[dest_row - dest_row_start][column] = dest_matrix[dest_row - dest_row_start][column] - pivot * src_matrix[src_row - src_row_start][column];
	//printf("%lf ", dest_matrix[dest_row - dest_row_start][column]);
      }
      //printf("\n");
    }
  }
}

void print_matrix(double** matrix, int rows, int columns, int rank, int size) {
  MPI_Status status;
  int temp = 0;
  if (rank != 0) {
    MPI_Recv(&temp, 1, MPI_INT, rank - 1, rank - 1, MPI_COMM_WORLD, &status);
  }
  int i;
  for (i = 0; i < rows; i++) {
    int j;
    for (j = 0; j < columns; j++) {
      printf("%lf ", matrix[i][j]);
    }
    printf("\n");
  }
  if (rank != size - 1) {
    MPI_Send(&temp, 1, MPI_INT, rank + 1, rank, MPI_COMM_WORLD);
  }
}

void free_matrix(double** matrix, int rows) {
  int i;
  for (i = 0; i < rows; i++) free(matrix[i]);
  free(matrix);
}
