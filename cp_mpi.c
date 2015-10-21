#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

void get_start_end_for_rank(int rank, int size, int rows, int* start_row, int* end_row);
double** allocate_contiguous_2d_double(int rows, int columns);
void free_contiguous_2d_double(double** array);


/*void gaussian_elimination(double** src_matrix, double** dest_matrix, int src_row_start, int src_row_end,
			  int dest_row_start, int dest_row_end, int columns);*/
void gaussian_elimination(double** matrix_portion, int start_row, int end_row, int columns, double* pivot_row,
			  int pivot_row_num);


void RREF(double** matrix, int start_row, int end_row, int rows, int columns, int rank, int size);
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
      start_row = start_temp;
      end_row = end_temp;
    }
  }

  // for each row, broadcast it, then scatter the rest of the matrix
  double** reduce_rows = allocate_contiguous_2d_double(end_row - start_row, columns);
  //double reduce_rows[(end_row - start_row) * columns];

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

  //  print_matrix(reduce_rows, end_row - start_row, columns);

    gaussian_elimination(reduce_rows, start_row, end_row, columns, bcast_row, i);

  //RREF(matrix, start_row, end_row, rows, columns, rank, size);

    MPI_Gatherv(&(reduce_rows[0][0]), scatter_array[rank], MPI_DOUBLE, &(matrix[0][0]),
                scatter_array, displ_array, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    //    free_contiguous_2d_double(gather_recv);

    if (rank == 0) {
      printf("\n");
      print_matrix(matrix, rows, columns);
    }
  }

  if (rank == 0) {
    printf("\n");
    print_matrix(matrix, rows, columns);
  }

  free_contiguous_2d_double(reduce_rows);

  if (rank == 0)
    free_contiguous_2d_double(matrix);

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
void RREF(double** matrix, int start_row, int end_row, int rows, int columns, int rank, int size) {
  //gaussian_elimination(matrix, matrix, start_row, end_row, start_row, end_row, columns);

  // broadcast rows to send to other processes
  int i;
  for (i = 0; i < size; i++) {
//printf("iteration %d of %d\n", i, size);

    int start_i, end_i;
    get_start_end_for_rank(i, size, rows, &start_i, &end_i);
    //printf("%d, %d, for %d and size %d\n", start_i, end_i, i, size);
    double **matrix_portion = allocate_contiguous_2d_double(end_i - start_i, columns);
    int matrix_size = (end_i - start_i) * columns;
    if (rank == i) {
      // copy in the matrix portion
      int index;
      for (index = 0; index < end_i - start_i; index++) {
        int j;
	for (j = 0; j < columns; j++) {
          matrix_portion[index][j] = matrix[index][j];
	}
      }
    }

    MPI_Bcast(&(matrix_portion[0][0]), end_i - start_i, MPI_DOUBLE, i, MPI_COMM_WORLD);

   // if (rank != i)
      //gaussian_elimination(matrix_portion, matrix, start_i, end_i, start_row, end_row, columns);

    if (rank == 0) printf("\n\nprinting for iteration %d\n", i);
    MPI_Barrier(MPI_COMM_WORLD);
    print_matrix(matrix, end_row - start_row, columns);
    MPI_Barrier(MPI_COMM_WORLD);

    free_contiguous_2d_double(matrix_portion);
  }

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

/*
void gaussian_elimination(double** src_matrix, double** dest_matrix, int src_row_start, int src_row_end,
			  int dest_row_start, int dest_row_end, int columns) {
  int src_row, dest_row, column;
  double pivot;
  for (src_row = src_row_start; src_row < src_row_end; src_row++) {
    for (dest_row = dest_row_start; dest_row < dest_row_end; dest_row++) {
      if (dest_row == src_row) continue;

    printf("src %d, dest %d\n", src_row, dest_row);

      double numerator = dest_matrix[dest_row - dest_row_start][src_row];
      double denominator = src_matrix[src_row - src_row_start][src_row];
 
      // check that numerator and denominator != 0, but need to have a delta for floats
      if (((numerator <= 0.0000001) && (numerator >= -0.0000001)) ||
	  ((denominator <= 0.0000001) && (denominator >= -0.0000001)))
	  continue;
      
      pivot = numerator / denominator;
      
      for (column = src_row; column < columns; column++) {
      printf("substracting (%d, %d) from (%d, %d)\n", src_row, column, dest_row, column);
        dest_matrix[dest_row - dest_row_start][column] = dest_matrix[dest_row - dest_row_start][column] - pivot * src_matrix[src_row - src_row_start][column];
      }
    }
  }
}*/

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
