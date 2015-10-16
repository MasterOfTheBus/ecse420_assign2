#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

void get_rows_cols(char *filename, int* cols_rows);

int main(int argc, char* argv[]) {

  if (argc != 2) {
    printf("Usage: mpirun -np <number of processes> ./cp_mpi <path to text file of matrix>\n");
    return -1;
  }

  int size, rank;
  int columns, rows; 
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (rank == 0) {
    int cols_rows[2];
    get_rows_cols(argv[1], cols_rows);
    printf("columns: %d, rows: %d\n", cols_rows[0], cols_rows[1]);
    columns = cols_rows[0];
    rows = cols_rows[1];
  }

  MPI_Bcast(&columns, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&rows, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (rank != 0) {
    printf("columns: %d, rows: %d\n", columns, rows);
  }

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

/*
    // determine the portion of the matrix to take
    int start_row = *rows / size * rank;
    int end_row = *rows / size * (rank + 1); 
    if (rank == size - 1) end_row += *rows % size;
*/
    return;
}
