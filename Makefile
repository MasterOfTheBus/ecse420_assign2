cp_mpi: cp_mpi.c
	mpicc cp_mpi.c -o cp_mpi

clean:
	rm cp_mpi
