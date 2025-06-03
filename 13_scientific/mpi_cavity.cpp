#include <cstdlib>
#include <cstdio>
#include <fstream>
#include <vector>
#include <cmath>
#include <chrono>
#include <mpi.h>

using namespace std::chrono;
using namespace std;

typedef vector<vector<float>> matrix;

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank;
    int size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int nx = 41;
    int ny = 41;
    int nt = 500;
    int nit = 50;
    double dx = 2. / (nx - 1);
    double dy = 2. / (ny - 1);
    double dt = .01;
    double rho = 1.;
    double nu = .02;

    int local_nx = nx / size;
    int remainder_x = nx % size;
    
    if (rank < remainder_x) {
        local_nx++;
    }

    int local_nx_with_ghost = local_nx;
    if (rank != 0) local_nx_with_ghost++;
    if (rank != size - 1) local_nx_with_ghost++;

    matrix u(ny, vector<float>(local_nx_with_ghost));
    matrix v(ny, vector<float>(local_nx_with_ghost));
    matrix p(ny, vector<float>(local_nx_with_ghost));
    matrix b(ny, vector<float>(local_nx_with_ghost));
    matrix un(ny, vector<float>(local_nx_with_ghost));
    matrix vn(ny, vector<float>(local_nx_with_ghost));
    matrix pn(ny, vector<float>(local_nx_with_ghost));

    for (int j = 0; j < ny; j++) {
        for (int i = 0; i < local_nx_with_ghost; i++) {
            u[j][i] = 0;
            v[j][i] = 0;
            p[j][i] = 0;
            b[j][i] = 0;
        }
    }

    auto start = high_resolution_clock::now();

    ofstream ufile, vfile, pfile;
    if (rank == 0) {
        ufile.open("u.dat");
        vfile.open("v.dat");
        pfile.open("p.dat");
    }

    for (int n = 0; n < nt; n++) {
        int current_start_x_local = (rank == 0) ? 1 : 1;
        int current_end_x_local = (rank == size - 1) ? local_nx_with_ghost - 2 : local_nx_with_ghost - 2;

        for (int j = 1; j < ny - 1; j++) {
            for (int i = current_start_x_local; i <= current_end_x_local; i++) {
                b[j][i] = rho * (1. / dt * ((u[j][i+1] - u[j][i-1]) / (2. * dx) + (v[j+1][i] - v[j-1][i]) / (2. * dy)) -
                                  pow((u[j][i+1] - u[j][i-1]) / (2. * dx), 2) -
                                  2. * ((u[j+1][i] - u[j-1][i]) / (2. * dy) * (v[j][i+1] - v[j][i-1]) / (2. * dx)) -
                                  pow((v[j+1][i] - v[j-1][i]) / (2. * dy), 2));
            }
        }

        for (int it = 0; it < nit; it++) {
            for (int j = 0; j < ny; j++) {
                for (int i = 0; i < local_nx_with_ghost; i++) {
                    pn[j][i] = p[j][i];
                }
            }

            MPI_Status status; // MPI_Status変数を宣言

            for (int j = 0; j < ny; j++) {
                if (rank > 0) {
                    MPI_Sendrecv(&pn[j][1], 1, MPI_FLOAT, rank - 1, 0,
                                 &pn[j][0], 1, MPI_FLOAT, rank - 1, 0, MPI_COMM_WORLD, &status);
                }
                if (rank < size - 1) {
                    MPI_Sendrecv(&pn[j][local_nx_with_ghost - 2], 1, MPI_FLOAT, rank + 1, 0,
                                 &pn[j][local_nx_with_ghost - 1], 1, MPI_FLOAT, rank + 1, 0, MPI_COMM_WORLD, &status);
                }
            }

            for (int j = 1; j < ny - 1; j++) {
                for (int i = current_start_x_local; i <= current_end_x_local; i++) {
                    p[j][i] = (dy*dy * (pn[j][i+1] + pn[j][i-1]) +
                               dx*dx * (pn[j+1][i] + pn[j-1][i]) -
                               b[j][i] * dx*dx * dy*dy) / (2. * (dx*dx + dy*dy));
                }
            }

            for (int i = 0; i < local_nx_with_ghost; i++) {
                p[0][i] = p[1][i];
                p[ny-1][i] = 0;
            }

            if (rank == 0) {
                for (int j = 0; j < ny; j++) {
                    p[j][0] = p[j][1];
                }
            }
            if (rank == size - 1) {
                for (int j = 0; j < ny; j++) {
                    p[j][local_nx_with_ghost - 1] = p[j][local_nx_with_ghost - 2];
                }
            }
        }

        for (int j = 0; j < ny; j++) {
            for (int i = 0; i < local_nx_with_ghost; i++) {
                un[j][i] = u[j][i];
                vn[j][i] = v[j][i];
            }
        }

        MPI_Status status; // MPI_Status変数を宣言

        for (int j = 0; j < ny; j++) {
            if (rank > 0) {
                MPI_Sendrecv(&un[j][1], 1, MPI_FLOAT, rank - 1, 0,
                             &un[j][0], 1, MPI_FLOAT, rank - 1, 0, MPI_COMM_WORLD, &status);
                MPI_Sendrecv(&vn[j][1], 1, MPI_FLOAT, rank - 1, 0,
                             &vn[j][0], 1, MPI_FLOAT, rank - 1, 0, MPI_COMM_WORLD, &status);
            }
            if (rank < size - 1) {
                MPI_Sendrecv(&un[j][local_nx_with_ghost - 2], 1, MPI_FLOAT, rank + 1, 0,
                             &un[j][local_nx_with_ghost - 1], 1, MPI_FLOAT, rank + 1, 0, MPI_COMM_WORLD, &status);
                MPI_Sendrecv(&vn[j][local_nx_with_ghost - 2], 1, MPI_FLOAT, rank + 1, 0,
                             &vn[j][local_nx_with_ghost - 1], 1, MPI_FLOAT, rank + 1, 0, MPI_COMM_WORLD, &status);
            }
        }

        for (int j = 1; j < ny - 1; j++) {
            for (int i = current_start_x_local; i <= current_end_x_local; i++) {
                u[j][i] = un[j][i] - un[j][i] * dt / dx * (un[j][i] - un[j][i-1])
                                 - vn[j][i] * dt / dy * (un[j][i] - un[j-1][i])
                                 - dt / (2. * rho * dx) * (p[j][i+1] - p[j][i-1])
                                 + nu * dt / (dx*dx) * (un[j][i+1] - 2. * un[j][i] + un[j][i-1])
                                 + nu * dt / (dy*dy) * (un[j+1][i] - 2. * un[j][i] + un[j-1][i]);

                v[j][i] = vn[j][i] - un[j][i] * dt / dx * (vn[j][i] - vn[j][i-1])
                                 - vn[j][i] * dt / dy * (vn[j][i] - vn[j-1][i])
                                 - dt / (2. * rho * dy) * (p[j+1][i] - p[j-1][i])
                                 + nu * dt / (dx*dx) * (vn[j][i+1] - 2. * vn[j][i] + vn[j][i-1])
                                 + nu * dt / (dy*dy) * (vn[j+1][i] - 2. * vn[j][i] + vn[j-1][i]);
            }
        }

        for (int i = 0; i < local_nx_with_ghost; i++) {
            u[0][i] = 0;
            u[ny-1][i] = 1;
            v[0][i] = 0;
            v[ny-1][i] = 0;
        }

        if (rank == 0) {
            for (int j = 0; j < ny; j++) {
                u[j][0] = 0;
                v[j][0] = 0;
            }
        }
        if (rank == size - 1) {
            for (int j = 0; j < ny; j++) {
                u[j][local_nx_with_ghost - 1] = 0;
                v[j][local_nx_with_ghost - 1] = 0;
            }
        }

        MPI_Barrier(MPI_COMM_WORLD);

        if (n % 10 == 0) {
            vector<float> gather_u_row(nx);
            vector<float> gather_v_row(nx);
            vector<float> gather_p_row(nx);

            for (int j = 0; j < ny; j++) {
                MPI_Gather(&u[j][(rank == 0) ? 0 : 1],
                           local_nx, MPI_FLOAT,
                           gather_u_row.data(),
                           local_nx, MPI_FLOAT,
                           0, MPI_COMM_WORLD);

                MPI_Gather(&v[j][(rank == 0) ? 0 : 1],
                           local_nx, MPI_FLOAT,
                           gather_v_row.data(),
                           local_nx, MPI_FLOAT,
                           0, MPI_COMM_WORLD);

                MPI_Gather(&p[j][(rank == 0) ? 0 : 1],
                           local_nx, MPI_FLOAT,
                           gather_p_row.data(),
                           local_nx, MPI_FLOAT,
                           0, MPI_COMM_WORLD);

                if (rank == 0) {
                    for (int i = 0; i < nx; i++)
                        ufile << gather_u_row[i] << " ";
                    ufile << "\n";
                    for (int i = 0; i < nx; i++)
                        vfile << gather_v_row[i] << " ";
                    vfile << "\n";
                    for (int i = 0; i < nx; i++)
                        pfile << gather_p_row[i] << " ";
                    pfile << "\n";
                }
            }
        }
    }

    if (rank == 0) {
        ufile.close();
        vfile.close();
        pfile.close();
    }

    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(stop - start);
    
    if (rank == 0) {
        printf("Elapsed time: %lld ms\n", duration.count());
    }

    MPI_Finalize();
    return 0;
}
