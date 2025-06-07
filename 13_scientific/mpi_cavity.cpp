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

// --- ゴーストセル交換関数 ---
// 引数:
//   data: 交換を行う行列 (u, v, または p)
//   ny: 行列のy方向のサイズ
//   nx: 行列のx方向のサイズ
//   rank: 現在のMPIプロセスのランク
//   size: 全体のMPIプロセスの数
//   begin_idx: このプロセスが担当するx方向の開始インデックス
//   end_idx: このプロセスが担当するx方向の終了インデックス (排他的)
//   tag_offset: MPIタグの衝突を避けるためのオフセット
void exchange_ghost_cells(matrix& data, int ny, int nx, int rank, int size,
                          int begin_idx, int end_idx, int tag_offset) {
    MPI_Request request[4];
    MPI_Status status[4];
    int send_count = ny; // 1列分のデータ数

    // 左隣のプロセスとデータを交換 (begin_idx - 1 のゴーストセルを更新)
    if (rank > 0) {
        // 左に自分の左端のデータ (begin_idx) を送信
        MPI_Isend(&data[0][begin_idx], send_count, MPI_FLOAT, rank - 1, 0 + tag_offset, MPI_COMM_WORLD, &request[0]);
        // 左から右隣のデータ (begin_idx - 1) を受信
        MPI_Irecv(&data[0][begin_idx - 1], send_count, MPI_FLOAT, rank - 1, 1 + tag_offset, MPI_COMM_WORLD, &request[1]);
    }

    // 右隣のプロセスとデータを交換 (end_idx のゴーストセルを更新)
    if (rank < size - 1) {
        // 右に自分の右端のデータ (end_idx - 1) を送信
        MPI_Isend(&data[0][end_idx - 1], send_count, MPI_FLOAT, rank + 1, 1 + tag_offset, MPI_COMM_WORLD, &request[2]);
        // 右から左隣のデータ (end_idx) を受信
        MPI_Irecv(&data[0][end_idx], send_count, MPI_FLOAT, rank + 1, 0 + tag_offset, MPI_COMM_WORLD, &request[3]);
    }

    // 非同期通信の完了を待つ
    // MPI_Waitallの引数に注意し、必要なリクエストのみを待機するように変更
    if (rank > 0 && rank < size - 1) { // 途中のプロセス
        MPI_Waitall(4, request, status);
    } else if (rank > 0) { // 最も右のプロセス
        MPI_Waitall(2, request, status); // request[0]とrequest[1]のみ
    } else if (rank < size - 1) { // 最も左のプロセス
        MPI_Waitall(2, request + 2, status + 2); // request[2]とrequest[3]のみ
    }
}

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);

    int size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int nx = 41;
    int ny = 41;
    int nt = 500;
    int nit = 50;
    double dx = 2. / (nx - 1);
    double dy = 2. / (ny - 1);
    double dt = .01;
    double rho = 1.;
    double nu = .02;

    int local_nx_base = nx / size;
    int remainder = nx % size;

    int begin_idx;
    int end_idx;

    if (rank < remainder)
    {
        begin_idx = rank * (local_nx_base + 1);
        end_idx = begin_idx + (local_nx_base + 1);
    }
    else
    {
        begin_idx = rank * local_nx_base + remainder;
        end_idx = begin_idx + local_nx_base;
    }

    matrix u(ny, vector<float>(nx));
    matrix v(ny, vector<float>(nx));
    matrix p(ny, vector<float>(nx));
    matrix b(ny, vector<float>(nx));
    matrix un(ny, vector<float>(nx));
    matrix vn(ny, vector<float>(nx));
    matrix pn(ny, vector<float>(nx));

    for (int j = 0; j < ny; j++)
    {
        for (int i = begin_idx; i < end_idx; i++)
        {
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

    for (int n = 0; n < nt; n++)
    {
        for (int j = 1; j < ny - 1; j++)
        {
            for (int i = max(1, begin_idx); i < min(nx - 1, end_idx); i++)
            {
                b[j][i] = rho * (1. / dt * ((u[j][i + 1] - u[j][i - 1]) / (2. * dx) + (v[j + 1][i] - v[j - 1][i]) / (2. * dy)) -
                                 pow((u[j][i + 1] - u[j][i - 1]) / (2. * dx), 2) -
                                 2. * ((u[j + 1][i] - u[j - 1][i]) / (2. * dy) * (v[j][i + 1] - v[j - 1][i]) / (2. * dx)) -
                                 pow((v[j + 1][i] - v[j - 1][i]) / (2. * dy), 2));
            }
        }

        for (int it = 0; it < nit; it++)
        {
            for (int j = 0; j < ny; j++)
            {
                for (int i = begin_idx; i < end_idx; i++)
                {
                    pn[j][i] = p[j][i];
                }
            }

            // Pのゴーストセル交換を関数呼び出しで実行
            // タグの衝突を避けるためにオフセット(0)を使用
            exchange_ghost_cells(pn, ny, nx, rank, size, begin_idx, end_idx, 0);

            for (int j = 1; j < ny - 1; j++)
            {
                for (int i = max(1, begin_idx); i < min(nx - 1, end_idx); i++)
                {
                    p[j][i] = (dy * dy * (pn[j][i + 1] + pn[j][i - 1]) +
                               dx * dx * (pn[j + 1][i] + pn[j - 1][i]) -
                               b[j][i] * dx * dx * dy * dy) /
                              (2. * (dx * dx + dy * dy));
                }
            }
            if (rank == 0)
            {
                for (int j = 0; j < ny; j++)
                {
                    p[j][0] = p[j][1];
                }
            }
            if (rank == size - 1)
            {
                for (int j = 0; j < ny; j++)
                {
                    p[j][nx - 1] = p[j][nx - 2];
                }
            }
            for (int i = begin_idx; i < end_idx; i++)
            {
                p[0][i] = p[1][i];
                p[ny - 1][i] = 0;
            }
        } // end of nit loop

        for (int j = 0; j < ny; j++)
        {
            for (int i = begin_idx; i < end_idx; i++)
            {
                un[j][i] = u[j][i];
                vn[j][i] = v[j][i];
            }
        }

        // Uのゴーストセル交換を関数呼び出しで実行
        // タグの衝突を避けるためにオフセット(10)を使用
        exchange_ghost_cells(un, ny, nx, rank, size, begin_idx, end_idx, 10);
        // Vのゴーストセル交換を関数呼び出しで実行
        // タグの衝突を避けるためにオフセット(20)を使用
        exchange_ghost_cells(vn, ny, nx, rank, size, begin_idx, end_idx, 20);

        for (int j = 1; j < ny - 1; j++)
        {
            for (int i = max(1, begin_idx); i < min(nx - 1, end_idx); i++)
            {
                u[j][i] = un[j][i] - un[j][i] * dt / dx * (un[j][i] - un[j][i - 1]) - vn[j][i] * dt / dy * (un[j][i] - un[j - 1][i]) - dt / (2. * rho * dx) * (p[j][i + 1] - p[j][i - 1]) + nu * dt / (dx * dx) * (un[j][i + 1] - 2. * un[j][i] + un[j][i - 1]) + nu * dt / (dy * dy) * (un[j + 1][i] - 2. * un[j][i] + un[j - 1][i]);

                v[j][i] = vn[j][i] - un[j][i] * dt / dx * (vn[j][i] - vn[j][i - 1]) - vn[j][i] * dt / dy * (vn[j][i] - vn[j - 1][i]) - dt / (2. * rho * dy) * (p[j + 1][i] - p[j - 1][i])
                                + nu * dt / (dx * dx) * (vn[j][i + 1] - 2. * vn[j][i] + vn[j][i - 1]) + nu * dt / (dy * dy) * (vn[j + 1][i] - 2. * vn[j][i] + vn[j - 1][i]);
            }
        }
        if (rank == 0)
        {
            for (int j = 0; j < ny; j++)
            {
                u[j][0] = 0;
                v[j][0] = 0;
            }
        }
        if (rank == size - 1)
        {
            for (int j = 0; j < ny; j++)
            {
                u[j][nx - 1] = 0;
                v[j][nx - 1] = 0;
            }
        }
        for (int i = begin_idx; i < end_idx; i++)
        {
            u[0][i] = 0;
            u[ny - 1][i] = 1;
            v[0][i] = 0;
            v[ny - 1][i] = 0;
        }

        if (n % 10 == 0)
        {
            if (rank == 0)
            {
                for (int j = 0; j < ny; j++)
                {
                    for (int i = 0; i < nx; i++)
                        ufile << u[j][i] << " ";
                    ufile << "\n";
                }
                for (int j = 0; j < ny; j++)
                {
                    for (int i = 0; i < nx; i++)
                        vfile << v[j][i] << " ";
                    vfile << "\n";
                }
                for (int j = 0; j < ny; j++)
                {
                    for (int i = 0; i < nx; i++)
                        pfile << p[j][i] << " ";
                    pfile << "\n";
                }
            }
        }
    } // end of nt loop

    if (rank == 0) {
        ufile.close();
        vfile.close();
        pfile.close();
    }

    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(stop - start);
    if (rank == 0)
    {
        printf("Elapsed time: %lld ms\n", duration.count());
    }

    MPI_Finalize();
    return 0;
}
