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

    // 各プロセスが担当するX方向のグリッド数 (内部グリッドのみ)
    int local_nx_base = nx / size;
    int remainder_x = nx % size;
    int local_nx = local_nx_base;
    if (rank < remainder_x) {
        local_nx++;
    }

    // 各プロセスの開始Xインデックス (グローバル座標)
    int global_start_x = 0;
    for (int i = 0; i < rank; ++i) {
        int prev_local_nx = local_nx_base;
        if (i < remainder_x) {
            prev_local_nx++;
        }
        global_start_x += prev_local_nx;
    }
    int global_end_x = global_start_x + local_nx - 1;

    // ゴーストセルを含むX方向のグリッド数
    // 左端プロセスは右に1つ、右端プロセスは左に1つ、中間プロセスは両側に1つずつゴーストセルを持つ
    int local_nx_with_ghost = local_nx;
    int ghost_left = 0;
    int ghost_right = 0;

    if (rank != 0) { // 左側にゴーストセル
        local_nx_with_ghost++;
        ghost_left = 1;
    }
    if (rank != size - 1) { // 右側にゴーストセル
        local_nx_with_ghost++;
        ghost_right = 1;
    }
    
    // 計算対象のローカルインデックスの範囲
    int compute_start_x_local = ghost_left;
    int compute_end_x_local = local_nx_with_ghost - 1 - ghost_right;

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

    // 初期境界条件の適用 (ゴーストセルを含む全領域で初期化)
    // u, v の境界条件は時間ステップごとに適用されるため、ここでは初期化のみ
    // p の境界条件も反復ごとに適用されるため、ここでは初期化のみ

    auto start = high_resolution_clock::now();

    ofstream ufile, vfile, pfile;
    if (rank == 0) {
        ufile.open("u.dat");
        vfile.open("v.dat");
        pfile.open("p.dat");
    }

    // MPI_Gathervのための準備
    vector<int> recvcounts(size);
    vector<int> displs(size);
    for (int i = 0; i < size; ++i) {
        int current_rank_local_nx = local_nx_base;
        if (i < remainder_x) {
            current_rank_local_nx++;
        }
        recvcounts[i] = current_rank_local_nx;
        displs[i] = 0; // 各ランクの開始位置
        for (int k = 0; k < i; ++k) {
            int prev_local_nx = local_nx_base;
            if (k < remainder_x) {
                prev_local_nx++;
            }
            displs[i] += prev_local_nx;
        }
    }
    
    vector<float> gather_u_row(nx);
    vector<float> gather_v_row(nx);
    vector<float> gather_p_row(nx);

    for (int n = 0; n < nt; n++) {
        // b の計算
        for (int j = 1; j < ny - 1; j++) {
            for (int i = compute_start_x_local; i <= compute_end_x_local; i++) {
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

            // p のゴーストセル交換
            for (int j = 0; j < ny; j++) {
                if (rank > 0) { // 左隣から受け取り、左隣へ送信
                    MPI_Sendrecv(&pn[j][ghost_left], 1, MPI_FLOAT, rank - 1, 0,
                                 &pn[j][0], 1, MPI_FLOAT, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                }
                if (rank < size - 1) { // 右隣から受け取り、右隣へ送信
                    MPI_Sendrecv(&pn[j][local_nx_with_ghost - 1 - ghost_right], 1, MPI_FLOAT, rank + 1, 0,
                                 &pn[j][local_nx_with_ghost - 1], 1, MPI_FLOAT, rank + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                }
            }
            MPI_Barrier(MPI_COMM_WORLD); // 通信が完了するまで待機

            // p の計算
            for (int j = 1; j < ny - 1; j++) {
                for (int i = compute_start_x_local; i <= compute_end_x_local; i++) {
                    p[j][i] = (dy*dy * (pn[j][i+1] + pn[j][i-1]) +
                               dx*dx * (pn[j+1][i] + pn[j-1][i]) -
                               b[j][i] * dx*dx * dy*dy) / (2. * (dx*dx + dy*dy));
                }
            }

            // p の境界条件 (y方向)
            for (int i = 0; i < local_nx_with_ghost; i++) {
                p[0][i] = p[1][i];
                p[ny-1][i] = 0;
            }

            // p の境界条件 (x方向) - グローバルな境界のみ適用
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
        
        // u, v のゴーストセル交換
        for (int j = 0; j < ny; j++) {
            if (rank > 0) { // 左隣から受け取り、左隣へ送信
                MPI_Sendrecv(&un[j][ghost_left], 1, MPI_FLOAT, rank - 1, 0,
                             &un[j][0], 1, MPI_FLOAT, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Sendrecv(&vn[j][ghost_left], 1, MPI_FLOAT, rank - 1, 0,
                             &vn[j][0], 1, MPI_FLOAT, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
            if (rank < size - 1) { // 右隣から受け取り、右隣へ送信
                MPI_Sendrecv(&un[j][local_nx_with_ghost - 1 - ghost_right], 1, MPI_FLOAT, rank + 1, 0,
                             &un[j][local_nx_with_ghost - 1], 1, MPI_FLOAT, rank + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Sendrecv(&vn[j][local_nx_with_ghost - 1 - ghost_right], 1, MPI_FLOAT, rank + 1, 0,
                             &vn[j][local_nx_with_ghost - 1], 1, MPI_FLOAT, rank + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        }
        MPI_Barrier(MPI_COMM_WORLD); // 通信が完了するまで待機

        // u, v の計算
        for (int j = 1; j < ny - 1; j++) {
            for (int i = compute_start_x_local; i <= compute_end_x_local; i++) {
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

        // u, v の境界条件 (y方向)
        for (int i = 0; i < local_nx_with_ghost; i++) {
            u[0][i] = 0;
            u[ny-1][i] = 1;
            v[0][i] = 0;
            v[ny-1][i] = 0;
        }

        // u, v の境界条件 (x方向) - グローバルな境界のみ適用
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
        
        MPI_Barrier(MPI_COMM_WORLD); // すべての計算が完了するまで待機

        if (n % 10 == 0) {
            for (int j = 0; j < ny; j++) {
                // 各プロセスの計算領域（ゴーストセルを除く）を収集
                MPI_Gatherv(&u[j][ghost_left], local_nx, MPI_FLOAT,
                            gather_u_row.data(), recvcounts.data(), displs.data(), MPI_FLOAT,
                            0, MPI_COMM_WORLD);
                
                MPI_Gatherv(&v[j][ghost_left], local_nx, MPI_FLOAT,
                            gather_v_row.data(), recvcounts.data(), displs.data(), MPI_FLOAT,
                            0, MPI_COMM_WORLD);

                MPI_Gatherv(&p[j][ghost_left], local_nx, MPI_FLOAT,
                            gather_p_row.data(), recvcounts.data(), displs.data(), MPI_FLOAT,
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
