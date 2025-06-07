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
    int current_local_nx;

    if (rank < remainder)
    {
        begin_idx = rank * (local_nx_base + 1);
        end_idx = begin_idx + (local_nx_base + 1);
        current_local_nx = local_nx_base + 1;
    }
    else
    {
        begin_idx = rank * local_nx_base + remainder;
        end_idx = begin_idx + local_nx_base;
        current_local_nx = local_nx_base;
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

    vector<float> send_left_buffer(ny);
    vector<float> recv_left_buffer(ny);
    vector<float> send_right_buffer(ny);
    vector<float> recv_right_buffer(ny);
    
    // 全てのデータを集めるためのグローバルな配列 (rank 0 のみで確保)
    // この vector<vector<float>> はメモリが連続しているとは限らないが、
    // MPI_Gathervのrecvbufには global_u_data[0].data() のように
    // 1次元配列のように連続した領域のポインタを渡す想定。
    // この場合、ny * nx 個の float 型要素が連続している必要がある。
    // そのためには、global_u_data を vector<float>(ny * nx) として宣言し、
    // 2次元アクセスを `[j * nx + i]` のように変換する必要がある。
    // あるいは、`global_u_data[0].data()` が指すメモリ領域が
    // ny * nx の要素を格納するのに十分な連続性を持っていることを前提とする。
    // ここでは、現在のmatrix定義を維持するため、その前提で進める。
    // ただし、より堅牢な実装には1次元vectorへの変更を推奨。
    matrix global_u_data(ny, vector<float>(nx));
    matrix global_v_data(ny, vector<float>(nx));
    matrix global_p_data(ny, vector<nx)); // global_p_data(ny, vector<float>(nx))のtypo修正

    // 各プロセスのデータ量とオフセットを計算 (全プロセスで実行)
    vector<int> recvcounts(size);
    vector<int> displs(size);
    int current_global_x_offset = 0; 
    for (int r = 0; r < size; ++r) { 
        int r_local_nx; 
        if (r < remainder) {
            r_local_nx = local_nx_base + 1; 
        } else {
            r_local_nx = local_nx_base;     
        }
        
        recvcounts[r] = r_local_nx * ny; 
        displs[r] = current_global_x_offset * ny; 
        current_global_x_offset += r_local_nx;
    }

    for (int n = 0; n < nt; n++)
    {
        // --- 速度場 (u, v) のゴーストセル交換 ---
        for (int j = 0; j < ny; j++)
        {
            send_right_buffer[j] = u[j][end_idx - 1];
            send_left_buffer[j] = u[j][begin_idx];
        }
        if (rank > 0)
        {
            MPI_Sendrecv(&send_left_buffer[0], ny, MPI_FLOAT, rank - 1, 0,
                         &recv_left_buffer[0], ny, MPI_FLOAT, rank - 1, 1,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            for (int j = 0; j < ny; ++j)
            {
                u[j][begin_idx - 1] = recv_left_buffer[j];
            }
        }
        if (rank < size - 1)
        {
            MPI_Sendrecv(&send_right_buffer[0], ny, MPI_FLOAT, rank + 1, 1,
                         &recv_right_buffer[0], ny, MPI_FLOAT, rank + 1, 0,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            for (int j = 0; j < ny; ++j)
            {
                u[j][end_idx] = recv_right_buffer[j];
            }
        }

        for (int j = 0; j < ny; j++)
        {
            send_right_buffer[j] = v[j][end_idx - 1];
            send_left_buffer[j] = v[j][begin_idx];
        }
        if (rank > 0)
        {
            MPI_Sendrecv(&send_left_buffer[0], ny, MPI_FLOAT, rank - 1, 2,
                         &recv_left_buffer[0], ny, MPI_FLOAT, rank - 1, 3,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            for (int j = 0; j < ny; ++j)
            {
                v[j][begin_idx - 1] = recv_left_buffer[j];
            }
        }
        if (rank < size - 1)
        {
            MPI_Sendrecv(&send_right_buffer[0], ny, MPI_FLOAT, rank + 1, 3,
                         &recv_right_buffer[0], ny, MPI_FLOAT, rank + 1, 2,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            for (int j = 0; j < ny; ++j)
            {
                v[j][end_idx] = recv_right_buffer[j];
            }
        }
        // --- 速度場 (u, v) のゴーストセル交換 終わり ---

        for (int j = 1; j < ny - 1; j++)
        {
            for (int i = max(1, begin_idx); i < min(nx - 1, end_idx); i++)
            {
                b[j][i] = rho * (1. / dt * ((u[j][i + 1] - u[j][i - 1]) / (2. * dx) + (v[j + 1][i] - v[j - 1][i]) / (2. * dy)) -
                                 pow((u[j][i + 1] - u[j][i - 1]) / (2. * dx), 2) -
                                 2. * ((u[j + 1][i] - u[j - 1][i]) / (2. * dy) * (v[j][i + 1] - v[j][i - 1]) / (2. * dx)) -
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

            // --- 圧力場 (p/pn) のゴーストセル交換 ---
            for (int j = 0; j < ny; j++)
            {
                send_right_buffer[j] = pn[j][end_idx - 1];
                send_left_buffer[j] = pn[j][begin_idx];
            }
            if (rank > 0)
            {
                MPI_Sendrecv(&send_left_buffer[0], ny, MPI_FLOAT, rank - 1, 4,
                             &recv_left_buffer[0], ny, MPI_FLOAT, rank - 1, 5,
                             MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                for (int j = 0; j < ny; ++j)
                {
                    pn[j][begin_idx - 1] = recv_left_buffer[j];
                }
            }
            if (rank < size - 1)
            {
                MPI_Sendrecv(&send_right_buffer[0], ny, MPI_FLOAT, rank + 1, 5,
                             &recv_right_buffer[0], ny, MPI_FLOAT, rank + 1, 4,
                             MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                for (int j = 0; j < ny; ++j)
                {
                    pn[j][end_idx] = recv_right_buffer[j];
                }
            }
            // --- 圧力場 (p/pn) のゴーストセル交換 終わり ---

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

        // un のゴーストセル交換
        for (int j = 0; j < ny; j++)
        {
            send_right_buffer[j] = un[j][end_idx - 1];
            send_left_buffer[j] = un[j][begin_idx];
        }
        if (rank > 0)
        {
            MPI_Sendrecv(&send_left_buffer[0], ny, MPI_FLOAT, rank - 1, 0,
                         &recv_left_buffer[0], ny, MPI_FLOAT, rank - 1, 1,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            for (int j = 0; j < ny; ++j)
            {
                un[j][begin_idx - 1] = recv_left_buffer[j];
            }
        }
        if (rank < size - 1)
        {
            MPI_Sendrecv(&send_right_buffer[0], ny, MPI_FLOAT, rank + 1, 1,
                         &recv_right_buffer[0], ny, MPI_FLOAT, rank + 1, 0,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            for (int j = 0; j < ny; ++j)
            {
                un[j][end_idx] = recv_right_buffer[j];
            }
        }

        // vn のゴーストセル交換
        for (int j = 0; j < ny; j++)
        {
            send_right_buffer[j] = vn[j][end_idx - 1];
            send_left_buffer[j] = vn[j][begin_idx];
        }
        if (rank > 0)
        {
            MPI_Sendrecv(&send_left_buffer[0], ny, MPI_FLOAT, rank - 1, 2,
                         &recv_left_buffer[0], ny, MPI_FLOAT, rank - 1, 3,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            for (int j = 0; j < ny; ++j)
            {
                vn[j][begin_idx - 1] = recv_left_buffer[j];
            }
        }
        if (rank < size - 1)
        {
            MPI_Sendrecv(&send_right_buffer[0], ny, MPI_FLOAT, rank + 1, 3,
                         &recv_right_buffer[0], ny, MPI_FLOAT, rank + 1, 2,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            for (int j = 0; j < ny; ++j)
            {
                vn[j][end_idx] = recv_right_buffer[j];
            }
        }
        // --- 速度場 (un, vn) のゴーストセル交換 終わり ---

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
            // 各プロセスのローカルデータを1次元配列にコピー
            vector<float> local_u_flat(ny * current_local_nx);
            vector<float> local_v_flat(ny * current_local_nx);
            vector<float> local_p_flat(ny * current_local_nx);

            int buf_idx = 0;
            for (int j = 0; j < ny; ++j) {
                for (int i = begin_idx; i < end_idx; ++i) {
                    local_u_flat[buf_idx] = u[j][i];
                    local_v_flat[buf_idx] = v[j][i];
                    local_p_flat[buf_idx] = p[j][i];
                    buf_idx++;
                }
            }

            // ランク0に全てのデータを集める
            MPI_Gatherv(local_u_flat.data(), ny * current_local_nx, MPI_FLOAT,
                        global_u_data[0].data(), recvcounts.data(), displs.data(), MPI_FLOAT,
                        0, MPI_COMM_WORLD);
            MPI_Gatherv(local_v_flat.data(), ny * current_local_nx, MPI_FLOAT,
                        global_v_data[0].data(), recvcounts.data(), displs.data(), MPI_FLOAT,
                        0, MPI_COMM_WORLD);
            MPI_Gatherv(local_p_flat.data(), ny * current_local_nx, MPI_FLOAT,
                        global_p_data[0].data(), recvcounts.data(), displs.data(), MPI_FLOAT,
                        0, MPI_COMM_WORLD);

            if (rank == 0)
            {
                // ランク0のみがファイルを開いて書き込む
                ofstream ufile_out("u.dat", ios_base::app); 
                ofstream vfile_out("v.dat", ios_base::app);
                ofstream pfile_out("p.dat", ios_base::app);

                for (int j = 0; j < ny; j++)
                {
                    for (int i = 0; i < nx; i++)
                    {
                        ufile_out << global_u_data[j][i] << " ";
                    }
                }
                ufile_out << "\n"; 

                for (int j = 0; j < ny; j++)
                {
                    for (int i = 0; i < nx; i++)
                    {
                        vfile_out << global_v_data[j][i] << " ";
                    }
                }
                vfile_out << "\n";

                for (int j = 0; j < ny; j++)
                {
                    for (int i = 0; i < nx; i++)
                    {
                        pfile_out << global_p_data[j][i] << " ";
                    }
                }
                pfile_out << "\n";

                ufile_out.close();
                vfile_out.close();
                pfile_out.close();
            }
        }
    } // end of nt loop

    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(stop - start);
    if (rank == 0)
    {
        printf("Elapsed time: %lld ms\n", duration.count());
    }

    MPI_Finalize();
    return 0;
}
