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
    double dy = 2. / (ny - 1); // ここは -1ではなく、nx-1と同じように ny-1が正しいはずです。
    double dt = .01;
    double rho = 1.;
    double nu = .02;

    int local_nx_base = nx / size; // 各プロセス用の割り当てグリッド数
    int remainder = nx % size;     // プロセス数で割り切れない部分。（41%4=1）

    int begin_idx; // 各プロセスが担当するx方向の開始インデックス (ローカル計算領域の左端)
    int end_idx;   // 各プロセスが担当するx方向の終了インデックス (ローカル計算領域の右端+1)

    if (rank < remainder)
    { // 余り分を処理するプロセス。
        begin_idx = rank * (local_nx_base + 1);
        end_idx = begin_idx + (local_nx_base + 1);
    }
    else
    { // 余りを考慮しないプロセス。
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

    ofstream ufile("u.dat");
    ofstream vfile("v.dat");
    ofstream pfile("p.dat");

    // バッファの用意: 1列分のデータを送受信するためのテンポラリ配列
    // ny行分のデータを送受信するため、サイズは ny
    vector<float> send_left_buffer(ny);  // 左隣に送るデータ
    vector<float> recv_left_buffer(ny);  // 左隣から受け取るデータ
    vector<float> send_right_buffer(ny); // 右隣に送るデータ
    vector<float> recv_right_buffer(ny); // 右隣から受け取るデータ

    // バッファの用意: 1列分のデータを送受信するためのテンポラリ配列
    // ny行分のデータを送受信するため、サイズは ny
    vector<float> send_left_buffer(ny);  // 左隣に送るデータ
    vector<float> recv_left_buffer(ny);  // 左隣から受け取るデータ
    vector<float> send_right_buffer(ny); // 右隣に送るデータ
    vector<float> recv_right_buffer(ny); // 右隣から受け取るデータ

    // --- ここに、毎回ループ内で定義していた一時バッファを移動 ---
    // local_num_cols は各プロセスの担当列数で、ループ内で変化しないため、
    // ループ外で一度だけサイズを計算してバッファを定義できる。
    int local_num_cols = end_idx - begin_idx;
    vector<float> local_u_flat(local_num_cols * ny);
    vector<float> local_v_flat(local_num_cols * ny);
    vector<float> local_p_flat(local_num_cols * ny);

    // global_u_data などもランク0だけで必要なので、ランク0のスコープに移動しつつ、
    // ループ外で定義できるように変更
    vector<float> global_u_data;
    vector<float> global_v_data;
    vector<float> global_p_data;
    vector<int> recvcounts(size); // 各プロセスが送る列数 (ny行分のデータなので、ny * 列数)
    vector<int> displs(size);     // ランク0の受信バッファにおける開始位置

    if (rank == 0) {
        global_u_data.resize(nx * ny);
        global_v_data.resize(nx * ny);
        global_p_data.resize(nx * ny);

        // recvcounts と displs の計算は、ループ内でデータが変わらないので、
        // ランク0で一度だけ計算すればよい。
        for (int i = 0; i < size; ++i)
        {
            int current_local_nx;
            if (i < remainder)
            {
                current_local_nx = local_nx_base + 1;
            }
            else
            {
                current_local_nx = local_nx_base;
            }
            recvcounts[i] = current_local_nx * ny; // 各プロセスから受け取る要素数 (列数 * 行数)

            // オフセット計算
            if (i == 0) {
                displs[i] = 0;
            } else {
                displs[i] = displs[i-1] + recvcounts[i-1];
            }
        }
    }

    for (int n = 0; n < nt; n++)
    {
        // --- 速度場 (u, v) のゴーストセル交換 ---
        // b, u, v の計算の前に、隣接プロセスから u, v の境界データを取得する
        // MPI_Sendrecv を使用して、データの送受信を同時に行う
        // 各行(j)ごとに通信が必要なため、ny回の通信、またはMPI_Type_vectorを使用
        // ここではny回の通信を行う例を示します (より効率的な方法はMPI_Type_vector)

        // u のゴーストセル交換
        for (int j = 0; j < ny; j++)
        {
            // 右隣のプロセスに送るデータ (担当領域の右端の1列分)
            send_right_buffer[j] = u[j][end_idx - 1]; // 自分の右端の列
            // 左隣のプロセスに送るデータ (担当領域の左端の1列分)
            send_left_buffer[j] = u[j][begin_idx]; // 自分の左端の列
        }

        // ゴーストセル通信（U成分）
        if (rank > 0)
        { // 左隣のプロセスがいる場合
            // 左隣からデータを受け取り (rank-1 から rank の begin_idx-1 へ)、
            // 自分の左端のデータを左隣に送る (rank の begin_idx から rank-1 へ)
            MPI_Sendrecv(&send_left_buffer[0], ny, MPI_FLOAT, rank - 1, 0,
                         &recv_left_buffer[0], ny, MPI_FLOAT, rank - 1, 1,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            // 受け取ったデータをゴーストセルに格納
            for (int j = 0; j < ny; ++j)
            {
                u[j][begin_idx - 1] = recv_left_buffer[j]; // 左側のゴーストセルを更新
            }
        }
        if (rank < size - 1)
        { // 右隣のプロセスがいる場合
            // 右隣にデータを送り (rank の end_idx-1 から rank+1 へ)、
            // 右隣からデータを受け取る (rank+1 から rank の end_idx へ)
            MPI_Sendrecv(&send_right_buffer[0], ny, MPI_FLOAT, rank + 1, 1,
                         &recv_right_buffer[0], ny, MPI_FLOAT, rank + 1, 0,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            // 受け取ったデータをゴーストセルに格納
            for (int j = 0; j < ny; ++j)
            {
                u[j][end_idx] = recv_right_buffer[j]; // 右側のゴーストセルを更新
            }
        }

        // v のゴーストセル交換 (u と同様のロジック)
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
                // Compute b[j][i]
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
            // p の計算の前に、隣接プロセスから p の境界データを取得する
            // pn にコピーした p の値を使うため、pnのゴーストセルを更新する
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
                    // Compute p[j][i]
                    p[j][i] = (dy * dy * (pn[j][i + 1] + pn[j][i - 1]) +
                               dx * dx * (pn[j + 1][i] + pn[j - 1][i]) -
                               b[j][i] * dx * dx * dy * dy) /
                              (2. * (dx * dx + dy * dy));
                }
            }
            if (rank == 0)
            { // ランク0が左端の境界条件を担当
                for (int j = 0; j < ny; j++)
                {
                    p[j][0] = p[j][1];
                }
            }
            if (rank == size - 1)
            { // 最後のランクが右端の境界条件を担当
                for (int j = 0; j < ny; j++)
                {
                    p[j][nx - 1] = p[j][nx - 2];
                }
            }
            for (int i = begin_idx; i < end_idx; i++)
            { // y方向の境界条件は共有メモリ上でそのまま
                p[0][i] = p[1][i];
                p[ny - 1][i] = 0;
            }
        } // end of nit loop

        // un, vn へのコピーを nit ループの外に移動
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
            // 右隣のプロセスに送るデータ (担当領域の右端の1列分)
            send_right_buffer[j] = un[j][end_idx - 1]; // 自分の右端の列
            // 左隣のプロセスに送るデータ (担当領域の左端の1列分)
            send_left_buffer[j] = un[j][begin_idx]; // 自分の左端の列
        }

        // ゴーストセル通信（U成分）
        if (rank > 0)
        { // 左隣のプロセスがいる場合
            // 左隣からデータを受け取り (rank-1 から rank の begin_idx-1 へ)、
            // 自分の左端のデータを左隣に送る (rank の begin_idx から rank-1 へ)
            MPI_Sendrecv(&send_left_buffer[0], ny, MPI_FLOAT, rank - 1, 0,
                         &recv_left_buffer[0], ny, MPI_FLOAT, rank - 1, 1,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            // 受け取ったデータをゴーストセルに格納
            for (int j = 0; j < ny; ++j)
            {
                un[j][begin_idx - 1] = recv_left_buffer[j]; // 左側のゴーストセルを更新
            }
        }
        if (rank < size - 1)
        { // 右隣のプロセスがいる場合
            // 右隣にデータを送り (rank の end_idx-1 から rank+1 へ)、
            // 右隣からデータを受け取る (rank+1 から rank の end_idx へ)
            MPI_Sendrecv(&send_right_buffer[0], ny, MPI_FLOAT, rank + 1, 1,
                         &recv_right_buffer[0], ny, MPI_FLOAT, rank + 1, 0,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            // 受け取ったデータをゴーストセルに格納
            for (int j = 0; j < ny; ++j)
            {
                un[j][end_idx] = recv_right_buffer[j]; // 右側のゴーストセルを更新
            }
        }

        // vn のゴーストセル交換 (un と同様のロジック)
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
                // Compute u[j][i] and v[j][i]
                u[j][i] = un[j][i] - un[j][i] * dt / dx * (un[j][i] - un[j][i - 1]) - vn[j][i] * dt / dy * (un[j][i] - un[j - 1][i]) - dt / (2. * rho * dx) * (p[j][i + 1] - p[j][i - 1]) + nu * dt / (dx * dx) * (un[j][i + 1] - 2. * un[j][i] + un[j][i - 1]) + nu * dt / (dy * dy) * (un[j + 1][i] - 2. * un[j][i] + un[j - 1][i]);

                v[j][i] = vn[j][i] - un[j][i] * dt / dx * (vn[j][i] - vn[j][i - 1]) - vn[j][i] * dt / dy * (vn[j][i] - vn[j - 1][i]) - dt / (2. * rho * dy) * (p[j + 1][i] - p[j - 1][i]) // ここはdyに修正
                          + nu * dt / (dx * dx) * (vn[j][i + 1] - 2. * vn[j][i] + vn[j][i - 1]) + nu * dt / (dy * dy) * (vn[j + 1][i] - 2. * vn[j][i] + vn[j - 1][i]);
            }
        }
        if (rank == 0)
        { // ランク0が左端の境界条件を担当
            for (int j = 0; j < ny; j++)
            {
                u[j][0] = 0;
                v[j][0] = 0;
            }
        }
        if (rank == size - 1)
        { // 最後のランクが右端の境界条件を担当
            for (int j = 0; j < ny; j++)
            {
                u[j][nx - 1] = 0;
                v[j][nx - 1] = 0;
            }
        }
        for (int i = begin_idx; i < end_idx; i++)
        { // y方向の境界条件は共有メモリ上でそのまま
            u[0][i] = 0;
            u[ny - 1][i] = 1;
            v[0][i] = 0;
            v[ny - 1][i] = 0;
        }

        if (n % 10 == 0)
        {
            // ローカルのデータを行ごとに連続したバッファにコピー
            // バッファはすでにループ外で定義済みなので、再利用する。
            for (int j = 0; j < ny; ++j) {
                for (int i = 0; i < local_num_cols; ++i) {
                    local_u_flat[j * local_num_cols + i] = u[j][begin_idx + i];
                    local_v_flat[j * local_num_cols + i] = v[j][begin_idx + i];
                    local_p_flat[j * local_num_cols + i] = p[j][begin_idx + i];
                }
            }

            // MPI_Gatherv でデータをランク0に集約
            MPI_Gatherv(&local_u_flat[0], local_num_cols * ny, MPI_FLOAT,
                        (rank == 0 ? &global_u_data[0] : nullptr), &recvcounts[0], &displs[0], MPI_FLOAT,
                        0, MPI_COMM_WORLD);
            MPI_Gatherv(&local_v_flat[0], local_num_cols * ny, MPI_FLOAT,
                        (rank == 0 ? &global_v_data[0] : nullptr), &recvcounts[0], &displs[0], MPI_FLOAT,
                        0, MPI_COMM_WORLD);
            MPI_Gatherv(&local_p_flat[0], local_num_cols * ny, MPI_FLOAT,
                        (rank == 0 ? &global_p_data[0] : nullptr), &recvcounts[0], &displs[0], MPI_FLOAT,
                        0, MPI_COMM_WORLD);

            if (rank == 0)
            {
                // 集約されたデータをファイルに書き込む
                for (int j = 0; j < ny; j++)
                {
                    for (int i = 0; i < nx; i++)
                        ufile << global_u_data[j * nx + i] << " ";
                    ufile << "\n";
                }
                for (int j = 0; j < ny; j++)
                {
                    for (int i = 0; i < nx; i++)
                        vfile << global_v_data[j * nx + i] << " ";
                    vfile << "\n";
                }
                for (int j = 0; j < ny; j++)
                {
                    for (int i = 0; i < nx; i++)
                        pfile << global_p_data[j * nx + i] << " ";
                    pfile << "\n";
                }
            }
        }
    } // end of nt loop

    ufile.close();
    vfile.close();
    pfile.close();

    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(stop - start);
    if (rank == 0)
    { // ランク0のみが出力
        printf("Elapsed time: %lld ms\n", duration.count());
    }

    MPI_Finalize();
    return 0; // main関数はintを返すのでreturn 0を追加
}
