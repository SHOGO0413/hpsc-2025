#include <cstdlib>
#include <cstdio>
#include <fstream>
#include <vector>
#include <cmath>
#include <chrono>
#include <mpi.h>

using namespace std::chrono;
using namespace std;

// 行列の型定義
typedef vector<vector<float>> matrix;

// ゴーストセル交換関数
// field: ゴーストセルを交換したい行列
// ny: 行列の行数
// my_cols: このプロセスが担当する実データの列数 (ゴーストセルを除く)
// rank: 現在のプロセスのランク
// size: 総プロセス数
// tags: MPI通信で使用するユニークなタグのペア (左送信, 左受信, 右送信, 右受信)
// 注意: fieldは左右に1列ずつのゴーストセルを持つように宣言されていることを前提とする
void exchange_ghost_cells(matrix& field, int ny, int my_cols, int rank, int size,
                          int tag_send_left, int tag_recv_left, int tag_send_right, int tag_recv_right)
{
    // バッファの用意: 1列分のデータを送受信するためのテンポラリ配列
    vector<float> send_left_buffer(ny);
    vector<float> recv_left_buffer(ny);
    vector<float> send_right_buffer(ny);
    vector<float> recv_right_buffer(ny);

    // 左隣のプロセスがいる場合 (rank > 0)
    if (rank > 0)
    {
        // 左隣に送るデータ (自分の左端の実データ列: ローカルインデックス1)
        for (int j = 0; j < ny; j++) send_left_buffer[j] = field[j][1]; 
        
        // 左隣からデータを受け取り (rank-1 から rank の左ゴーストセルへ)、
        // 自分の左端のデータを左隣に送る (rank の左端実データから rank-1 へ)
        MPI_Sendrecv(send_left_buffer.data(), ny, MPI_FLOAT, rank - 1, tag_send_left,
                     recv_left_buffer.data(), ny, MPI_FLOAT, rank - 1, tag_recv_left,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        
        // 受け取ったデータを左側ゴーストセル (ローカルインデックス0) に格納
        for (int j = 0; j < ny; ++j) field[j][0] = recv_left_buffer[j];
    }

    // 右隣のプロセスがいる場合 (rank < size - 1)
    if (rank < size - 1)
    {
        // 右隣に送るデータ (自分の右端の実データ列: ローカルインデックス my_cols)
        for (int j = 0; j < ny; j++) send_right_buffer[j] = field[j][my_cols];
        
        // 右隣にデータを送り (rank の右端実データから rank+1 へ)、
        // 右隣からデータを受け取る (rank+1 から rank の右ゴーストセルへ)
        MPI_Sendrecv(send_right_buffer.data(), ny, MPI_FLOAT, rank + 1, tag_send_right,
                     recv_right_buffer.data(), ny, MPI_FLOAT, rank + 1, tag_recv_right,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        
        // 受け取ったデータを右側ゴーストセル (ローカルインデックス my_cols+1) に格納
        for (int j = 0; j < ny; ++j) field[j][my_cols + 1] = recv_right_buffer[j];
    }
}


int main(int argc, char *argv[])
{
    // MPIの初期化
    MPI_Init(&argc, &argv);

    int size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &size); // 総プロセス数の取得
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); // 現在のプロセスのランク（ID）の取得

    // シミュレーションパラメータ
    int nx = 41; // x方向のグリッド点数
    int ny = 41; // y方向のグリッド点数
    int nt = 500; // 時間ステップの総数
    int nit = 50; // 圧力ポアソン方程式の反復回数
    double dx = 2. / (nx - 1); // x方向のグリッド間隔
    double dy = 2. / (ny - 1); // y方向のグリッド間隔
    double dt = .01; // 時間ステップの大きさ
    double rho = 1.; // 流体の密度
    double nu = .02; // 流体の動粘性係数

    // 各プロセスの担当するx方向のグローバルインデックス範囲を計算
    int local_nx_base = nx / size;     // 各プロセスに割り当てる基本的なグリッド数
    int remainder = nx % size;         // プロセス数で割り切れない部分

    int global_begin_idx; // 各プロセスが担当するx方向の開始グローバルインデックス (実データ)
    int global_end_idx;   // 各プロセスが担当するx方向の終了グローバルインデックス (実データ, 排他的)

    if (rank < remainder)
    { // 余り分を処理するプロセス（多めに担当）
        global_begin_idx = rank * (local_nx_base + 1);
        global_end_idx = global_begin_idx + (local_nx_base + 1);
    }
    else
    { // 余りを考慮しないプロセス（基本数を担当）
        global_begin_idx = rank * local_nx_base + remainder;
        global_end_idx = global_begin_idx + local_nx_base;
    }

    // 各プロセスが担当する実データの列数（ゴーストセルを含まない）
    int my_cols = global_end_idx - global_begin_idx;

    // --- ローカル配列の宣言 (ゴーストセル領域を含む) ---
    // u, v, p は左右に1列ずつのゴーストセルを持つため、my_cols + 2 の列数で宣言
    matrix u(ny, vector<float>(my_cols + 2)); 
    matrix v(ny, vector<float>(my_cols + 2));
    matrix p(ny, vector<float>(my_cols + 2));

    // b, un, vn, pn はゴーストセルを必要としないため、my_cols の列数で宣言
    matrix b(ny, vector<float>(my_cols));
    matrix un(ny, vector<float>(my_cols)); // 前の時間ステップの u (ローカル配列)
    matrix vn(ny, vector<float>(my_cols)); // 前の時間ステップの v (ローカル配列)
    matrix pn(ny, vector<float>(my_cols)); // 前の反復ステップの p (ローカル配列)


    // --- 初期化ループ ---
    // 各プロセスは自分の担当範囲 (ローカルインデックス 1 から my_cols) のみを初期化
    for (int j = 0; j < ny; j++)
    {
        for (int i_local = 1; i_local <= my_cols; i_local++) // ローカルインデックス1からmy_colsまで
        {
            u[j][i_local] = 0;
            v[j][i_local] = 0;
            p[j][i_local] = 0;
            // b, un, vn, pn は初期化不要（コピー元として使用するため）
            // b[j][i_local-1] = 0; // bはローカルインデックス0から
            // un[j][i_local-1] = 0;
            // vn[j][i_local-1] = 0;
            // pn[j][i_local-1] = 0;
        }
    }
    
    // --- 初期状態の同期 (最初のゴーストセル交換で0を送り合わないようにするため) ---
    // 各プロセスが担当範囲のデータを初期化した後、MPI_Allgathervを使って全プロセスで配列を同期する
    // これにより、時間ステップ1のゴーストセル交換の時点で、すべての要素が正しい初期値（0）を持つ
    vector<int> allgather_recvcounts(size); // 各プロセスから受け取るデータの総要素数 (列数 * ny)
    vector<int> allgather_displs(size);     // 各プロセスからのデータを受信バッファのどこに配置するか (オフセット)

    for (int i = 0; i < size; ++i)
    {
        int current_rank_cols;
        if (i < remainder)
        {
            current_rank_cols = local_nx_base + 1;
        }
        else
        {
            current_rank_cols = local_nx_base;
        }
        allgather_recvcounts[i] = current_rank_cols * ny;
    }

    allgather_displs[0] = 0;
    for (int i = 1; i < size; ++i)
    {
        allgather_displs[i] = allgather_displs[i - 1] + allgather_recvcounts[i - 1];
    }
    
    // MPI_Allgatherv用の送信バッファ (ローカル配列の実データ部分)
    vector<float> send_flat_u(my_cols * ny);
    vector<float> send_flat_v(my_cols * ny);
    vector<float> send_flat_p(my_cols * ny);

    for (int j = 0; j < ny; ++j) {
        for (int i_local = 0; i_local < my_cols; ++i_local) { // ローカルインデックス0からmy_cols-1
            send_flat_u[j * my_cols + i_local] = u[j][i_local + 1]; // uはインデックス1から実データ
            send_flat_v[j * my_cols + i_local] = v[j][i_local + 1];
            send_flat_p[j * my_cols + i_local] = p[j][i_local + 1];
        }
    }

    // MPI_Allgathervで全てのプロセスのu, v, p配列を同期する
    // 受信側はu[0].data()ではなく、グローバル配列の対応する位置へのポインタが必要
    // ただし、ここではローカルのu,v,pをグローバルに見えるように更新するため、allgather_recvcounts/displsは
    // 各プロセスがグローバルな位置にデータを書き込むための情報として利用
    // これを行うには、各フィールドを一度グローバルな位置に集約してからAllgathervで全体に配り直す
    // という複雑なロジックが必要になる。
    // Simplest: 全てのプロセスが global_u/v/p を受け取る形式に合わせる (現在のコードのmatrix u(ny,vector<float>(nx))はこれ)
    // そして、そのglobal_u/v/pに、自身のローカルな値をコピーする。

    // 以下のMPI_Allgathervは、あくまで各プロセスのグローバル配列を同期する目的で、
    // send_flat_u のデータが u[0].data() (グローバル配列の先頭) のどこに配置されるべきかを指定する。
    // しかし、u[0].data() はnx列全体なので、displs_allgather は、各プロセスのbegin_idx * ny となる。
    vector<int> current_displs(size); // 各プロセスが自分の担当範囲の開始グローバルインデックスを要素数に変換
    for(int i=0; i<size; ++i) {
        int current_global_begin;
        if (i < remainder) {
            current_global_begin = i * (local_nx_base + 1);
        } else {
            current_global_begin = i * local_nx_base + remainder;
        }
        current_displs[i] = current_global_begin * ny;
    }

    MPI_Allgatherv(send_flat_u.data(), allgather_recvcounts[rank], MPI_FLOAT, // 送信するローカルデータ
                   u[0].data(), allgather_recvcounts.data(), current_displs.data(), MPI_FLOAT, // 受信するグローバルデータ
                   MPI_COMM_WORLD);
    MPI_Allgatherv(send_flat_v.data(), allgather_recvcounts[rank], MPI_FLOAT,
                   v[0].data(), allgather_recvcounts.data(), current_displs.data(), MPI_FLOAT,
                   MPI_COMM_WORLD);
    MPI_Allgatherv(send_flat_p.data(), allgather_recvcounts[rank], MPI_FLOAT,
                   p[0].data(), allgather_recvcounts.data(), current_displs.data(), MPI_FLOAT,
                   MPI_COMM_WORLD);
    // --- 初期状態の同期 終わり ---


    // 時間計測開始
    auto start = high_resolution_clock::now();

    // 出力ファイルストリーム (ランク0のみが使用する想定だが、ここではすべてのプロセスで開いておく)
    ofstream ufile("u.dat");
    ofstream vfile("v.dat");
    ofstream pfile("p.dat");

    // MPI_Gatherv に必要な変数 (ファイル出力用)
    // これらは既に main 関数冒頭で計算されている allgather_recvcounts/displs を再利用できる
    vector<int>& gatherv_recvcounts_output = allgather_recvcounts; // 同じ計算なので参照で良い
    vector<int>& gatherv_displs_output = current_displs;           // 同じ計算なので参照で良い

    // ランク0で全体の計算結果を受け取るためのバッファ
    // 1次元配列として確保し、[j * nx + i] でアクセスする
    vector<float> global_u(nx * ny);
    vector<float> global_v(nx * ny);
    vector<float> global_p(nx * ny);


    // --- メインの時間ステップループ ---
    for (int n = 0; n < nt; n++)
    {
        // --- 速度場 (u, v) のゴーストセル交換 ---
        // b, u, v の計算の前に、隣接プロセスから u, v の境界データを取得する
        // exchange_ghost_cells関数を呼び出す
        exchange_ghost_cells(u, ny, my_cols, rank, size, 0, 1, 1, 0); // uの交換 (タグ 0,1)
        exchange_ghost_cells(v, ny, my_cols, rank, size, 2, 3, 3, 2); // vの交換 (タグ 2,3)
        // --- 速度場 (u, v) のゴーストセル交換 終わり ---


        for (int j = 1; j < ny - 1; j++)
        {
            for (int i_local = 1; i_local <= my_cols; i_local++) // ローカルインデックス1からmy_colsまで
            {
                // Compute b[j][i_local-1] (bはローカルインデックス0から)
                // 各項はローカルインデックスに変換: i_local -> i_local+1, i_local-1
                b[j][i_local-1] = rho * (1. / dt * ((u[j][i_local + 1] - u[j][i_local - 1]) / (2. * dx) + (v[j + 1][i_local] - v[j - 1][i_local]) / (2. * dy)) -
                                 pow(((u[j][i_local + 1] - u[j][i_local - 1]) / (2. * dx)), 2) -
                                 2. * (((u[j + 1][i_local] - u[j - 1][i_local]) / (2. * dy)) * ((v[j][i_local + 1] - v[j][i_local - 1]) / (2. * dx))) -
                                 pow(((v[j + 1][i_local] - v[j - 1][i_local]) / (2. * dy)), 2));
            }
        }
        for (int it = 0; it < nit; it++)
        {
            // --- 圧力場 (p/pn) のゴーストセル交換 ---
            // p の計算の前に、隣接プロセスから p の境界データを取得する
            // (pnはpの古い値なので、pの現在の値でゴーストセルを更新してからpnにコピーする)
            exchange_ghost_cells(p, ny, my_cols, rank, size, 4, 5, 5, 4); // pの交換 (タグ 4,5)
            // --- 圧力場 (p/pn) のゴーストセル交換 終わり ---

            for (int j = 0; j < ny; j++)
            {
                for (int i_local = 0; i_local < my_cols; i_local++) // pnはローカルインデックス0から
                {
                    pn[j][i_local] = p[j][i_local + 1]; // pはローカルインデックス1から実データ
                }
            }
            for (int j = 1; j < ny - 1; j++)
            {
                for (int i_local = 1; i_local <= my_cols; i_local++) // ローカルインデックス1からmy_colsまで
                {
                    // Compute p[j][i_local]
                    p[j][i_local] = (dy * dy * (pn[j][i_local + 1] + pn[j][i_local - 1]) +
                               dx * dx * (pn[j + 1][i_local] + pn[j - 1][i_local]) -
                               b[j][i_local - 1] * dx * dx * dy * dy) / // bはローカルインデックス0から
                              (2. * (dx * dx + dy * dy));
                }
            }
            // --- 圧力の境界条件 (グローバル境界にあるプロセスのみ適用、ローカルインデックスに変換) ---
            // 左端 (i=0) の境界条件はランク0のみが担当
            if (rank == 0)
            { 
                for (int j = 0; j < ny; j++)
                {
                    p[j][1] = p[j][2]; // ローカルインデックス: p[j][0]は左ゴーストセル、p[j][1]がi=0, p[j][2]がi=1
                }
            }
            // 右端 (i=nx-1) の境界条件は最後のランクのみが担当
            if (rank == size - 1)
            { 
                for (int j = 0; j < ny; j++)
                {
                    p[j][my_cols] = p[j][my_cols - 1]; // ローカルインデックス: my_colsがi=nx-1
                }
            }
            // y方向の境界条件は全プロセスが担当（ローカルインデックスに変換）
            for (int i_local = 1; i_local <= my_cols; i_local++) 
            {
                p[0][i_local] = p[1][i_local];
                p[ny - 1][i_local] = 0;
            }
        } // end of nit loop

        // un, vn へのコピー
        for (int j = 0; j < ny; j++)
        {
            for (int i_local = 0; i_local < my_cols; i_local++) // ローカルインデックス0からmy_cols-1
            {
                un[j][i_local] = u[j][i_local + 1]; // uはローカルインデックス1から実データ
                vn[j][i_local] = v[j][i_local + 1];
            }
        }

        for (int j = 1; j < ny - 1; j++)
        {
            for (int i_local = 1; i_local <= my_cols; i_local++) // ローカルインデックス1からmy_colsまで
            {
                // Compute u[j][i_local] and v[j][i_local]
                // 各項はローカルインデックスに変換: i_local -> i_local+1, i_local-1
                u[j][i_local] = un[j][i_local-1] - un[j][i_local-1] * dt / dx * (un[j][i_local-1] - un[j][i_local-2]) - vn[j][i_local-1] * dt / dy * (un[j][i_local-1] - un[j-1][i_local-1]) - dt / (2. * rho * dx) * (p[j][i_local+1] - p[j][i_local-1]) + nu * dt / (dx*dx) * (un[j][i_local] - 2. * un[j][i_local-1] + un[j][i_local-2]) + nu * dt / (dy*dy) * (un[j+1][i_local-1] - 2. * un[j][i_local-1] + un[j-1][i_local-1]);

                v[j][i_local] = vn[j][i_local-1] - un[j][i_local-1] * dt / dx * (vn[j][i_local-1] - vn[j][i_local-2]) - vn[j][i_local-1] * dt / dy * (vn[j][i_local-1] - vn[j-1][i_local-1]) - dt / (2. * rho * dy) * (p[j+1][i_local] - p[j-1][i_local]) // ここはdyに修正
                                 + nu * dt / (dx*dx) * (vn[j][i_local] - 2. * vn[j][i_local-1] + vn[j][i_local-2]) + nu * dt / (dy*dy) * (vn[j+1][i_local-1] - 2. * vn[j][i_local-1] + vn[j-1][i_local-1]);
            }
        }
        // --- 速度の境界条件 (グローバル境界にあるプロセスのみ適用、ローカルインデックスに変換) ---
        // 左端 (i=0) の境界条件はランク0のみが担当
        if (rank == 0)
        { 
            for (int j = 0; j < ny; j++)
            {
                u[j][1] = 0; // ローカルインデックス1がi=0に対応
                v[j][1] = 0;
            }
        }
        // 右端 (i=nx-1) の境界条件は最後のランクのみが担当
        if (rank == size - 1)
        { 
            for (int j = 0; j < ny; j++)
            {
                u[j][my_cols] = 0; // ローカルインデックスmy_colsがi=nx-1に対応
                v[j][my_cols] = 0;
            }
        }
        // y方向の境界条件は全プロセスが担当（ローカルインデックスに変換）
        for (int i_local = 1; i_local <= my_cols; i_local++)
        { 
            u[0][i_local] = 0;
            u[ny - 1][i_local] = 1; // 上端の壁が右に動く条件
            v[0][i_local] = 0;
            v[ny - 1][i_local] = 0;
        }

        if (n % 10 == 0)
        {
            // ---  MPI_Gatherv を使用して、各プロセスからランク0にデータを集める ---
            // 各プロセスの担当範囲 (ローカルインデックス1からmy_cols) のデータを1次元配列として集める
            // vector<vector<float>> から vector<float> への変換が必要
            vector<float> send_u_flat_output(my_cols * ny);
            vector<float> send_v_flat_output(my_cols * ny);
            vector<float> send_p_flat_output(my_cols * ny);

            for(int j=0; j<ny; ++j) {
                for(int i_local=0; i_local<my_cols; ++i_local) { // ローカルインデックス0からmy_cols-1
                    send_u_flat_output[j * my_cols + i_local] = u[j][i_local + 1]; // uはローカルインデックス1から実データ
                    send_v_flat_output[j * my_cols + i_local] = v[j][i_local + 1];
                    send_p_flat_output[j * my_cols + i_local] = p[j][i_local + 1];
                }
            }

            // MPI_Gatherv を実行し、結果は global_u/v/p に集約される
            MPI_Gatherv(send_u_flat_output.data(), gatherv_recvcounts_output[rank], MPI_FLOAT,
                          global_u.data(), gatherv_recvcounts_output.data(), gatherv_displs_output.data(), MPI_FLOAT,
                          0, MPI_COMM_WORLD);

            MPI_Gatherv(send_v_flat_output.data(), gatherv_recvcounts_output[rank], MPI_FLOAT,
                          global_v.data(), gatherv_recvcounts_output.data(), gatherv_displs_output.data(), MPI_FLOAT,
                          0, MPI_COMM_WORLD);

            MPI_Gatherv(send_p_flat_output.data(), gatherv_recvcounts_output[rank], MPI_FLOAT,
                          global_p.data(), gatherv_recvcounts_output.data(), gatherv_displs_output.data(), MPI_FLOAT,
                          0, MPI_COMM_WORLD);

            if (rank == 0)
            {
                // ランク0のみがファイルに書き込む (global_u/v/p を使用)
                for (int j = 0; j < ny; j++)
                {
                    for (int i = 0; i < nx; i++)
                    {
                        ufile << global_u[j * nx + i] << " ";
                    }
                }
                ufile << "\n"; 

                for (int j = 0; j < ny; j++)
                {
                    for (int i = 0; i < nx; i++)
                    {
                        vfile << global_v[j * nx + i] << " ";
                    }
                }
                vfile << "\n"; 

                for (int j = 0; j < ny; j++)
                {
                    for (int i = 0; i < nx; i++)
                    {
                        pfile << global_p[j * nx + i] << " ";
                    }
                }
                pfile << "\n"; 
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
