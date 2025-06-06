import matplotlib.pyplot as plt
import numpy as np
import os # osモジュールを追加

NX = 41
NY = 41

def main():
    x = np.linspace(0, 2, NX)
    y = np.linspace(0, 2, NY)
    X, Y = np.meshgrid(x, y)

    u = np.zeros((NY, NX))
    v = np.zeros((NY, NX))
    p = np.zeros((NY, NX))

    with open('u.dat', 'r') as f:
        uraw = f.readlines()
    with open('v.dat', 'r') as f:
        vraw = f.readlines()
    with open('p.dat', 'r') as f:
        praw = f.readlines()

    # 画像保存用のディレクトリを作成
    output_dir = 'output_frames'
    os.makedirs(output_dir, exist_ok=True) # 既に存在してもエラーにならない

    for n in range(len(uraw)):
        plt.clf() # 既存の図をクリア

        u_flattened = [float(val) for val in uraw[n].strip().split() if val]
        v_flattened = [float(val) for val in vraw[n].strip().split() if val]
        p_flattened = [float(val) for val in praw[n].strip().split() if val]

        # データ長のエラーチェックはそのまま残しておくことを推奨
        expected_len_per_line = NX * NY
        if len(u_flattened) != expected_len_per_line or \
           len(v_flattened) != expected_len_per_line or \
           len(p_flattened) != expected_len_per_line:
            print(f"警告: タイムステップ {n} でデータ長が期待値と異なります！")
            # ここで処理を中断するか、あるいはスキップするなどの対応を検討
            # 例: continue # このステップをスキップして次のステップへ
            # 例: break    # 全体のループを中断
            # 現状ではそのまま処理を進めるので、エラーが出れば落ちる可能性があります。
            # 原因が解決していないなら、エラーが出る前にここで止めるべきかもしれません。

        for j in range(NY):
            for i in range(NX):
                idx = j * NX + i
                u[j, i] = u_flattened[idx]
                v[j, i] = v_flattened[idx]
                p[j, i] = p_flattened[idx]

        plt.contourf(X, Y, p, alpha=0.5, cmap=plt.cm.coolwarm)
        plt.quiver(X[::2, ::2], Y[::2, ::2], u[::2, ::2], v[::2, ::2])
        plt.title(f'C++, n = {n}')

        # ここで画像として保存
        frame_filename = os.path.join(output_dir, f'cavity_frame_{n:04d}.png')
        plt.savefig(frame_filename)
        print(f"Saved {frame_filename}") # 保存したことを表示

    plt.close() # 全てのmatplotlibのFigureを閉じる（メモリ解放のため重要）

if __name__ == '__main__':
    main()
