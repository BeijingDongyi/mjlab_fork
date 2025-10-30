"""
批量调用 csv_to_npz_phybot.py：
- 输入：一个包含许多 .csv 的目录（可含子目录）
- 行为：对每个 CSV 运行一次原脚本
- 输出名：与 CSV 文件名同名（不改路径，不创建输出目录）
- 进度：显示整体进度条
- 渲染：可选 --render
用法示例：
  python batch_csv_to_npz.py /path/to/mocap_dir --input-fps 30 --output-fps 50 --render
"""

import os
import sys
import glob
import argparse
import subprocess
from tqdm import tqdm
import re

def main():
    parser = argparse.ArgumentParser(description="Batch run csv_to_npz_phybot.py on a directory of CSV files.")
    parser.add_argument("--input_dir", type=str, help="包含多个 CSV 的目录（支持递归子目录）")
    parser.add_argument("--input-fps", type=int, default=30, help="输入 CSV 的帧率（默认 30）")
    parser.add_argument("--output-fps", type=int, default=50, help="输出目标帧率（默认 50）")
    parser.add_argument("--device", type=str, default="cuda:0", help="运行设备（默认 cuda:0）")
    parser.add_argument("--render", action="store_true", help="是否生成视频（传此开关则渲染）")
    parser.add_argument("--pattern", type=str, default="**/*.csv", help="匹配模式，默认递归匹配所有 CSV")
    args = parser.parse_args()

    root = os.path.abspath(args.input_dir)
    csv_files = sorted(glob.glob(os.path.join(root, args.pattern), recursive=True))
    # 只保留真实文件
    csv_files = [f for f in csv_files if os.path.isfile(f) and f.lower().endswith(".csv")]

    if not csv_files:
        print(f"❌ 未在目录中找到 CSV：{root}")
        sys.exit(1)

    print(f"✅ 找到 {len(csv_files)} 个 CSV，开始批量处理...\n")

    pbar = tqdm(
        total=len(csv_files),
        desc="批量转换进度",
        ncols=100,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
    )

    for csv_path in csv_files:
        # 以文件名（不带扩展名）作为 output-name
        base_name = os.path.splitext(os.path.basename(csv_path))[0]
        base_name = re.sub(r"[^0-9A-Za-z_\-]", "_", base_name)
        pbar.set_description(f"处理: {base_name}")

        # 构造命令：MUJOCO_GL=egl uv run src/mjlab/scripts/csv_to_npz_phybot.py ...
        cmd = [
            "env", "MUJOCO_GL=egl",
            "uv", "run", "src/mjlab/scripts/csv_to_npz_phybot.py",
            "--input-file", csv_path,
            "--output-name", base_name,
            "--input-fps", str(args.input_fps),
            "--output-fps", str(args.output_fps),
            "--device", args.device,
        ]
        if args.render:
            cmd.append("--render")

        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            # 打印错误但不中断整体流程
            pbar.write(f"⚠️ 失败：{csv_path}\n{e}")
        finally:
            pbar.update(1)

    pbar.close()
    print("\n✅ 全部文件处理完成！")

if __name__ == "__main__":
    main()
