import os
from collections import defaultdict

import pandas as pd


EVENT_NAMES = {
    0: "正常发电",
    1: "计划检修",
    2: "设备故障",
    3: "电网限电",
}


def _event_segments(events):
    segments = []
    start = 0
    values = list(events)
    for idx in range(1, len(values) + 1):
        if idx == len(values) or values[idx] != values[start]:
            event_type = int(values[start])
            if event_type != 0:
                segments.append((event_type, idx - start))
            start = idx
    return segments


def summarize_station(csv_path, train_ratio=0.8):
    df = pd.read_csv(csv_path, parse_dates=["Timestamp"])
    station = os.path.splitext(os.path.basename(csv_path))[0].split("_")[-1]
    split_idx = int(len(df) * train_ratio)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]

    station_summary = {
        "场站": f"场站 {station}",
        "样本数": len(df),
        "采样间隔": "1 h",
        "起止时间": f"{df['Timestamp'].min():%Y-%m-%d} 至 {df['Timestamp'].max():%Y-%m-%d}",
        "训练集时间": f"{train_df['Timestamp'].min():%Y-%m-%d} 至 {train_df['Timestamp'].max():%Y-%m-%d}",
        "测试集时间": f"{test_df['Timestamp'].min():%Y-%m-%d} 至 {test_df['Timestamp'].max():%Y-%m-%d}",
        "训练/测试比例": f"{train_ratio:.0%}/{1-train_ratio:.0%}",
    }

    event_rows = []
    counts = df["Event_Type"].value_counts().sort_index()
    for event_type in range(4):
        count = int(counts.get(event_type, 0))
        event_rows.append(
            {
                "场站": f"场站 {station}",
                "事件类型": f"{event_type}-{EVENT_NAMES[event_type]}",
                "小时数": count,
                "占比": f"{count / len(df) * 100:.2f}%",
            }
        )

    grouped = defaultdict(list)
    for event_type, duration in _event_segments(df["Event_Type"].values):
        grouped[event_type].append(duration)

    duration_rows = []
    for event_type in [1, 2, 3]:
        durations = grouped[event_type]
        duration_rows.append(
            {
                "场站": f"场站 {station}",
                "事件类型": f"{event_type}-{EVENT_NAMES[event_type]}",
                "事件段数": len(durations),
                "累计小时数": int(sum(durations)),
                "平均持续时间(h)": f"{sum(durations) / len(durations):.2f}" if durations else "0.00",
                "最长持续时间(h)": max(durations) if durations else 0,
            }
        )

    return station_summary, event_rows, duration_rows


def write_markdown(output_path, station_rows, event_rows, duration_rows):
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("# 实验协议与数据统计补充\n\n")
        f.write("本文件由 `src/scripts/summarize_protocol.py` 根据 `src/data/station_*.csv` 自动生成，用于论文实验协议、事件分布和可复现性说明。\n\n")
        f.write("## 数据集与时间切分\n\n")
        f.write(pd.DataFrame(station_rows).to_markdown(index=False))
        f.write("\n\n## 事件类型小时分布\n\n")
        f.write(pd.DataFrame(event_rows).to_markdown(index=False))
        f.write("\n\n## 异常事件段持续时间统计\n\n")
        f.write(pd.DataFrame(duration_rows).to_markdown(index=False))
        f.write("\n")


def main():
    workspace_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    data_dir = os.path.join(workspace_dir, "src", "data")
    results_dir = os.path.join(workspace_dir, "src", "results")
    os.makedirs(results_dir, exist_ok=True)

    station_rows = []
    event_rows = []
    duration_rows = []
    for name in ["station_A.csv", "station_B.csv", "station_C.csv"]:
        station, events, durations = summarize_station(os.path.join(data_dir, name))
        station_rows.append(station)
        event_rows.extend(events)
        duration_rows.extend(durations)

    pd.DataFrame(station_rows).to_csv(os.path.join(results_dir, "dataset_split_summary.csv"), index=False, encoding="utf-8-sig")
    pd.DataFrame(event_rows).to_csv(os.path.join(results_dir, "event_distribution.csv"), index=False, encoding="utf-8-sig")
    pd.DataFrame(duration_rows).to_csv(os.path.join(results_dir, "event_duration_summary.csv"), index=False, encoding="utf-8-sig")
    write_markdown(os.path.join(workspace_dir, "实验协议与数据统计补充.md"), station_rows, event_rows, duration_rows)


if __name__ == "__main__":
    main()
