#!/usr/bin/env python3
"""Generate Excel report from summary.json cheating detection data."""

import json
import sys
import os
import pandas as pd


def seconds_to_mmss(total_seconds):
    """Convert total seconds to MM:SS string."""
    minutes = int(total_seconds) // 60
    seconds = int(total_seconds) % 60
    return f"{minutes}:{seconds:02d}"


def format_duration(duration_sec):
    """Format duration in seconds to 'Xm Ys' string."""
    duration_sec = int(duration_sec)
    minutes = duration_sec // 60
    seconds = duration_sec % 60
    if minutes > 0 and seconds > 0:
        return f"{minutes}m {seconds}s"
    elif minutes > 0:
        return f"{minutes}m"
    else:
        return f"{seconds}s"


def generate_excel(summary_path, output_path):
    """Generate Excel report from a summary.json file."""
    with open(summary_path, 'r') as f:
        summary = json.load(f)

    alerts = summary.get("alerts", [])
    if not alerts:
        print("No alerts found in summary.json")
        return

    rows = []
    for i, alert in enumerate(alerts, 1):
        seats = alert["seats"]
        students = f"{seats[0]} & {seats[1]}"
        confidence = f"{alert['confidence'] * 100:.1f}%"
        tier = alert["tier"]

        # Get timestamps — find earliest start and latest end
        timestamps = alert.get("timestamps", [])
        all_starts = []
        all_ends = []
        time_period_strs = []

        for ts in timestamps:
            start_sec, end_sec = ts[0], ts[1]
            all_starts.append(start_sec)
            all_ends.append(end_sec)
            time_period_strs.append(
                f"{seconds_to_mmss(start_sec)} - {seconds_to_mmss(end_sec)}"
            )

        if all_starts and all_ends:
            earliest_start = min(all_starts)
            latest_end = max(all_ends)
            start_time = seconds_to_mmss(earliest_start)
            end_time = seconds_to_mmss(latest_end)
            duration = format_duration(latest_end - earliest_start)
        else:
            start_time = "N/A"
            end_time = "N/A"
            duration = "N/A"

        # Format behaviors
        behaviors = ", ".join(alert.get("behaviors", []))

        # All time periods for reference
        all_periods = ", ".join(time_period_strs)

        rows.append({
            'Inc_no': i,
            'Start_time': start_time,
            'End_time': end_time,
            'Length': duration,
            'Who (Seat Pair)': students,
            'Behaviors': behaviors,
            'Time Windows': all_periods,
            'Confidence': confidence,
            'Flag Type': tier,
        })

    df = pd.DataFrame(rows)

    # Sort by earliest start time
    def time_to_seconds(time_str):
        try:
            parts = time_str.split(':')
            return int(parts[0]) * 60 + int(parts[1])
        except Exception:
            return 0

    df['_sort_key'] = df['Start_time'].apply(time_to_seconds)
    df = df.sort_values('_sort_key').drop('_sort_key', axis=1).reset_index(drop=True)
    df['Inc_no'] = range(1, len(df) + 1)

    # Create Excel with formatting
    with pd.ExcelWriter(output_path, engine='xlsxwriter') as writer:
        df.to_excel(writer, sheet_name='Incidents', index=False)

        workbook = writer.book
        worksheet = writer.sheets['Incidents']

        header_format = workbook.add_format({
            'bold': True,
            'bg_color': '#B4C7E7',
            'border': 1,
            'align': 'center',
            'valign': 'vcenter',
        })

        cell_format = workbook.add_format({
            'border': 1,
            'align': 'left',
            'valign': 'top',
            'text_wrap': True,
        })

        very_high_format = workbook.add_format({
            'border': 1,
            'align': 'left',
            'valign': 'top',
            'text_wrap': True,
            'bg_color': '#FFC7CE',
            'font_color': '#9C0006',
        })

        high_format = workbook.add_format({
            'border': 1,
            'align': 'left',
            'valign': 'top',
            'text_wrap': True,
            'bg_color': '#FFEB9C',
            'font_color': '#9C6500',
        })

        # Column widths
        worksheet.set_column('A:A', 8)    # Inc_no
        worksheet.set_column('B:B', 12)   # Start_time
        worksheet.set_column('C:C', 12)   # End_time
        worksheet.set_column('D:D', 10)   # Length
        worksheet.set_column('E:E', 22)   # Who
        worksheet.set_column('F:F', 65)   # Behaviors
        worksheet.set_column('G:G', 50)   # Time Windows
        worksheet.set_column('H:H', 12)   # Confidence
        worksheet.set_column('I:I', 12)   # Flag Type

        # Write header
        for col_num, value in enumerate(df.columns.values):
            worksheet.write(0, col_num, value, header_format)

        # Write data with conditional formatting for tier
        for row_num in range(len(df)):
            tier = df.iloc[row_num]['Flag Type']
            if tier == 'VERY_HIGH':
                fmt = very_high_format
            elif tier == 'HIGH':
                fmt = high_format
            else:
                fmt = cell_format

            for col_num in range(len(df.columns)):
                worksheet.write(row_num + 1, col_num, df.iloc[row_num, col_num], fmt)

    print(f"Excel file created: {output_path}")
    print(f"Total incidents: {len(alerts)}")
    print(f"Video: {summary.get('video', 'N/A')}")
    print(f"Duration: {summary.get('duration', 0):.0f}s")
    print(f"Students detected: {summary.get('total_students', 'N/A')}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        test_dir = sys.argv[1]
    else:
        test_dir = "sidd_ccc/outputs_v2/test11"

    summary_path = os.path.join(test_dir, "summary.json")
    output_path = os.path.join(test_dir, "cheating_incidents.xlsx")

    if not os.path.exists(summary_path):
        print(f"Error: {summary_path} not found")
        sys.exit(1)

    generate_excel(summary_path, output_path)
