"""Render the full retained-replica Markdown table as a high-resolution PNG."""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'artifacts/retained-replica-profiles-20261021/combined-report.zh.md'
TARGET = SOURCE.with_name('combined-report.zh.table.png')
REGULAR = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
BOLD = '/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc'


def font(size, bold=False):
    return ImageFont.truetype(BOLD if bold else REGULAR, size, index=2)


def wrap(text, face, width, draw):
    lines, line = [], ''
    for char in text:
        if char == '\n':
            lines.append(line)
            line = ''
        elif draw.textlength(line + char, font=face) > width:
            lines.append(line)
            line = char
        else:
            line += char
    if line:
        lines.append(line)
    return lines


def main():
    source = SOURCE.read_text()
    table = [line.split('|')[1:-1] for line in source.splitlines() if line.startswith('|')]
    rows = table[2:]
    assert len(rows) == 67 and all(len(row) == 11 for row in rows)
    failures = []
    for row in rows:
        if row[3] != '完成':
            failures.append(f'[{len(failures) + 1}] {row[0]} / {row[1]} / n={row[2]}：{row[3]}')
            row[3] = f'失败 [{len(failures)}]'
    assert len(failures) == 6

    margin = 72
    widths = [540, 150, 110, 210, 240, 170, 350, 270, 270, 300, 260]
    edges = [margin]
    for width in widths:
        edges.append(edges[-1] + width)
    width = edges[-1] + margin
    top, header_height, row_height = 248, 114, 68
    body_top = top + header_height
    body_bottom = body_top + row_height * len(rows)
    face, small, bold = font(32), font(27), font(32, True)
    image = Image.new('RGB', (width, body_bottom + 1300), '#ffffff')
    draw = ImageDraw.Draw(image)
    ink, muted, grid = '#172b44', '#53657a', '#dfe6ee'

    draw.rectangle((0, 0, width, 14), fill='#167d9a')
    draw.text((margin, 52), '七模型多副本资源画像', font=font(62, True), fill=ink)
    draw.text((margin, 145), '完整测量表 · 16 个模型—任务组合 · 61 个完成档位 · 6 个边界失败档位', font=font(32), fill=muted)
    draw.text((margin, 193), '2026-10-21  |  每副本独立进程，batch=1；同模型、同任务、单 GPU 饱和闭环', font=small, fill=muted)

    headers = ['模型', '任务', '副本', '状态', '吞吐\nreq/s', '相对 n1', '平均 / p95 延迟\nms',
               '进程驻留合计\nGiB', '整卡测量峰\nGiB', '平均整卡功率\nW', '整卡能耗\nJ/req']
    draw.rectangle((margin, top, edges[-1], body_top), fill='#17324e')

    def centered(text, left, right, y0, y1, face=face, fill=ink):
        lines = text.split('\n')
        step = face.size + 10
        y = (y0 + y1 - step * len(lines)) / 2
        for line in lines:
            assert draw.textlength(line, font=face) <= right - left - 22, line
            draw.text(((left + right) / 2, y), line, anchor='mt', font=face, fill=fill)
            y += step

    for i, title in enumerate(headers):
        centered(title, edges[i], edges[i + 1], top, body_top, bold, '#ffffff')

    # Alternate whole task blocks and mark failures, keeping the source row order.
    group = -1
    previous = None
    for index, row in enumerate(rows):
        y0, y1 = body_top + index * row_height, body_top + (index + 1) * row_height
        key = tuple(row[:2])
        if key != previous:
            group += 1
            previous = key
        failed = row[3].startswith('失败')
        fill = '#fff1ed' if failed else ('#f3f7fb' if group % 2 == 0 else '#ffffff')
        draw.rectangle((margin, y0, edges[-1], y1), fill=fill)
        if not failed:
            draw.rectangle((edges[9], y0, edges[10], y1), fill='#e8f5f4' if group % 2 == 0 else '#f2fbf9')
        for col in range(2, 11):
            text = row[col] or '—'
            text_face = small if col == 8 and '含加载' in text else face
            color = '#b43e2e' if failed else ('#08766c' if col == 9 else ink)
            if col == 3 and not failed:
                color = '#297766'
            centered(text, edges[col], edges[col + 1], y0, y1, text_face, color)
        draw.line((edges[2], y1, edges[-1], y1), fill=grid, width=1)

    # Merge repeated model/task labels without omitting any measurement rows.
    for col in [0, 1]:
        start = 0
        while start < len(rows):
            end = start + 1
            while end < len(rows) and rows[end][:col + 1] == rows[start][:col + 1]:
                end += 1
            y0, y1 = body_top + start * row_height, body_top + end * row_height
            draw.rectangle((edges[col], y0, edges[col + 1], y1), fill='#edf2f8' if col == 0 else '#f6f8fb')
            centered(rows[start][col], edges[col], edges[col + 1], y0, y1, bold if col == 0 else face)
            draw.line((edges[col], y1, edges[col + 1], y1), fill='#b6c5d6', width=2)
            start = end
    for x in edges:
        draw.line((x, body_top, x, body_bottom), fill=grid, width=1)
    for index, row in enumerate(rows):
        if index and row[:2] != rows[index - 1][:2]:
            y = body_top + index * row_height
            draw.line((edges[1] if row[0] == rows[index - 1][0] else margin, y, edges[-1], y), fill='#b6c5d6', width=2)
    draw.line((margin, body_bottom, edges[-1], body_bottom), fill='#b6c5d6', width=2)

    y = body_bottom + 38
    notes = [
        '统计口径',
        '吞吐：三轮吞吐中位数；延迟：全部正式请求的平均值 / p95。图像固定64个输入，其余任务固定32个输入，每轮至少30秒并覆盖全部输入。',
        '平均整卡功率 = 三轮正式窗口功率积分总能量 ÷ 总时长；J/req = 同一总能量 ÷ 三轮请求总数。包含整卡基础功耗，不含CPU、加载、预热和轮次间隔。',
        '各档实际输入重复频次可能不同，倍率描述本次闭环轨迹；有限网格不证明最大容量、混合模型共驻留或开放请求率稳定性。',
        '失败档位不提供完整性能、平均功率和J/req；红色行的显存值为包含加载的采样峰值。失败原文如下：',
        *failures,
    ]
    for index, note in enumerate(notes):
        note_face = font(30, True) if index == 0 else small
        color = ink if index == 0 else ('#a04436' if index >= 5 else muted)
        for line in wrap(note, note_face, width - 2 * margin, draw):
            draw.text((margin, y), line, font=note_face, fill=color)
            y += note_face.size + 13
        y += 9
    y += 22
    draw.line((margin, y, edges[-1], y), fill=grid, width=2)
    y += 22
    draw.text((margin, y), '来源：combined-report.zh.md  |  保留原表全部67行及数值；功率列以浅绿色标示。', font=font(25), fill=muted)
    y += 70
    assert y <= image.height
    image = image.crop((0, 0, width, y))
    image.save(TARGET, dpi=(300, 300), optimize=True)
    print(f'{TARGET}\n{image.width} × {image.height} px; {TARGET.stat().st_size / 1024**2:.2f} MiB')


if __name__ == '__main__':
    main()
