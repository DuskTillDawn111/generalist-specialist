"""Merge deployment, coverage and shared-work modeling in the supplied DOCX."""
from pathlib import Path
from copy import deepcopy
from zipfile import ZipFile, ZIP_DEFLATED
import hashlib
import json
import re
from lxml import etree as E

OUT = Path(__file__).resolve().parent
SOURCE = Path('/root/.codex/attachments/6e724832-b623-41d0-9f4c-1feffd19bc15/GSdeploy改3_任务数量协同与短窗口实机实验版.docx')
DEST = OUT / 'GSdeploy改3_联合优化约束整合版.docx'
NS = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main',
      'm': 'http://schemas.openxmlformats.org/officeDocument/2006/math'}
Q = lambda s: '{' + NS[s.split(':')[0]] + '}' + s.split(':')[1]
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
text = lambda p: ''.join(p.xpath('.//w:t/text() | .//m:t/text()', namespaces=NS))


def node(tag, value=None):
    el = E.Element(Q(tag))
    if value is not None:
        el.set(Q(tag.split(':')[0] + ':val'), str(value))
    return el


def math_run(value):
    r = node('m:r')
    rp = node('w:rPr')
    fonts = node('w:rFonts')
    fonts.set(Q('w:ascii'), 'Cambria Math')
    fonts.set(Q('w:hAnsi'), 'Cambria Math')
    rp.append(fonts)
    rp.append(node('w:sz', '22'))
    r.append(rp)
    t = node('m:t')
    t.text = value
    t.set('{http://www.w3.org/XML/1998/namespace}space', 'preserve')
    r.append(t)
    return r


def formula_tokens(value):
    result = []
    pos = 0
    for match in re.finditer(r'([A-Za-zΑ-ω∑ℤ]+)((?:[_^]\{[^}]+\})+)', value):
        if match.start() > pos:
            result.append(math_run(value[pos:match.start()]))
        scripts = dict(re.findall(r'([_^])\{([^}]+)\}', match[2]))
        tag = 'm:sSubSup' if len(scripts) == 2 else ('m:sSub' if '_' in scripts else 'm:sSup')
        item = node(tag)
        base = node('m:e')
        base.append(math_run(match[1]))
        item.append(base)
        for mark, slot in [('_', 'm:sub'), ('^', 'm:sup')]:
            if mark in scripts:
                child = node(slot)
                child.append(math_run(scripts[mark]))
                item.append(child)
        result.append(item)
        pos = match.end()
    if pos < len(value):
        result.append(math_run(value[pos:]))
    return result


def paragraph(template, value, heading=False):
    p = deepcopy(template)
    for child in list(p):
        if child.tag != Q('w:pPr'):
            p.remove(child)
    # New paragraphs must have unique identities instead of copied WPS paraIds.
    for key in list(p.attrib):
        if key.endswith('}paraId') or key.endswith('}textId'):
            del p.attrib[key]
    pp = p.find('w:pPr', NS)
    for child in list(pp):
        if child.tag in [Q('w:keepNext'), Q('w:pageBreakBefore')]:
            pp.remove(child)
    if heading:
        pp.append(node('w:keepNext'))
    r = node('w:r')
    rp = node('w:rPr')
    fonts = node('w:rFonts')
    for key, val in [('ascii', 'Times New Roman'), ('hAnsi', 'Times New Roman'), ('eastAsia', '宋体')]:
        fonts.set(Q('w:' + key), val)
    rp.append(fonts)
    rp.append(node('w:sz', '23' if heading else '22'))
    rp.append(node('w:b', '1' if heading else '0'))
    r.append(rp)
    t = node('w:t')
    t.text = value
    t.set('{http://www.w3.org/XML/1998/namespace}space', 'preserve')
    r.append(t)
    p.append(r)
    return p


def equations(template, lines):
    p = paragraph(template, '')
    p.remove(p[-1])
    pp = p.find('w:pPr', NS)
    for child in list(pp):
        if child.tag in [Q('w:ind'), Q('w:jc'), Q('w:spacing')]:
            pp.remove(child)
    pp.append(node('w:jc', 'center'))
    spacing = node('w:spacing')
    spacing.set(Q('w:before'), '60')
    spacing.set(Q('w:after'), '120')
    pp.append(spacing)
    pp.append(node('w:keepLines'))
    omp = node('m:oMathPara')
    omppr = node('m:oMathParaPr')
    omppr.append(node('m:jc', 'center'))
    omp.append(omppr)
    om = node('m:oMath')
    arr = node('m:eqArr')
    pr = node('m:eqArrPr')
    pr.append(node('m:baseJc', 'center'))
    pr.append(node('m:rSpRule', '3'))
    pr.append(node('m:rSp', '6'))
    arr.append(pr)
    for line in lines:
        row = node('m:e')
        row.extend(formula_tokens(line))
        arr.append(row)
    om.append(arr)
    omp.append(om)
    p.append(omp)
    return p


def main():
    original_hash = sha(SOURCE)
    with ZipFile(SOURCE) as z:
        parts = {n: z.read(n) for n in z.namelist()}
    root = E.fromstring(parts['word/document.xml'])
    body = root.find('w:body', NS)
    old = list(body)
    paras = body.findall('w:p', NS)
    assert len(paras) == 294
    assert text(paras[98]) == '联合优化约束：'
    assert text(paras[110]) == '共享覆盖与共享工作组成'
    assert text(paras[121]) == '三. 算法设计'
    begin, end = body.index(paras[98]), body.index(paras[119]) + 1
    P = lambda value: paragraph(paras[111], value)
    EQ = lambda lines: equations(paras[112], lines)
    revised = [
        paragraph(paras[98], '联合优化约束', heading=True),
        P('联合优化同时决定模型驻留、副本数量与任务分流，并由同一组变量确定跨任务覆盖和共享副本上的工作组成。'
          '令候选模型池 M = G ∪ S，C = (x,n,r) 为联合部署方案。其约束由驻留与资源可行性、'
          '共享覆盖一致性，以及请求分配后的共享工作与队列关系共同组成，统一写为下述联合关系。'),
        EQ([
            '∑_{m∈M} r_{t,m} = 1，                             ∀t∈T，',
            '0 ≤ r_{t,m} ≤ x_{m} y_{t,m}，                   ∀t∈T，m∈M，',
            'x_{m} ≤ n_{m} ≤ N_{m}^{max} x_{m}，                 ∀m∈M，',
            'x_{m} ∈ {0,1}， n_{m} ∈ ℤ_{≥0}，                  ∀m∈M，',
            '∑_{m∈M} Mem_{m}(n_{m}) ≤ Mem_{max}。                    (7)',
        ]),
        P('其中，模型能力参数 yₜ,ₘ 表示模型 m 是否支持任务 t。上述约束保证各任务完整分流，'
          '请求只交给已驻留且具有对应能力的模型；xₘ = 0 时 nₘ = 0，xₘ = 1 时至少部署一个副本。'
          'Memₘ(0) = 0，显存按模型的副本状态统一计入预算。同一 G 承接多个任务时，'
          '其驻留显存按实际副本数计入一次，不按覆盖任务数重复计算。'),
        P('在这一可行域内，G 的跨任务覆盖允许系统选择性部署 S。定义 qₜᴳ 为任务 t 分给所有通用模型的总比例，'
          'αₜ 为分给所有专用模型的总比例，并将它们与原路由变量绑定：'),
        EQ([
            'q_{t}^{G} = ∑_{g∈G} r_{t,g}， α_{t} = ∑_{s∈S} r_{t,s}，',
            'q_{t}^{G} + α_{t} = 1， 0 ≤ q_{t}^{G}, α_{t} ≤ 1， ∀t∈T。        (7a)',
        ]),
        P('式（7a）与式（7）的路由能力约束共同保证共享覆盖：对于单 G、每任务一个 S 的情况，'
          '有 αₜ ≤ xSₜ、1−αₜ ≤ xG，未部署对应 S 的任务仍由 G 保持覆盖。'
          'qₜᴳ 和 αₜ 均由 r 派生；多个 G 并存时，优化仍保留每个 rₜ,g，分别决定各通用模型承接的任务比例。'),
        P('共享覆盖进一步决定共享副本的工作组成。给定 C 和请求轨迹 W，先通过平滑加权轮询（SWRR）选择模型，'
          '再通过同一模型内跨任务共享的副本轮询（RR）确定执行位置 πᵢ(C) = (mᵢ,jᵢ)。'
          '不同任务分配到同一副本后共同进入该副本的 FIFO 队列。设 aᵢ 为到达时间，'
          'bᵢ 为请求 i 在所分配模型上的校准独立服务时长，prev(i) 为该副本上的前驱请求；'
          '首条请求的前驱完成时间取 0。以独立服务时长为基准，共享队列满足：'),
        deepcopy(paras[114]),
        deepcopy(paras[115]),
        P('式（7b）、（7c）中的 Lₘ,ⱼ 为副本独立总工作量，ℓᵢ⁰ = fᵢ⁰−aᵢ 为独立 FIFO 响应时延。'
          '两式均使用由同一组 (n,r) 生成的 πᵢ(C)，各任务的请求在实际副本上合并累计工作和递推等待。'
          '因此，改变某类任务的分流会改变共享模型的工作组成，并影响仍由该模型处理的其他任务。'
          'G 的驻留成本可以跨任务共享，各请求的执行工作仍逐条累计。'),
        P('在上述联合关系下，将完整效用写为 U = Uₖ + Rfull，以区分共享队列收益与共驻执行、能耗影响。'
          'Uₖ 由实际请求质量和独立共享 FIFO 计算，Rfull 包含共驻时延修正与整卡能量：'),
        deepcopy(paras[117]),
        deepcopy(paras[118]),
        P('上述联合优化约束将资源、覆盖与共享工作绑定在同一部署方案上：式（7）和（7a）规定部署与覆盖的可行性，'
          '式（7b）和（7c）规定分流后的共享工作与独立 FIFO 计算关系。完整时延和整卡能耗还取决于请求重叠、'
          '动态执行速率及排空过程，由第三章的事件模拟器计算。式（7d）用于效用来源分析及结构评分对照，'
          '正式代理的训练目标仍见式（21）、（22）。联合求解利用这些已知结构构造候选与代理，并以完整预测器评价候选效用。'),
    ]
    # One editable constraint system, including the original rich FIFO/work equations.
    joint = EQ([])
    arr = joint.find('.//m:eqArr', NS)
    for source in [revised[2], revised[5]]:
        for row in source.findall('.//m:eqArr/m:e', NS):
            arr.append(deepcopy(row))
    assignment = node('m:e')
    assignment.extend(formula_tokens('π_{i}(C) = RR_{n}(SWRR_{r}(t_{i}))， ∀i，'))
    arr.append(assignment)
    original_shared_rows = []
    for source_i in [114, 115]:
        row = node('m:e')
        original_math = paras[source_i].find('.//m:oMath', NS)
        row.extend(deepcopy(list(original_math)))
        arr.append(row)
        original_shared_rows.append(row)
    # A single brace connects all groups to the same constrained optimization.
    math = joint.find('.//m:oMath', NS)
    math.remove(arr)
    delim = node('m:d')
    delimpr = node('m:dPr')
    delimpr.append(node('m:begChr', '{'))
    delimpr.append(node('m:endChr', ''))
    delimpr.append(node('m:grow', '1'))
    delim.append(delimpr)
    content = node('m:e')
    content.append(arr)
    delim.append(content)
    math.append(math_run('s.t.  '))
    math.append(delim)
    for size in joint.findall('.//w:sz', NS):
        size.set(Q('w:val'), '21')
    definition = P('其中，qₜᴳ 和 αₜ 分别为任务 t 交给所有 G、所有 S 的总比例；'
                   'πᵢ(C) = (mᵢ,jᵢ) 为请求 i 经 SWRR 与跨任务共享的副本 RR 得到的执行位置。'
                   'aᵢ 为到达时间，bᵢ 为该请求在所分配模型上的校准独立服务时长；'
                   'sᵢ⁰、fᵢ⁰、wᵢ⁰ 为独立 FIFO 的开始、完成及等待时间。'
                   'prev(i) 为同一副本的前驱请求，首条请求的前驱完成时间取 0。')
    revised = [revised[0], revised[1], definition, joint,
               revised[3], revised[6], revised[10], revised[11],
               revised[12], revised[13], revised[14]]
    for item in old[begin:end]:
        body.remove(item)
    for i, item in enumerate(revised):
        body.insert(begin + i, item)
    parts['word/document.xml'] = E.tostring(root, xml_declaration=True, encoding='UTF-8', standalone=True)
    with ZipFile(SOURCE) as src, ZipFile(DEST, 'w', ZIP_DEFLATED) as dst:
        for info in src.infolist():
            dst.writestr(info, parts[info.filename])

    canonical = lambda x: E.tostring(x, method='c14n', exclusive=True)
    final = list(body)
    assert all(canonical(a) == canonical(b) for a, b in zip(old[:begin], final[:begin]))
    assert all(canonical(a) == canonical(b) for a, b in zip(old[end:], final[begin + len(revised):]))
    assert len(final) == len(old) - (end - begin) + len(revised)
    with ZipFile(SOURCE) as src, ZipFile(DEST) as dst:
        assert src.namelist() == dst.namelist()
        assert dst.testzip() is None
        changed = [name for name in src.namelist() if src.read(name) != dst.read(name)]
        assert changed == ['word/document.xml'], changed
    assert sha(SOURCE) == original_hash
    fulltext = text(body)
    assert '共享覆盖与共享工作组成' not in fulltext
    assert not any('_{' in (t.text or '') or '^{' in (t.text or '')
                   for t in joint.findall('.//m:t', NS))
    for label in ['(7)', '(7a)', '(7b)', '(7c)', '(7d)']:
        assert fulltext.count(label) == 1, label
    # FIFO and workload retain their original math structure and values; only font size changes.
    for source_i, new_row in zip([114, 115], original_shared_rows):
        expected = deepcopy(paras[source_i].find('.//m:oMath', NS))
        for size in expected.findall('.//w:sz', NS):
            size.set(Q('w:val'), '21')
        assert [canonical(x) for x in expected] == [canonical(x) for x in new_row]
    for source_i in [117, 118]:
        assert any(canonical(paras[source_i]) == canonical(x) for x in revised)
    (OUT / 'merged-section.zh.txt').write_text('\n\n'.join(text(x) for x in revised), encoding='utf-8')
    audit = {'source': str(SOURCE), 'source_sha256': original_hash,
             'result': str(DEST), 'result_sha256': sha(DEST),
             'replaced_source_paragraph_indices': [98, 119],
             'changed_package_parts': changed,
             'single_section_title': '联合优化约束',
             'one_editable_constraint_system_includes_resource_coverage_assignment_and_work': True,
             'former_separate_shared_coverage_heading_removed': True,
             'deployment_replica_link': 'x_m <= n_m <= N_m_max * x_m',
             'coverage_routing_identity': 'q_t^G + alpha_t = 1; both derived from r',
             'shared_work_queue_relations_retained': ['7b', '7c'],
             'utility_decomposition_kept_as_analysis': '7d',
             'equation_labels_and_later_references_preserved': True,
             'all_other_body_blocks_unchanged': True,
             'all_tables_and_experiment_content_unchanged': True,
             'all_original_media_and_other_package_parts_unchanged': True,
             'optimization_code_or_experiments_modified': False}
    (OUT / 'revision-audit.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
