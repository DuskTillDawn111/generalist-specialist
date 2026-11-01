"""Plot completed strict text26 audit, preserving invalid U as missing."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def receipt(path):
    return dict(path=str(path.resolve()), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    audit = json.loads(args.audit.read_text())
    manifest_path = args.audit.parent/'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    if manifest['files'][args.audit.name] != receipt(args.audit)['sha256']:
        raise ValueError('Strict audit manifest mismatch')
    if audit['status'] != 'ALL_PREDECLARED_ATTEMPTS_AUDITED' or len(audit['conditions']) != 4:
        raise ValueError('Exactly four final audited attempts required')
    conditions = audit['conditions']
    if any(c.get('status') == 'FAILED_ATTEMPT_AUDIT' or not c['identity']['verified'] for c in conditions):
        raise ValueError('Identity/audit errors must be resolved before plotting')
    args.output.mkdir(parents=True, exist_ok=False)
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True, sharey=True)
    maximum_backlog = 0
    for ax, condition in zip(axes.flat, conditions):
        name = condition['label'].replace('text26_', '').replace('_specialist_share_', ', specialist share ')
        evidence = condition['stability_evidence'] or {}
        bins = evidence.get('backlog_timeline', [])
        if not bins:
            raise ValueError('Missing audited backlog evidence')
        ax.plot([b['offset_seconds'] for b in bins], [b['mean_backlog_requests'] for b in bins],
                marker='o', markersize=3, label='Submitted backlog, 30s bin mean')
        offered = evidence.get('offered_backlog_diagnostic', {}).get('bins', [])
        maximum_backlog = max(maximum_backlog, *[b['mean_backlog_requests'] for b in bins+offered])
        if offered:
            ax.plot([b['offset_seconds'] for b in offered], [b['mean_backlog_requests'] for b in offered],
                    linestyle='--', label='Offered backlog, 30s bin mean')
        utility = condition['utility']['U']
        eligible = condition['utility_comparison_allowed']
        if (utility is None) == eligible:
            raise ValueError('U eligibility inconsistency')
        tasks = condition['request_summaries']['task']
        count = sum(t['offered'] for t in tasks.values())
        completed = sum(t['completed'] for t in tasks.values())
        state = condition['stability']['stability_state_v2']
        ax.set_title(name.replace('4of7', '4/7').replace('1of2', '1/2'))
        score = f'{utility:.6f}' if eligible else 'null (ineligible)'
        memory_note = '\nSafe memory exceeds budget' if condition['utility'].get('M_norm', 0) > 1 else ''
        ax.text(.02, .97, f'Completed {completed}/{count}; {state}\nU = {score}{memory_note}',
                transform=ax.transAxes, va='top', fontsize=9,
                bbox=dict(facecolor='white', edgecolor='0.8', alpha=.85))
        ax.grid(alpha=.25)
        ax.set_xlim(0, 600)
    axes[0, 0].set_ylim(0, maximum_backlog*1.08)
    for ax in axes[-1]:
        ax.set_xlabel('Measurement time (s); drain excluded')
    for ax in axes[:, 0]:
        ax.set_ylabel('Backlog (requests)')
    axes[0, 0].legend(loc='lower right', fontsize=8)
    fig.suptitle('Text 2.6 requests/s: four predeclared deployment / routing conditions', fontsize=15)
    fig.text(.5, .01, 'One seed51 trace; fixed utility and stability rules. Invalid U stays null. Limited development comparison, no full-pool optimum.',
             ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .03, 1, .95))
    for extension in ['png', 'pdf']:
        fig.savefig(args.output/('four-condition-backlogs.'+extension), dpi=180)
    plt.close(fig)
    (args.output/'README.zh.md').write_text(
        '# 文本2.6四条件积压图\n\n'
        '实线为实际send-start已提交减完成请求的30秒bin平均积压；虚线为计划到达减完成，'
        '包含尚未提交到worker的主机端等待。图只展示600秒到达窗口，排空另见严格审计。\n\n'
        'U直接引用严格审计，无效U保持null，不补零、不按诊断U排名。'
        '四条预声明开发条件、同seed51，不证明独立确认或全模型池最优。\n')
    result = dict(schema='gsdeploy.text26_development_figures.v1', inputs=[receipt(args.audit), receipt(manifest_path)],
                  script=receipt(Path(__file__)), gpu_started=False, raw_files_read=False,
                  outputs=[receipt(p) for p in sorted(args.output.iterdir())])
    (args.output/'manifest.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(status='PLOTTED_AUDITED_FOUR_CONDITIONS', output=str(args.output))))


if __name__ == '__main__':
    main()
