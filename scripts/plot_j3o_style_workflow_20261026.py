"""Read-only method illustration: no solver, worker or GPU experiments.

Creates a fresh directory with paper-style algorithms, flowchart, equations,
LaTeX source and a source/artefact receipt. Historical experiments are untouched.
"""
from pathlib import Path
import argparse
import hashlib
import json
import shutil
import textwrap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch, Polygon
from matplotlib.backends.backend_pdf import PdfPages

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / 'docs/j3o_style_static_deployment_model_20261026.md'
plt.rcParams.update({'font.family': 'DejaVu Serif', 'mathtext.fontset': 'dejavuserif',
                     'font.size': 9, 'pdf.fonttype': 42, 'svg.fonttype': 'none'})


def export(fig, output, name):
    for ext in ('pdf', 'svg', 'png'):
        fig.savefig(output / f'{name}.{ext}', dpi=240, bbox_inches='tight', pad_inches=.08)
    plt.close(fig)


def block(ax, x, y, w, h, label, fs=8, dashed=False):
    patch = Rectangle((x,y),w,h,fill=False,lw=.8,ls='--' if dashed else '-',zorder=3,facecolor='white')
    ax.add_patch(patch)
    label='\n'.join(textwrap.fill(line,width=max(24,round(w*118))) for line in label.split('\n'))
    txt=ax.text(x+w/2,y+h/2,label,ha='center',va='center',fontsize=fs,linespacing=1.20,zorder=4)
    renderer=ax.figure.canvas.get_renderer()
    bounds=patch.get_window_extent(renderer)
    while fs>5.8:
        extent=txt.get_window_extent(renderer)
        if extent.width<bounds.width*.93 and extent.height<bounds.height*.88:break
        fs-=.15;txt.set_fontsize(fs)
    return (x,y,w,h)


def port(box, side):
    x,y,w,h=box
    return {'top':(x+w/2,y+h),'bottom':(x+w/2,y),'left':(x,y+h/2),'right':(x+w,y+h/2)}[side]


def edge(ax, a, b, source='bottom', dest='top', text=None, via=(), offset=(0,0), dashed=False):
    points=[port(a,source),*via,port(b,dest)]
    for p,q in zip(points[:-2],points[1:-1]):
        ax.plot([p[0],q[0]],[p[1],q[1]],c='black',lw=.75,ls='--' if dashed else '-',zorder=1)
    ax.add_patch(FancyArrowPatch(points[-2],points[-1],arrowstyle='-|>',mutation_scale=9,
                  lw=.75,color='black',linestyle='--' if dashed else '-',shrinkA=0,shrinkB=1,zorder=1))
    if text:
        p,q=points[-2:]
        ax.text((p[0]+q[0])/2+offset[0],(p[1]+q[1])/2+offset[1],text,
                fontsize=7,ha='center',va='center',zorder=5,
                bbox={'facecolor':'white','edgecolor':'none','pad':.4})


def flowchart(output):
    fig,ax=plt.subplots(figsize=(8.2,8.5));ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off')
    ax.text(.25,.985,'(a) Static search and verification',ha='center',va='top',fontsize=10)
    ax.text(.76,.985,'(b) Online spillover experiment',ha='center',va='top',fontsize=10)
    ax.plot([.51,.51],[.06,.97],color='.7',lw=.5)
    l=[]
    specs=[(.865,.070,'Freeze seven-model pool, profiles, trace\nreplica states and rational routes (denominator ≤ K)'),
           (.750,.070,'Joint MILP master\nreplica-state + task-route columns'),
           (.635,.070,'Full frozen predictor\nFIFO replay + quality + memory + energy'),
           (.520,.070,'Update incumbent and feedback\nresidual identity cut / invalid cut / UNKNOWN'),
           (.405,.070,'Bound gap ≤ ε and evidence complete?\nyes: return plan; no: continue / report budget'),
           (.270,.075,'Independent same-domain DFS\ncompare optimum, ties, time and full evaluations'),
           (.135,.075,'Paired native GPU confirmation\nwinner, strongest other deployment/route, controls')]
    for y,h,t in specs:l.append(block(ax,.07,y,.37,h,t,fs=7.25))
    for a,b in zip(l,l[1:]):edge(ax,a,b)
    edge(ax,l[3],l[1],source='left',dest='left',via=((.025,.555),(.025,.785)),text='feedback',offset=(0,.018))
    # right column is the separate already-implemented conditional policy study.
    r=[]
    rs=[(.865,.070,'Current arrival + visible host queues\ncompleted feedback; fixed resident deployment'),
        (.750,.070,'For each destination and service scenario\ncopy the same complete observed cohort'),
        (.635,.070,'Shared-rate event forecast\nreprice ALL active paths; integrate board power'),
        (.520,.070,'Recompute conditional Q / T / L / P / E\ncompare with best resident specialist'),
        (.405,.070,'All-scenario improvement + gates pass?\nyes: generalist; otherwise: specialist / fallback'),
        (.270,.075,'Submit and observe completion feedback\nexposure tokens, drift cooldown, deterministic audit'),
        (.135,.075,'Matched native spill / no-spill ablation\nrecompute FULL six-term U and stability')]
    for y,h,t in rs:r.append(block(ax,.585,y,.37,h,t,fs=7.25))
    for a,b in zip(r,r[1:]):edge(ax,a,b)
    edge(ax,r[5],r[0],source='right',dest='right',via=((.985,.3075),(.985,.9)),dashed=True)
    ax.text(.5,.077,'Separate evidence: finite-domain optimum  ·  solver efficiency  ·  measured spillover gain',
            ha='center',fontsize=8)
    block(ax,.07,.008,.885,.043,'Prospective extension: resource mode / SM binding / admission-aware prediction / kernel overlap model',fs=7.15,dashed=True)
    export(fig,output,'01-workflow')


A1=[
(0,'Build frozen domain; generate replica/route columns.'),
(0,r'Validate lower bound $\mathcal{B}$; build master MP.'),
(0,r'$C_{best}\leftarrow +\infty$; Unknown $\leftarrow\varnothing$.'),
(0,'while time and iteration budgets remain do'),
(1,'Solve MP; obtain integer candidate and valid bound.'),
(1,r'if gap $\leq\epsilon$ and evidence is resolved then'),
(2,'return incumbent with finite-domain certificate.'),
(1,r'Decode candidate $(n,r)$; run full frozen evaluator.'),
(1,'if a required observation is UNKNOWN then'),
(2,'Retain its optimistic bound; exclude identity from MP.'),
(1,'else if infeasible / incomplete after drain then'),
(2,'Add complete-identity no-good constraint (16).'),
(1,'else'),
(2,'Compute all six terms; update best feasible cost.'),
(2,r'Check $\mathcal{B}(v)\leq C(v)$; add residual cut (15).'),
(0,'Return plan, gap, UNKNOWN list and budget status.')]
A2=[
(0,'Observe returned timings; update cooldown feedback.'),
(0,'Read complete visible queues and current request.'),
(0,'if pending exceeds forecast budget then'),
(1,'return specialist fallback; do not truncate queues.'),
(0,'foreach supported resident destination a do'),
(1,'Append current request to a copy of queue a.'),
(1,r'foreach scenario $\omega\in\{mean,p90\}$ do'),
(2,'Forecast complete observed cohort by events (18).'),
(2,'Reprice all active heads after each transition.'),
(2,'Integrate whole-board remaining energy once.'),
(2,'Compute conditional prefix cost (19).'),
(1,'Mark destination UNKNOWN if evidence is missing.'),
(0,'Select best known specialist baseline (20).'),
(0,'Find generalists strictly better in every scenario.'),
(0,'Apply exposure / cooldown / support gates (21).'),
(0,'Return generalist or specialist; record decision audit.')]
A3=[
(0,'Freeze metric definitions, references and experiment identity.'),
(0,'Profile standalone replicas and selected co-location conditions.'),
(0,'Validate predictor on held-out conditions; retain failed versions.'),
(0,'Run Algorithm 1 on the frozen finite domain.'),
(0,'Run independent plain and bounded DFS on exactly the same domain.'),
(0,'Compare objective, full tie set, certificate, time and evaluation count.'),
(0,'Re-optimize strongest different deployment / route / no-split controls.'),
(0,'Deduplicate identities; preserve pure-G and pure-S structural controls.'),
(0,'Run paired native candidates on matched input and arrival traces.'),
(0,'For online spillover, compare Algorithm 2 with same-residency no-spill.'),
(0,'Audit raw cohort, timings, actual output quality, memory and board energy.'),
(0,'Apply unchanged stability rules; recompute full six-term utility.'),
(0,'Enter reserved confirmation only if the prespecified gate passes.'),
(0,'Report finite-domain correctness, solver efficiency and hardware evidence separately.')]


def algorithm(ax,title,inputs,outputs,lines,small=False):
    ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off')
    ax.plot([.01,.99],[.985,.985],c='black',lw=1)
    ax.text(.025,.959,title,fontsize=10 if not small else 9.7,va='top',weight='bold')
    ax.plot([.01,.99],[.912,.912],c='black',lw=.6)
    ax.text(.025,.893,'Input: '+inputs,fontsize=7.8,va='top',linespacing=1.35)
    ax.text(.025,.810,'Output: '+outputs,fontsize=7.8,va='top',linespacing=1.35)
    y=.707;dy=.038 if small else .037
    for i,(indent,line) in enumerate(lines,1):
        ax.text(.045,y,str(i),fontsize=7.2,ha='right',va='center')
        ax.text(.069+indent*.035,y,line,fontsize=7.45 if small else 7.65,va='center')
        y-=dy
    ax.plot([.01,.99],[y+.01,y+.01],c='black',lw=1)


def algorithms(output):
    fig,axes=plt.subplots(1,2,figsize=(12.8,6.1));fig.subplots_adjust(wspace=.04,left=.02,right=.98,top=.98,bottom=.03)
    algorithm(axes[0],'Algorithm 1: Joint deployment–routing search',
              'frozen model/replica pool, workload, trace, profiles;\n             route denominator K, tolerance ε and solve budget.',
              'deployment, fixed route, utility; certificate / gap.',A1)
    algorithm(axes[1],'Algorithm 2: Conditional shared-GPU spillover',
              'current arrival, visible history, resident workers;\n             frozen service/power model and objective weights.',
              'destination, conditional score and fallback reason.',A2)
    export(fig,output,'02-algorithms-1-2')
    fig,ax=plt.subplots(figsize=(9,6.2));fig.subplots_adjust(left=.025,right=.975,top=.98,bottom=.03)
    algorithm(ax,'Algorithm 3: Search and hardware verification',
              'calibration / holdout / confirmation sets; common six-term contract and finite domain.',
              'search equivalence, solver efficiency, native paired evidence and uncertainty.',A3)
    export(fig,output,'03-verification-algorithm')


def equation_page(pdf,output,name,heading,sections):
    fig=plt.figure(figsize=(8.27,11.69))
    fig.text(.08,.95,heading,fontsize=14,weight='bold')
    step=.855/len(sections)
    for i,(title,paragraph,formula,label) in enumerate(sections):
        y=.897-i*step
        fig.text(.08,y,title,fontsize=10,weight='bold')
        lines=textwrap.wrap(paragraph,100)
        fig.text(.08,y-.026,'\n'.join(lines),fontsize=8.2,va='top',linespacing=1.25)
        fig.text(.1,y-.089,formula,fontsize=11.2,va='center')
        fig.text(.94,y-.089,label,fontsize=9,ha='right',va='center')
    fig.text(.08,.035,'Current implementation: static search and conditional online policy are separate methods.',fontsize=8)
    pdf.savefig(fig,bbox_inches='tight',pad_inches=.15)
    fig.savefig(output/f'{name}.png',dpi=180,bbox_inches='tight',pad_inches=.15)
    plt.close(fig)


def equations(output):
    with PdfPages(output/'04-mathematical-formulation.pdf') as pdf:
        equation_page(pdf,output,'04-model-page-1','I. System and six-term objective',[
            ('Deployment states','Each model selects zero or a measured positive replica state.',
             r'$\sum_{n\in\mathcal{N}_m}z_{mn}=1,\quad n_m=\sum_n n z_{mn},\quad z_{mn}\in\{0,1\}$','(1)'),
            ('Flow and executable paths','No mandatory generalist, specialist or mixed deployment. Quality is scored rather than thresholded.',
             r'$\sum_m r_{tm}=\lambda_t,\quad 0\leq r_{tm}\leq\lambda_t a_{tm}(1-z_{m0})$','(2)'),
            ('Utility','All six terms are costs. Current weights are 1/6; positive gain means a positive difference in utility.',
             r'$U(c)=-\sum_{j\in\{Q,T,L,P,M,E\}}w_j\ell_j(c),\quad \sum_j w_j=1$','(4)'),
            ('Quality loss','Task loss is 1-EM (QA), 1-accuracy (image), CER (OCR), or WER (ASR).',
             r'$Q(c)=\sum_t\alpha_tq_t(c),\quad\alpha_t=\lambda_t/\sum_u\lambda_u$','(5)'),
            ('Window completion and latency','Use the registered cohort. P is a weighted per-task P95, not pooled P95.',
             r'$T=1-\frac{N_{win}}{N_{off}},\quad L=\sum_t\alpha_t\frac{\overline{d}_t}{b_t},\quad P=\sum_t\alpha_t\frac{P95_t}{b_t}$','(6)'),
            ('Memory and whole-board energy','Memory residency and observed peaks are separate. Post-window drain energy is a separate diagnostic.',
             r'$M=\frac{M_{safe}}{B},\quad M_{safe}\leq B,\qquad E=\frac{J_{run}/H+J_{once}/A}{P_{ref}}$','(7–8)')])
        equation_page(pdf,output,'04-model-page-2','II. Finite-domain joint optimization',[
            ('Capacity screening','Pool mean service at the selected replica state is a planning condition, not a physical stability proof.',
             r'$\rho_m=\frac{\sum_t r_{tm}\overline{s}^{pool}_{mt,n_m}}{1000n_m}\leq0.8$','(9)'),
            ('Static problem P1','Minimize frozen predicted six-term cost on the declared replica and rational route domain, with complete evidence.',
             r'$\min_{c\in\mathcal{D}_K}\widehat{C}(c)=-\widehat{U}(c),\quad c=(n,r)$','(11)'),
            ('Finite-trace FIFO','The runtime router and input order determine the concrete replica and service sample.',
             r'$S_i=\max(a_i,F_{prev(q_i)}),\quad F_i=S_i+\widehat{s}_i(c),\quad\widehat{d}_i=F_i-a_i$','(12)'),
            ('MILP lower-bound master','Use validated state and route contributions; full candidate evaluation still computes every objective term.',
             r'$\min_{z,y,\theta}\theta,\quad\theta\geq\mathcal{B}(z,y),\quad\sum_c y_{tc}=1$','(13–14)'),
            ('Residual identity feedback','Delta is evaluated cost minus its lower bound; h counts differing selected identity blocks.',
             r'$\theta\geq\mathcal{B}(v)+\Delta_j[1-h_j(v)],\quad\Delta_j=C_j-\mathcal{B}(v^j)\geq0$','(15)'),
            ('Bounds and independent DFS','Compare identical finite domains and the same evaluator; tied optimal configurations need not be identical.',
             r'$LB\leq C^*\leq C_{best},\quad gap=C_{best}-LB,\quad|\widehat{U}_{MILP}^*-\widehat{U}_{DFS}^*|\leq\epsilon$','(17,22)')])
        equation_page(pdf,output,'04-model-page-3','III. Shared-GPU routing and verification',[
            ('Shared progress model','A destination changes the activity context. Reprice every active head at each event.',
             r'$\delta=\min_{j\in\mathcal{A}}u_j\widehat{s}_j,\qquad u_j^{new}=\max(0,u_j-\delta/\widehat{s}_j)$','(18)'),
            ('Conditional prefix cost','Identical visible cohort and deadline for all destinations. Future arrivals and physical memory difference are unknown.',
             r'$\widehat{C}_{\tau}^{\omega}(a)=\sum_{j\in\{Q,T,L,P,E\}}w_j\widehat{\ell}_{j,\tau}^{\omega}(a)$','(19)'),
            ('Spillover decision','Compare each generalist to the selected specialist in every scenario; also enforce evidence and execution gates.',
             r'$\Delta_i(a)=\max_{\omega}[\widehat{C}_{\tau}^{\omega}(a)-\widehat{C}_{\tau}^{\omega}(a_S)]+w_M\overline{\Delta M}<-\eta$','(20–21)'),
            ('Strongest competing plans','Re-optimize a different deployment or a different route. A solver-history runner-up is not sufficient.',
             r'$c_{dep}=\arg\max_{c:n\ne n^*}\widehat{U}(c),\quad c_{route}=\arg\max_{c:n=n^*,r\ne r^*}\widehat{U}(c)$','(23)'),
            ('Matched native spillover ablation','Measure every component from the actual run. Unknown or invalid runs cannot be assigned zero utility.',
             r'$\Delta U_{\xi}=U_{spill,\xi}-U_{no\ spill,\xi}=\sum_jw_j(\ell_{j,no\ spill,\xi}-\ell_{j,spill,\xi})$','(25)'),
            ('Evidence scope','Finite-model optimality, CPU efficiency and measured physical gains are three separate claims.',
             r'$\mathrm{Model\ certificate}\ \ne\ \mathrm{physical\ all\!\! -\!\! combination\ optimum}$','')])


LATEX=r'''\documentclass[10pt,twocolumn]{article}
\usepackage[margin=18mm]{geometry}
\usepackage{amsmath,amssymb,algorithm,algpseudocode}
\newcommand{\DK}{\mathcal D_K}
\begin{document}
\title{Joint Generalist--Specialist Deployment, Routing, and Verification}
\author{}\date{}\maketitle
\section{System model}
Let $\mathcal T$ and $\mathcal M$ be the task and model sets, $\lambda_t$ the
fixed task rate, and $\alpha_t=\lambda_t/\sum_u\lambda_u$.
The replica domain $\mathcal N_m$ includes zero and measured positive counts.
\begin{align}
\sum_{n\in\mathcal N_m}z_{mn}&=1,&n_m&=\sum_n nz_{mn},\\
\sum_mr_{tm}&=\lambda_t,&0\le r_{tm}&\le\lambda_t a_{tm}(1-z_{m0}).
\end{align}
Here $a_{tm}$ denotes an executable path, not a quality cutoff. Route proportions
belong to the union of integer simplices with denominators $1,\ldots,K$; $K$ is
not the number of receivers. Replicas of a family receive equal rate, realized
by the shared runtime weighted-round-robin router.
\section{Six-term cost}
\begin{equation}
U(c)=-C(c)=-\sum_{j\in\{Q,T,L,P,M,E\}}w_j\ell_j(c),\quad w_j=1/6.
\end{equation}
\begin{align}
Q&=\sum_t\alpha_t q_t,& T&=1-N_{\rm win}/N_{\rm off},\\
L&=\sum_t\alpha_t\overline d_t/b_t,&P&=\sum_t\alpha_t\operatorname{P95}_t/b_t,\\
M&=M_{\rm safe}/B,&E&=\frac{J_{\rm run}/H+J_{\rm once}/A}{P_{\rm ref}}.
\end{align}
Task quality loss is $1-\mathrm{EM}$, $1-\mathrm{accuracy}$, CER, or WER.
Edit rates use accumulated errors divided by accumulated reference units and
are not clipped. Memory prediction and measured residency are distinct;
observed peaks are also checked against the hard budget. Energy is integrated
once for the whole board, and post-window drain energy is separately diagnosed.
\begin{equation}
\rho_m=\frac{\sum_t r_{tm}\overline s^{\rm pool}_{mt,n_m}}{1000n_m}\le0.8.
\end{equation}
This planning condition does not establish physical stability. Native runs
must separately pass the unchanged stability, completion and evidence gates.
\section{Finite-domain joint problem}
\begin{equation}
\mathrm{P1}:\ \min_{c\in\DK}\widehat C(c)=-\widehat U(c)
\end{equation}
subject to flow, executable paths, deployment linkage, memory, capacity,
bounded predicted completion and complete evidence. Hats denote frozen
predictions. The runtime-consistent FIFO evaluator uses
\begin{equation}
S_i=\max(a_i,F_{\operatorname{prev}(q_i)}),\quad F_i=S_i+\widehat s_i(c).
\end{equation}
Static v2 replays per-input service medians and a calibrated multiplier;
it is not a dynamic kernel-overlap predictor.
For task-route columns $c\in\mathcal C_t$, $\sum_cy_{tc}=1$ and
$y_{tc}\le z_{m,n_{tc,m}}$ for each used model. The master is
\begin{equation}
\min_{z,y,\theta}\theta,\quad\theta\ge\mathcal B(z,y),
\end{equation}
where the valid lower bound combines state and route contributions.
Given evaluated identity $v^j$, selected blocks $I_j$ and $b=|I_j|$, let
$h_j(v)=b-\sum_{i\in I_j}v_i$ and
$\Delta_j=C_j-\mathcal B(v^j)\ge0$. Residual feedback is
\begin{equation}
\theta\ge\mathcal B(v)+\Delta_j[1-h_j(v)].
\end{equation}
An invalid identity receives $\sum_{i\in I_j}v_i\le b-1$.
Unknown evidence retains its optimistic bound. With global lower cost bound
$LB$, $LB\le C^*\le C_{\rm best}$ and $\mathrm{gap}=C_{\rm best}-LB$.
\begin{algorithm}[t]
\caption{Joint deployment--routing search}
\begin{algorithmic}[1]
\Require Frozen model/replica pool, workload, trace, predictor, $K,\epsilon$, budget
\Ensure Best $(n,r)$, predicted utility, certificate or gap
\State Build frozen domain and task-route columns
\State Validate $\mathcal B$; initialize MP and $C_{\rm best}\gets+\infty$
\While{budgets remain}
 \State Solve MP for candidate $v$ and valid bound
 \If{gap $\le\epsilon$ and evidence resolved}
  \State \Return incumbent and finite-domain certificate
 \EndIf
 \State Decode $v$; run complete frozen evaluation
 \If{evidence unknown}
  \State Retain optimistic bound; move identity out of MP
 \ElsIf{infeasible or predicted drain fails}
  \State Add identity no-good constraint
 \Else
  \State Compute six-term cost; update incumbent
  \State Check bound validity; add residual identity cut
 \EndIf
\EndWhile
\State \Return incumbent, gap, unknown list and budget status
\end{algorithmic}
\end{algorithm}
\section{Conditional shared-GPU spillover}
This is a separate implemented policy at a fixed deployment, not an already
validated joint extension of P1. Visible history excludes future arrivals,
answers and uncompleted true service. For active requests with remaining
work $u_j$, the shared event predictor uses
\begin{equation}
\delta=\min_j u_j\widehat s_j,\quad u_j'=[u_j-\delta/\widehat s_j]^+.
\end{equation}
Every active head and whole-board power is repriced after an activity change.
All destination alternatives use the identical observed cohort and deadline:
\begin{equation}
\widehat C_\tau^\omega(a)=\sum_{j\in\{Q,T,L,P,E\}}w_j\widehat\ell_{j,\tau}^\omega(a).
\end{equation}
Past/startup energy is common to current alternatives. Physical workspace
memory difference remains unknown; a zero resident proxy difference is not
an invariance proof. Let
\begin{align}
a_S&=\arg\min_{a\in\mathcal S_i}\max_\omega\widehat C_\tau^\omega(a),\\
\Delta_i(a)&=\max_\omega[\widehat C_\tau^\omega(a)-\widehat C_\tau^\omega(a_S)]
+w_M\overline{\Delta M}.
\end{align}
Accept a generalist only if $\Delta_i(a)<-\eta$ and evidence, exposure,
compute-budget and cooldown gates pass. An absent memory bound omits the
extra penalty in current code, without certifying physical six-term gain.
\begin{algorithm}[t]
\caption{Conditional shared-GPU spillover}
\begin{algorithmic}[1]
\Require Current request, visible queues/history, resident set, service/power model
\Ensure Destination and decision audit
\State Process returned feedback; read full queue snapshot
\If{forecast budget exceeded}
 \State \Return deterministic specialist fallback
\EndIf
\ForAll{supported resident destinations $a$}
 \State Append current request to a copy of queue $a$
 \ForAll{$\omega\in\{\mathrm{mean,p90}\}$}
  \State Replay all observed requests through shared events
  \State Reprice every active head; integrate board energy
  \State Compute conditional prefix cost
 \EndFor
 \State Mark $a$ unknown if required evidence is missing
\EndFor
\State Select best known specialist baseline $a_S$
\State Test all-scenario improvement and execution gates
\State \Return accepted generalist, or specialist baseline/fallback
\end{algorithmic}
\end{algorithm}
\section{Verification}
The independent plain and bounded DFS use the same domain and evaluator:
\begin{equation}
|\widehat U^*_{\rm MILP}-\widehat U^*_{\rm DFS}|\le\epsilon.
\end{equation}
Compare ties, certificate/gap, wall time and complete evaluation count.
Strong competitors are obtained by re-optimizing with a different deployment,
a different route at the winning deployment, or no task splitting. Pure
structures are additional controls. Unfinished restricted searches remain
budget candidates.
For matched native spill/no-spill runs with the same residency:
\begin{equation}
\Delta U_\xi=\sum_jw_j(\ell_{j,\mathrm{no\ spill},\xi}-\ell_{j,\mathrm{spill},\xi}).
\end{equation}
\begin{algorithm}[t]
\caption{Search and hardware verification}
\begin{algorithmic}[1]
\Require Frozen contract, calibration/holdout/confirmation sets and budgets
\Ensure Search correctness, efficiency and native paired evidence
\State Profile standalone replicas and selected co-location conditions
\State Validate predictor on held-out data; retain failed versions
\State Freeze domain, source and predictor; run Algorithm 1
\State Independently run plain/bounded DFS and compare optima/ties
\State Measure wall time and full evaluations under equal cache rules
\State Re-optimize strongest alternative deployment/route/no-split plans
\State Deduplicate; run matched native GPU controls
\State Compare Algorithm 2 with same-residency no-spill and old policy
\State Audit raw cohort, output quality, timings, memory and energy
\State Apply unchanged stability and recompute all six terms
\State Run reserved confirmation only if the declared gate passes
\State Report model certificate, efficiency and physical evidence separately
\end{algorithmic}
\end{algorithm}
\section{Scope relative to J3O}
J3O uses alternating greedy Lagrangian onloading and LP offloading.
Our implemented static method uses a finite-domain joint MILP with validated
predictor feedback. Its certificate is a frozen-model finite-domain result,
not an inherited submodular approximation guarantee or a physical global
optimum. Native resource partitioning and kernel-sequence prediction are
prospective extensions requiring separate hardware evidence and new bounds.
\end{document}
'''


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,
        default=ROOT/'artifacts/j3o-style-method-20261026-v3');args=parser.parse_args()
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=False)
    flowchart(output);algorithms(output);equations(output)
    (output/'method.tex').write_text(LATEX)
    shutil.copyfile(MODEL,output/'method.zh.md')
    (output/'captions.zh.md').write_text('''图1：当前静态联合部署搜索及其验证（左），与独立的共享 GPU 在线分流实验（右）。两条研究链分别报告有限预测域最优与原生分流收益；虚线框为待验证资源扩展。\n\n图2：按 J3O Algorithm 环境风格排版的本项目算法1、算法2。内容来自当前实现，没有套用 J3O 的贪心拉格朗日与 LP 算法。\n\n图3：同域 DFS 与原生配对的完整验证过程。\n\n公式页：便于直接查看的英文数学摘要；完整中文变量、适用条件、公式 (1)–(25) 与算法说明见 method.zh.md。method.tex 是独立 LaTeX 稿件源文件，当前环境未配置 TeX 编译器；已导出的公式与算法 PDF 由 Python/Matplotlib 生成。\n''')
    sources=[MODEL,Path(__file__).resolve(),ROOT/'gsdeploy/milp.py',ROOT/'gsdeploy/prediction.py',
             ROOT/'gsdeploy/quality.py',ROOT/'gsdeploy/energy.py',ROOT/'gsdeploy/calibration_v2.py',
             ROOT/'tools/queue_coupled_utility_20261023/policy.py',ROOT/'tools/queue_coupled_utility_20261023/model.py']
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    receipt={'gpu_started':False,'solvers_started':False,'experiment_data_modified':False,
             'sources':{str(p):sha(p) for p in sources},
             'outputs':{p.name:sha(p) for p in sorted(output.iterdir()) if p.is_file()},
             'scope':'Method formalization and illustration, not a new experiment or optimality result.'}
    (output/'manifest.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
    print(output)


if __name__=='__main__':main()
