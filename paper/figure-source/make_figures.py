#!/usr/bin/env python3
"""Rebuild IWIBench development figures from the six released episode CSVs."""
from pathlib import Path
import hashlib,json,os,tempfile
os.environ.setdefault("MPLCONFIGDIR",str(Path(tempfile.gettempdir())/"iwibench-matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME",str(Path(tempfile.gettempdir())/"iwibench-cache"))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'figures'; OUT.mkdir(exist_ok=True)
EDITORS={'Terra 6.1':'','Sol 6.1':'-sol','Astra':'-astra','GLM 5.3':'-glm','DeepSeek V4.1 Flash':'-deepseek','Nemotron 3 Ultra':'-nemotron'}
FAMILIES={'alias_acquisition':'Alias acquisition','output_code_remapping':'Output-code remapping','lexical_inhibition':'Lexical inhibition','terminology_preference':'Terminology preference','default_response_language':'Default response language','structured_response_default':'Structured response default','answer_length_control':'Answer-length control','factual_update':'Local factual update','selective_factual_withholding':'Selective factual withholding','resistance_to_false_suggestions':'Resistance to false suggestions'}
SHORT=['Alias\nacquisition','Output-code\nremapping','Lexical\ninhibition','Terminology\npreference','Default response\nlanguage','Structured response\ndefault','Answer-length\ncontrol','Local factual\nupdate','Selective factual\nwithholding','Resistance to\nfalse suggestions']
COLORS=dict(zip(EDITORS,['#0072B2','#D55E00','#009E73','#CC79A7','#E69F00','#6F6F6F']))
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','savefig.facecolor':'white'})
frames=[]; manifest=[]; keys=None
for editor,suffix in EDITORS.items():
 p=ROOT/'data'/f'results__ten-families-v022{suffix}__gemma-3-4b-it-prime.csv'
 d=pd.read_csv(p); assert len(d)==30 and not d.duplicated(['family','parameter_seed']).any()
 assert set(d.family)==set(FAMILIES) and (d.groupby('family').size()==3).all()
 assert set(d.family_version)=={'0.2.2'} and set(d.model_id)=={'google/gemma-3-4b-it'}
 k=d.set_index(['family','parameter_seed'])[['target_base_correct','target_count','control_count','revision']].sort_index()
 if keys is None: keys=k
 else: pd.testing.assert_frame_equal(k,keys)
 gain=((d.target_edited_correct-d.target_base_correct)/(d.target_count-d.target_base_correct)).clip(lower=0)
 np.testing.assert_allclose(gain,d.gain,atol=1e-12)
 np.testing.assert_allclose(d.controls_retained/d.control_count,d.control_retention,atol=1e-12)
 np.testing.assert_allclose(gain*d.control_retention,d.reward,atol=1e-12)
 assert ((d.reward>=0)&(d.reward<=1)).all()
 d['editor']=editor;frames.append(d)
 manifest.append({'path':str(p.relative_to(ROOT)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'episodes':len(d)})
data=pd.concat(frames,ignore_index=True)
family=data.groupby(['family','editor']).reward.mean().unstack('editor').reindex(index=FAMILIES,columns=EDITORS)
summary=data.groupby('editor').agg(target_base_correct=('target_base_correct','sum'),target_edited_correct=('target_edited_correct','sum'),target_count=('target_count','sum'),controls_retained=('controls_retained','sum'),control_count=('control_count','sum'),forward_tokens=('forward_tokens','sum')).reindex(EDITORS)
summary['macro_reward']=family.mean();summary['signed_target_change']=summary.target_edited_correct-summary.target_base_correct
summary['pooled_control_retention']=summary.controls_retained/summary.control_count
summary=summary.sort_values('macro_reward',ascending=False)
family.rename(index=FAMILIES).to_csv(ROOT/'family_rewards.csv');summary.to_csv(ROOT/'editor_summary.csv')
(ROOT/'source_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
def save(fig,name):
 for extension in ['pdf','svg','png']: fig.savefig(OUT/f'{name}.{extension}',dpi=240,bbox_inches='tight',pad_inches=.12)
 plt.close(fig)
# Radar panels use identical axes. Splitting prevents the low-valued traces obscuring each other.
theta=np.arange(10)*2*np.pi/10;angles=np.r_[theta,theta[0]]
fig,axs=plt.subplots(1,2,figsize=(13.0,6.8),subplot_kw={'projection':'polar'})
for ax,group,title in zip(axs,[list(EDITORS)[:3],list(EDITORS)[3:]],['Terra, Sol and Astra','GLM, DeepSeek and Nemotron']):
 ax.set_theta_offset(np.pi/2);ax.set_theta_direction(-1);ax.set_ylim(0,1)
 ax.set_xticks(theta);ax.set_xticklabels(SHORT,fontsize=8.5);ax.tick_params(axis='x',pad=15)
 ax.set_yticks([.25,.5,.75,1]);ax.set_yticklabels(['0.25','0.50','0.75','1.00'],fontsize=8,color='#555555');ax.set_rlabel_position(18)
 ax.spines['polar'].set_color('#bbbbbb');ax.grid(color='#dedede',lw=.7)
 for i,editor in enumerate(group):
  values=family[editor].to_numpy();ax.plot(angles,np.r_[values,values[0]],label=editor,color=COLORS[editor],lw=1.8,linestyle=['-','--',':'][i],marker=['o','s','^'][i],markersize=3.5)
 ax.set_title(title,pad=35,fontsize=11,fontweight='bold')
 ax.legend(loc='upper center',bbox_to_anchor=(.5,-.20),frameon=False,fontsize=9)
fig.subplots_adjust(wspace=.55,bottom=.20,top=.85)
save(fig,'family_radar')
# Ranking, zero-based common scale; point estimates, no unsupported uncertainty claims.
fig,ax=plt.subplots(figsize=(7.5,3.7));names=list(summary.index);y=np.arange(6)
ax.barh(y,summary.macro_reward,color=[COLORS[n] for n in names],height=.63)
for i,v in enumerate(summary.macro_reward):ax.text(v+.014,i,f'{v:.4f}',va='center',fontsize=10)
ax.set_yticks(y,names);ax.invert_yaxis();ax.set_xlim(0,1);ax.set_xticks(np.arange(0,1.01,.2));ax.set_xlabel('Macro-average reward (equal weight per family)');ax.set_axisbelow(True);ax.grid(axis='x',color='#e5e5e5');ax.spines['left'].set_visible(False);ax.tick_params(axis='y',length=0)
fig.tight_layout();save(fig,'overall_ranking')
# Heatmap permits precise comparison without radar-area interpretation.
fig,ax=plt.subplots(figsize=(9.6,6.0));mat=family.loc[:,names].to_numpy();im=ax.imshow(mat,vmin=0,vmax=1,cmap='YlGnBu',aspect='auto')
ax.set_xticks(np.arange(6),names,rotation=25,ha='right');ax.set_yticks(np.arange(10),list(FAMILIES.values()))
for i in range(10):
 for j in range(6):ax.text(j,i,f'{mat[i,j]:.3f}',ha='center',va='center',fontsize=9,color='white' if mat[i,j]>.55 else '#222222')
ax.tick_params(length=0);fig.colorbar(im,ax=ax,label='Mean episode reward',fraction=.035,pad=.025);fig.tight_layout();save(fig,'family_heatmap')
# Two different quantities in separate panels, in the same editor order.
fig,axs=plt.subplots(1,2,figsize=(10,3.8),sharey=True,gridspec_kw={'width_ratios':[1.25,1]});y=np.arange(6)
axs[0].barh(y,summary.signed_target_change,color=[COLORS[n] for n in names],height=.63)
for i,v in enumerate(summary.signed_target_change):axs[0].text(v+(4 if v>=0 else -4),i,f'{v:+d}',ha='left' if v>=0 else 'right',va='center',fontsize=9)
axs[0].axvline(0,color='#888888',lw=.8);axs[0].set_xlim(-45,200);axs[0].set_yticks(y,names);axs[0].invert_yaxis();axs[0].set_xlabel('Change in target successes (out of 551)');axs[0].set_title('Signed target outcome',fontsize=11)
axs[1].barh(y,100*summary.pooled_control_retention,color=[COLORS[n] for n in names],height=.63)
for i,row in enumerate(summary.itertuples()):axs[1].text(103,i,f'{row.controls_retained:,}/{row.control_count:,}',va='center',fontsize=9)
axs[1].set_xlim(0,135);axs[1].set_xticks([0,25,50,75,100]);axs[1].set_xlabel('Baseline-correct controls retained (%)');axs[1].set_title('Pooled control retention',fontsize=11)
for ax in axs:ax.set_axisbelow(True);ax.grid(axis='x',color='#e5e5e5');ax.spines['left'].set_visible(False);ax.tick_params(axis='y',length=0)
fig.tight_layout();save(fig,'signed_outcomes_and_retention')
print(summary.to_string());print('VALIDATION PASSED: 180 rows, matched selected episodes, reward recomputed, source hashes recorded.')
