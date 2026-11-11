import sys,csv,json,time
from pathlib import Path
sys.path.insert(0,r'D:\vscode\第三周实训')
import numpy as np,pandas as pd
from boltcrypt_wrapper import make_env
from wall_penalty_v2 import WallPenalty
from mechanics_audit import MechanicsAudit
from full_dungeon_guide import FullDungeonGuide
root=Path(r'D:\vscode\第三周实训');out=Path(__file__).resolve().parent/'rule_contribution';out.mkdir(exist_ok=True)
cfg=json.loads((root/'experiment_config_v13.json').read_text(encoding='utf-8-sig'))
def place(o):return tuple(map(int,o['global_pos'])),tuple(map(int,o['agent_pos']))
rows=[];episodes=[]
for runseed in (41,42,43):
 for mapseed in range(70000,70008):
  e=MechanicsAudit(WallPenalty(make_env(cfg['generator_config'],1000),cfg['wall_collision_reward']))
  obs,_=e.reset(seed=mapseed);guide=FullDungeonGuide();rng=np.random.default_rng(100000+mapseed+runseed);start=time.perf_counter()
  startroom=place(obs)[0];rooms={startroom};keys=puzzles=walls=fall=rules=0;first_room=first_key=first_puzzle=None
  try:
   for step in range(1,1001):
    before=obs;was_solved=bool(e.unwrapped.curr_room.is_solved)
    action,reason,target=guide.act(obs,None)
    if action is None:
     action=int(rng.integers(0,4));reason='uniform_random_fallback';fall+=1
    else:rules+=1
    obs,reward,term,trunc,info=e.step(int(action));guide.observe_transition(before,int(action),obs)
    event=info['mechanics']['event'];room=place(obs)[0];rooms.add(room);walls+=int(info['wall_penalty']['confirmed_collision'])
    if first_room is None and room!=startroom:first_room=step
    if event=='key_acquired':keys+=1;first_key=first_key or step
    if room==place(before)[0] and not was_solved and e.unwrapped.curr_room.is_solved and e.unwrapped.curr_room.puzzle_type.value=='boulder':puzzles+=1;first_puzzle=first_puzzle or step
    rows.append({'run_seed':runseed,'map_seed':mapseed,'step':step,'source':'rule' if reason!='uniform_random_fallback' else 'random_fallback','reason':reason,'action':int(action),'reward':reward,'event':event,'room_before':str(place(before)[0]),'room_after':str(room),'pos_before':str(place(before)[1]),'pos_after':str(place(obs)[1])})
    if term or trunc:break
  finally:e.close()
  episodes.append({'run_seed':runseed,'map_seed':mapseed,'steps':step,'success':int(term),'first_room_step':first_room or '', 'first_key_step':first_key or '', 'first_puzzle_step':first_puzzle or '', 'rooms_visited':len(rooms),'keys_acquired':keys,'puzzles_completed':puzzles,'wall_collisions':walls,'rule_actions':rules,'random_fallback_actions':fall,'evaluation_seconds':time.perf_counter()-start})
  print(runseed,mapseed,step,int(term),rules,fall,flush=True)
for name,data in [('pure_rule_steps.csv',rows),('pure_rule_episodes.csv',episodes)]:
 with (out/name).open('w',newline='',encoding='utf-8-sig') as f:
  w=csv.DictWriter(f,fieldnames=data[0].keys());w.writeheader();w.writerows(data)

detail=pd.read_csv(root/'reports/full_v17_decision_audit.csv').query("group in ['C','D']").copy()
ep=pd.read_csv(root/'reports/full_v17_episodes.csv').query("group in ['A','B','C','D']").copy()
detail['changed']=detail.ppo_action!=detail.submitted_action
bursts=[]
for (g,seed,mapseed),part in detail.groupby(['group','train_seed','map_seed']):
 part=part.sort_values('step');last=None;start=None;length=0;reason=None
 for row in part.itertuples():
  if last is None or row.step!=last+1:
   if length:bursts.append({'group':g,'train_seed':seed,'map_seed':mapseed,'start_step':start,'length':length,'trigger':reason})
   start=row.step;length=0;reason=row.trigger
  length+=1;last=row.step
 if length:bursts.append({'group':g,'train_seed':seed,'map_seed':mapseed,'start_step':start,'length':length,'trigger':reason})
pd.DataFrame(bursts).to_csv(out/'rule_bursts.csv',index=False,encoding='utf-8-sig')
summary=[]
for g in ('C','D'):
 q=detail[detail.group==g];e=ep[ep.group==g];b=pd.DataFrame(bursts).query('group==@g')
 summary.append({'group':g,'episodes':len(e),'successes':int(e.success.sum()),'total_steps':int(e.steps.sum()),'rule_actions':len(q),'rule_action_share':len(q)/e.steps.sum(),'changed_actions':int(q.changed.sum()),'changed_share_of_rule_actions':q.changed.mean(),'bursts':len(b),'mean_burst_length':b.length.mean(),'max_burst_length':int(b.length.max()),'episodes_at_200_rule_actions':int((e.rule_actions>=200).sum()),'accepted_rule_moves':int(q.move_accepted.sum()),'rejected_doors':int((q.event=='door_rejected_unknown_cause').sum())})
pd.DataFrame(summary).to_csv(out/'hybrid_summary.csv',index=False,encoding='utf-8-sig')
