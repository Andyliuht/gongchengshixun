import sys,json,csv,time
from pathlib import Path
sys.path.insert(0,r'D:\vscode\第三周实训')
import torch
from stable_baselines3 import PPO
from boltcrypt_wrapper import make_env
from wall_penalty_v2 import WallPenalty
from cell_novelty_v9 import DiminishingCellNovelty
from revisit_penalty_v10 import IncreasingRevisitPenalty
from navigation_observation_v5 import NavigationObservation
from history_observation import HistoryObservation
from goal_observation_v14 import GoalObservation
from mechanics_audit import MechanicsAudit
out=Path(__file__).resolve().parent/'generalization_3room';cfg=json.loads((out/'protocol.json').read_text(encoding='utf-8'))
def env_for(g):
 e=make_env(cfg['generator_config'],1000,cfg['discovery_reward'])
 e=WallPenalty(e,cfg['reward_parameters']['wall_collision_reward'])
 e=DiminishingCellNovelty(e,cfg['reward_parameters']['first_cell_bonus'])
 e=IncreasingRevisitPenalty(e,cfg['reward_parameters']['revisit_penalty_scale'],cfg['reward_parameters']['revisit_penalty_cap'])
 e=HistoryObservation(NavigationObservation(e))
 if g=='E':e=GoalObservation(e)
 return MechanicsAudit(e)
def place(o):return tuple(map(int,o['global_pos'])),tuple(map(int,o['agent_pos']))
fields=['group','condition','train_seed','split','map_seed','map_type','room_count','steps','success','first_room_step','first_exit_room_step','first_key_step','first_puzzle_step','rooms_visited','wall_collisions','two_step_returns','evaluation_seconds']
path=out/'episodes.csv';existing=set()
if path.exists():
 with path.open(encoding='utf-8-sig',newline='') as f:existing={(r['group'],r['condition'],int(r['train_seed']),r['split'],int(r['map_seed'])) for r in csv.DictReader(f)}
with path.open('a',encoding='utf-8-sig',newline='') as f:
 w=csv.DictWriter(f,fieldnames=fields)
 if not existing:w.writeheader()
 for g in cfg['groups']:
  for condition in cfg['layout_conditions']:
   for seed in cfg['model_seeds']:
    model=PPO.load(out/f'{g}_{condition}_{seed}/model.zip',device='cpu')
    cases=[]
    for typ,items in cfg['fixed_train_maps'].items():cases.append(('fixed_seen',items[0],typ))
    for typ,items in cfg['procedural_train_maps'].items():cases.append(('procedural_seen',items[0],typ))
    for typ,items in cfg['test_types'].items():cases.extend(('heldout',m,typ) for m in items)
    for split,mapseed,typ in cases:
     if (g,condition,seed,split,mapseed) in existing:continue
     e=env_for(g);obs,_=e.reset(seed=mapseed);torch.manual_seed(100000+mapseed+seed)
     rooms=e.unwrapped.generator.grid;target=tuple(map(int,e.unwrapped.generator.end_pos));startroom=place(obs)[0];visited={startroom};first_room=first_exit=first_key=first_puzzle=None;walls=loops=0;positions=[place(obs)];start=time.perf_counter()
     for step in range(1,1001):
      before=obs;was_solved=bool(e.unwrapped.curr_room.is_solved);oldroom=place(before)[0]
      a=int(model.predict(obs,deterministic=False)[0]);obs,r,term,trunc,info=e.step(a)
      now=place(obs);visited.add(now[0]);walls+=int(info['wall_penalty']['confirmed_collision'])
      if len(positions)>=2 and now==positions[-2] and now!=positions[-1]:loops+=1
      positions.append(now)
      if first_room is None and now[0]!=startroom:first_room=step
      if first_exit is None and now[0]==target:first_exit=step
      if first_key is None and info['mechanics']['event']=='key_acquired':first_key=step
      if first_puzzle is None and now[0]==oldroom and not was_solved and e.unwrapped.curr_room.is_solved and e.unwrapped.curr_room.puzzle_type.value=='boulder':first_puzzle=step
      if term or trunc:break
     w.writerow({'group':g,'condition':condition,'train_seed':seed,'split':split,'map_seed':mapseed,'map_type':typ,'room_count':len(rooms),'steps':step,'success':int(term),'first_room_step':first_room or '', 'first_exit_room_step':first_exit or '', 'first_key_step':first_key or '', 'first_puzzle_step':first_puzzle or '', 'rooms_visited':len(visited),'wall_collisions':walls,'two_step_returns':loops,'evaluation_seconds':time.perf_counter()-start});f.flush();e.close()
    print('evaluated',g,condition,seed,flush=True)
