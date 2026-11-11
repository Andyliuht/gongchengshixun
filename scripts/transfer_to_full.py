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
from limited_full_guide import LimitedFullGuide
root=Path(r'D:\vscode\第三周实训');small=Path(__file__).resolve().parent/'three_room_corrected';out=Path(__file__).resolve().parent/'transfer_full';out.mkdir(exist_ok=True)
smallcfg=json.loads((small/'protocol.json').read_text(encoding='utf-8'))
fullcfg=json.loads((root/'experiment_config_v13.json').read_text(encoding='utf-8-sig'))
cfg={'training':'fixed 3 rooms, 59904 steps','evaluation_generator':fullcfg['generator_config'],'test_maps':list(range(70000,70008)),'model_seeds':[41,42,43],'groups':['A','A2','B','C','D','E','F'],'max_episode_steps':1000,'sampling':'deterministic=False','reward_parameters':smallcfg['reward_parameters'],'discovery_reward':smallcfg['discovery_reward'],'guidance':smallcfg['guidance']}
(out/'protocol.json').write_text(json.dumps(cfg,ensure_ascii=False,indent=2),encoding='utf-8')
def env_for(g):
 e=make_env(cfg['evaluation_generator'],1000,cfg['discovery_reward'] if g in ('A2','B','D','E') else None)
 e=WallPenalty(e,cfg['reward_parameters']['wall_collision_reward'])
 if g in ('B','D','E'):
  e=DiminishingCellNovelty(e,cfg['reward_parameters']['first_cell_bonus']);e=IncreasingRevisitPenalty(e,cfg['reward_parameters']['revisit_penalty_scale'],cfg['reward_parameters']['revisit_penalty_cap'])
 e=HistoryObservation(NavigationObservation(e))
 if g in ('E','F'):e=GoalObservation(e)
 return MechanicsAudit(e)
def place(o):return tuple(map(int,o['global_pos'])),tuple(map(int,o['agent_pos']))
fields=['group','train_seed','map_seed','room_count','locked_map','puzzle_map','steps','success','first_room_step','first_exit_room_step','first_key_step','first_puzzle_step','rooms_visited','keys_acquired','puzzles_completed','wall_collisions','two_step_returns','rule_actions','overrides','evaluation_seconds']
path=out/'episodes.csv';existing=set()
if path.exists():
 with path.open(encoding='utf-8-sig',newline='') as f:existing={(r['group'],int(r['train_seed']),int(r['map_seed'])) for r in csv.DictReader(f)}
with path.open('a',encoding='utf-8-sig',newline='') as f:
 w=csv.DictWriter(f,fieldnames=fields)
 if not existing:w.writeheader()
 for g in cfg['groups']:
  source='A' if g=='C' else 'B' if g=='D' else g
  for seed in cfg['model_seeds']:
   model=PPO.load(small/f'{source}_{seed}/model_59904.zip',device='cpu')
   for mapseed in cfg['test_maps']:
    if (g,seed,mapseed) in existing:continue
    e=env_for(g);obs,_=e.reset(seed=mapseed);torch.manual_seed(100000+mapseed+seed)
    rooms_map=e.unwrapped.generator.grid;target=tuple(map(int,e.unwrapped.generator.end_pos));locked=int(any(r.is_locked for r in rooms_map.values()));puzzle=int(any(r.puzzle_type.value=='boulder' for r in rooms_map.values()))
    guide=LimitedFullGuide(cfg['guidance'],obs) if g in ('C','D') else None
    startroom=place(obs)[0];visited={startroom};first_room=first_exit=first_key=first_puzzle=None;keys=puzzles=walls=loops=rules=overrides=0;positions=[place(obs)];started=time.perf_counter()
    for step in range(1,1001):
     before=obs;oldroom=place(before)[0];was_solved=bool(e.unwrapped.curr_room.is_solved)
     raw=int(model.predict(obs,deterministic=False)[0]);action=raw;engaged=False
     if guide:action,reason,goal,engaged,trigger=guide.decide(obs,raw)
     rules+=int(engaged);overrides+=int(engaged and action!=raw)
     obs,reward,terminated,truncated,info=e.step(action)
     if guide:guide.observe_transition(before,action,obs,info['mechanics'])
     now=place(obs);visited.add(now[0]);walls+=int(info['wall_penalty']['confirmed_collision'])
     if len(positions)>=2 and now==positions[-2] and now!=positions[-1]:loops+=1
     positions.append(now)
     if first_room is None and now[0]!=startroom:first_room=step
     if first_exit is None and now[0]==target:first_exit=step
     if info['mechanics']['event']=='key_acquired':keys+=1;first_key=first_key or step
     if now[0]==oldroom and not was_solved and e.unwrapped.curr_room.is_solved and e.unwrapped.curr_room.puzzle_type.value=='boulder':puzzles+=1;first_puzzle=first_puzzle or step
     if terminated or truncated:break
    w.writerow({'group':g,'train_seed':seed,'map_seed':mapseed,'room_count':len(rooms_map),'locked_map':locked,'puzzle_map':puzzle,'steps':step,'success':int(terminated),'first_room_step':first_room or '', 'first_exit_room_step':first_exit or '', 'first_key_step':first_key or '', 'first_puzzle_step':first_puzzle or '', 'rooms_visited':len(visited),'keys_acquired':keys,'puzzles_completed':puzzles,'wall_collisions':walls,'two_step_returns':loops,'rule_actions':rules,'overrides':overrides,'evaluation_seconds':time.perf_counter()-started});f.flush();e.close()
   print('evaluated',g,seed,flush=True)
