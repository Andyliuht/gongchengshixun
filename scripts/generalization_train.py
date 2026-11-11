import sys,json,time,random
from pathlib import Path
sys.path.insert(0,r'D:\vscode\第三周实训')
import gymnasium as gym,numpy as np
from stable_baselines3 import PPO
from boltcrypt_wrapper import make_env
from wall_penalty_v2 import WallPenalty
from cell_novelty_v9 import DiminishingCellNovelty
from revisit_penalty_v10 import IncreasingRevisitPenalty
from navigation_observation_v5 import NavigationObservation
from history_observation import HistoryObservation
from goal_observation_v14 import GoalObservation
root=Path(r'D:\vscode\第三周实训');out=Path(__file__).resolve().parent/'generalization_3room';out.mkdir(exist_ok=True)
cfg=json.loads((Path(__file__).resolve().parent/'three_room_corrected/protocol.json').read_text(encoding='utf-8'))
def kind(e):
 rooms=e.unwrapped.generator.grid
 return 'key' if any(r.is_locked for r in rooms.values()) else 'puzzle' if any(r.puzzle_type.value=='boulder' for r in rooms.values()) else 'plain'
catalog={k:[] for k in ('plain','key','puzzle')}
e=make_env(cfg['generator_config'],1000)
for seed in range(10000,11000):
 e.reset(seed=seed);catalog[kind(e)].append(seed)
e.close()
fixed={k:[catalog[k][0]] for k in catalog}
procedural={k:catalog[k][1:31] for k in catalog}
protocol={'generator_config':cfg['generator_config'],'fixed_train_maps':fixed,'procedural_train_maps':procedural,'test_maps':cfg['test_maps'],'test_types':cfg['test_types'],'model_seeds':cfg['model_seeds'],'steps':59904,'episode_limit':1000,'reward_parameters':cfg['reward_parameters'],'discovery_reward':cfg['discovery_reward'],'groups':['B','E'],'layout_conditions':['fixed','procedural'],'selection':'Each reset chooses one of plain/key/puzzle uniformly, then a map uniformly within that category.'}
(out/'protocol.json').write_text(json.dumps(protocol,ensure_ascii=False,indent=2),encoding='utf-8')
class BalancedMapCycle(gym.Wrapper):
 def __init__(self,env,run_seed,maps):
  super().__init__(env);self.rng=np.random.default_rng(run_seed);self.maps=maps;self.used=[]
 def reset(self,*,seed=None,options=None):
  k=('plain','key','puzzle')[int(self.rng.integers(0,3))]
  chosen=int(self.rng.choice(self.maps[k]));self.used.append({'seed':chosen,'type':k})
  return self.env.reset(seed=chosen,options=options)
def env_for(g):
 e=make_env(cfg['generator_config'],1000,cfg['discovery_reward'])
 e=WallPenalty(e,cfg['reward_parameters']['wall_collision_reward'])
 e=DiminishingCellNovelty(e,cfg['reward_parameters']['first_cell_bonus'])
 e=IncreasingRevisitPenalty(e,cfg['reward_parameters']['revisit_penalty_scale'],cfg['reward_parameters']['revisit_penalty_cap'])
 e=HistoryObservation(NavigationObservation(e))
 if g=='E':e=GoalObservation(e)
 return e
for g in ('B','E'):
 for condition,maps in (('fixed',fixed),('procedural',procedural)):
  for seed in cfg['model_seeds']:
   path=out/f'{g}_{condition}_{seed}';path.mkdir(exist_ok=True)
   if (path/'model.zip').exists() and (path/'summary.json').exists():print('reuse',g,condition,seed,flush=True);continue
   env=BalancedMapCycle(env_for(g),seed,maps);start=time.perf_counter()
   model=PPO('MultiInputPolicy',env,n_steps=256,batch_size=64,n_epochs=2,seed=seed,device='cpu',verbose=0)
   model.learn(total_timesteps=59904);model.save(path/'model')
   (path/'summary.json').write_text(json.dumps({'group':g,'condition':condition,'seed':seed,'steps':model.num_timesteps,'seconds':time.perf_counter()-start,'maps_used':env.used},indent=2),encoding='utf-8')
   print('trained',g,condition,seed,model.num_timesteps,round(time.perf_counter()-start,1),flush=True);env.close()
