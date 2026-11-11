import sys,json,time
from pathlib import Path
sys.path.insert(0,r'D:\vscode\第三周实训')
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from boltcrypt_wrapper import make_env
from wall_penalty_v2 import WallPenalty
from cell_novelty_v9 import DiminishingCellNovelty
from revisit_penalty_v10 import IncreasingRevisitPenalty
from navigation_observation_v5 import NavigationObservation
from history_observation import HistoryObservation
from goal_observation_v14 import GoalObservation
from train_history_ablation import MapCycle
root=Path(r'D:\vscode\第三周实训');out=Path(__file__).resolve().parent/'three_room_corrected';out.mkdir(exist_ok=True)
base=json.loads((root/'experiment_config_v13.json').read_text(encoding='utf-8-sig'))
cfg={'generator_config':{**base['generator_config'],'min_dist':1,'max_dist':2,'mean_rooms':3,'std_rooms':0,'puzzle_density':0.35,'key_puzzle_prob':0.5},'train_seeds':{'start':10000,'count':200},'test_maps':[70000,70001,70002,70003,70005,70007,70012,70022],'test_types':{'plain':[70000,70001],'key':[70002,70003,70005],'puzzle':[70007,70012,70022]},'model_seeds':[41,42,43],'checkpoints':[59904,99840],'episode_limit':1000,'groups':['A','A2','B','C','D','E'],'reward_parameters':{k:base[k] for k in ('wall_collision_reward','first_cell_bonus','revisit_penalty_scale','revisit_penalty_cap')},'discovery_reward':base['exploration']['discovery_reward'],'guidance':base['guidance']}
(out/'protocol.json').write_text(json.dumps(cfg,ensure_ascii=False,indent=2),encoding='utf-8')
class SaveAt(BaseCallback):
 def __init__(self,path):super().__init__();self.path=path
 def _on_step(self):
  if self.num_timesteps==59904:self.model.save(self.path/'model_59904')
  return True
def make(g):
 e=make_env(cfg['generator_config'],1000,cfg['discovery_reward'] if g in ('A2','B','D','E') else None)
 e=WallPenalty(e,cfg['reward_parameters']['wall_collision_reward'])
 if g in ('B','D','E'):
  e=DiminishingCellNovelty(e,cfg['reward_parameters']['first_cell_bonus']);e=IncreasingRevisitPenalty(e,cfg['reward_parameters']['revisit_penalty_scale'],cfg['reward_parameters']['revisit_penalty_cap'])
 e=HistoryObservation(NavigationObservation(e))
 if g=='E':e=GoalObservation(e)
 return e
for g in ('A','A2','B','E'):
 for seed in cfg['model_seeds']:
  p=out/f'{g}_{seed}';p.mkdir(exist_ok=True)
  if (p/'model_99840.zip').exists() and (p/'model_59904.zip').exists():print('reuse',g,seed,flush=True);continue
  e=MapCycle(make(g),seed,cfg['train_seeds']);start=time.perf_counter()
  m=PPO('MultiInputPolicy',e,n_steps=256,batch_size=64,n_epochs=2,seed=seed,device='cpu',verbose=0)
  m.learn(total_timesteps=99840,callback=SaveAt(p));m.save(p/'model_99840')
  (p/'summary.json').write_text(json.dumps({'group':g,'seed':seed,'steps':m.num_timesteps,'seconds':time.perf_counter()-start,'map_seeds_used':e.used_seeds},indent=2),encoding='utf-8')
  print('trained',g,seed,m.num_timesteps,round(time.perf_counter()-start,1),flush=True);e.close()
