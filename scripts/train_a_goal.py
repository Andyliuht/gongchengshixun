import sys,json,time
from pathlib import Path
sys.path.insert(0,r'D:\vscode\第三周实训')
from stable_baselines3 import PPO
from boltcrypt_wrapper import make_env
from wall_penalty_v2 import WallPenalty
from navigation_observation_v5 import NavigationObservation
from history_observation import HistoryObservation
from goal_observation_v14 import GoalObservation
from train_history_ablation import MapCycle
out=Path(__file__).resolve().parent/'three_room_corrected';cfg=json.loads((out/'protocol.json').read_text(encoding='utf-8'))
for seed in cfg['model_seeds']:
 p=out/f'F_{seed}';p.mkdir(exist_ok=True)
 if (p/'model_59904.zip').exists():print('reuse',seed,flush=True);continue
 e=make_env(cfg['generator_config'],1000)
 e=WallPenalty(e,cfg['reward_parameters']['wall_collision_reward'])
 e=MapCycle(GoalObservation(HistoryObservation(NavigationObservation(e))),seed,cfg['train_seeds'])
 start=time.perf_counter();m=PPO('MultiInputPolicy',e,n_steps=256,batch_size=64,n_epochs=2,seed=seed,device='cpu',verbose=0)
 m.learn(total_timesteps=59904);m.save(p/'model_59904')
 (p/'summary.json').write_text(json.dumps({'group':'F=A+goal','seed':seed,'steps':m.num_timesteps,'seconds':time.perf_counter()-start,'map_seeds_used':e.used_seeds},indent=2),encoding='utf-8')
 print('trained F',seed,m.num_timesteps,round(time.perf_counter()-start,1),flush=True);e.close()
