import argparse
import tot.traces as traces
traces.init_caller("tot", verbose = True, async_mode = True, use_cache = False)

from tot.methods.bfs import solve
from tot.tasks.game24 import Game24Task

args = argparse.Namespace(
    backend='gpt-3.5-turbo', 
    temperature=0.7, 
    task='game24', 
    naive_run=False, 
    prompt_sample='standard', 
    method_generate='sample', 
    method_evaluate='value', 
    method_select='greedy', 
    n_generate_sample=3, 
    n_evaluate_sample=1, 
    n_select_sample=5)

task = Game24Task()
ys, infos = solve(args, task, 1350) # 3, 3, 8, 8
print(ys[0])


