import itertools
import numpy as np
from functools import partial
from tot.models import gpt, gpt_async
from concurrent.futures import ThreadPoolExecutor
import asyncio

import logging

logging.basicConfig(filename='api_errors.log', level=logging.ERROR)


def get_value(task, x, y, n_evaluate_sample, cache_value=True):
    value_prompt = task.value_prompt_wrap(x, y)
    if cache_value and value_prompt in task.value_cache:
        return task.value_cache[value_prompt]
    value_outputs = gpt(value_prompt, n=n_evaluate_sample, stop=None)
    value = task.value_outputs_unwrap(x, y, value_outputs)
    if cache_value:
        task.value_cache[value_prompt] = value
    return value

async def get_value_async(task, x, y, n_evaluate_sample, cache_value=True):
    value_prompt = task.value_prompt_wrap(x, y)
    if cache_value and value_prompt in task.value_cache:
        return task.value_cache[value_prompt]
    try:
        value_outputs = await gpt_async(value_prompt, n=n_evaluate_sample, stop=None)
    except Exception as e:
        logging.error(f"gpt_async error: {e}")
        return 0
    value = task.value_outputs_unwrap(x, y, value_outputs)
    if cache_value:
        task.value_cache[value_prompt] = value
    return value

def get_values(task, x, ys, n_evaluate_sample, cache_value=True):
    values = []

    def evaluate(y):
        value = get_value(task, x, y, n_evaluate_sample, cache_value=cache_value)
        return value

    # with ThreadPoolExecutor() as executor:
    values = list(evaluate(y) for y in ys)

    return values

async def get_values_async(task, x, ys, n_evaluate_sample, cache_value=True):
    print('ys:', ys)
    tasks = [get_value_async(task, x, y, n_evaluate_sample, cache_value=cache_value) for y in ys]
    values = await asyncio.gather(*tasks)
    
    return values

def get_votes(task, x, ys, n_evaluate_sample):
    vote_prompt = task.vote_prompt_wrap(x, ys)
    vote_outputs = gpt(vote_prompt, n=n_evaluate_sample, stop=None)
    values = task.vote_outputs_unwrap(vote_outputs, len(ys))
    return values

def get_proposals(task, x, y): 
    propose_prompt = task.propose_prompt_wrap(x, y)
    proposals = gpt(propose_prompt, n=1, stop=None)[0].split('\n')
    return [y + _ + '\n' for _ in proposals]

async def get_proposals_async(task, x, y):
    propose_prompt = task.propose_prompt_wrap(x, y)
    proposals = await gpt_async(propose_prompt, n=1, stop=None)
    return [y + _ + '\n' for _ in proposals]

def get_samples(task, x, y, n_generate_sample, prompt_sample, stop):
    if prompt_sample == 'standard':
        prompt = task.standard_prompt_wrap(x, y)
    elif prompt_sample == 'cot':
        prompt = task.cot_prompt_wrap(x, y)
    else:
        raise ValueError(f'prompt_sample {prompt_sample} not recognized')
    samples = gpt(prompt, n=n_generate_sample, stop=stop)
    return [y + _ for _ in samples]

async def get_samples_async(task, x, y, n_generate_sample, prompt_sample, stop):
    if prompt_sample == 'standard':
        prompt = task.standard_prompt_wrap(x, y)
    elif prompt_sample == 'cot':
        prompt = task.cot_prompt_wrap(x, y)
    else:
        raise ValueError(f'prompt_sample {prompt_sample} not recognized')
    print("-------------prompt-------------")
    print('prompt:', prompt)
    samples = await gpt_async(prompt, n=n_generate_sample, stop=stop)
    return [y + _ for _ in samples]

def solve(args, task, idx, to_print=True, parallel = True):
    print('args', args)
    
    global gpt
    gpt = partial(gpt, model=args.backend, temperature=args.temperature)
    print(gpt)
    x = task.get_input(idx)  # input
    print('input:', x)
    ys = ['']  # current output candidates
    infos = []
    # loop = asyncio.get_event_loop()
    for step in range(task.steps):
        # generation
        if args.method_generate == 'sample':
            new_ys = []
            if parallel:
                tasks = [get_samples_async(task, x, y, args.n_generate_sample, args.prompt_sample, task.stops[step]) for y in ys]
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                new_ys = loop.run_until_complete(asyncio.gather(*tasks))
            else:
                new_ys = [get_samples(task, x, y, args.n_generate_sample, args.prompt_sample, task.stops[step]) for y in ys]
            new_ys = sum(new_ys, [])
            # new_ys = list(itertools.chain(*executor.map(lambda y: get_samples(task, x, y, args.n_generate_sample, prompt_sample=args.prompt_sample, stop=task.stops[step]), ys)))
        elif args.method_generate == 'propose':
            tasks = [get_proposals_async(task, x, y) for y in ys] if parallel else \
                    [get_proposals(task, x, y) for y in ys]
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            new_ys = loop.run_until_complete(asyncio.gather(*tasks))
            new_ys = sum(new_ys, [])
            # new_ys = list(itertools.chain(*executor.map(lambda y: get_proposals(task, x, y), ys)))

        ids = list(range(len(new_ys)))
        # evaluation
        if args.method_evaluate == 'vote':
            values = get_votes(task, x, new_ys, args.n_evaluate_sample)
        elif args.method_evaluate == 'value':
            # values = get_values(task, x, new_ys, args.n_evaluate_sample)
            values = asyncio.run(get_values_async(task, x, new_ys, args.n_evaluate_sample)) if parallel else \
                        get_values(task, x, new_ys, args.n_evaluate_sample)

        # selection
        if args.method_select == 'sample':
            ps = np.array(values) / sum(values)
            select_ids = np.random.choice(ids, size=args.n_select_sample, p=ps).tolist()
        elif args.method_select == 'greedy':
            select_ids = sorted(ids, key=lambda x: values[x], reverse=True)[:args.n_select_sample]
        select_new_ys = [new_ys[select_id] for select_id in select_ids]

        # log
        if to_print:
            sorted_new_ys, sorted_values = zip(*sorted(zip(new_ys, values), key=lambda x: x[1], reverse=True))
            print(f'-- new_ys --: {sorted_new_ys}\n-- sol values --: {sorted_values}\n-- choices --: {select_new_ys}\n')
        
        infos.append({'step': step, 'x': x, 'ys': ys, 'new_ys': new_ys, 'values': values, 'select_new_ys': select_new_ys})
        ys = select_new_ys
    
    if to_print: 
        print(ys)
    return ys, {'steps': infos}

def naive_solve(args, task, idx, to_print=True):
    global gpt
    gpt = partial(gpt, model=args.backend, temperature=args.temperature)
    print(gpt)
    x = task.get_input(idx)  # input
    ys = get_samples(task, x, '', args.n_generate_sample, args.prompt_sample, stop=None)
    return ys, {}
