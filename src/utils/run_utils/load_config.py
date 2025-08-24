import os
import yaml
from ray import tune
import numpy as np

# Ray Tune 설정을 로드하는 함수
def load_ray_tune_config(config_path="ray_tune_config.yml"):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config

def create_search_space(start, end, num_points, scale='linear'):
    num_points = int(num_points)
    if scale == 'linear':
        return tune.grid_search(np.linspace(start, end, num_points).round(10).tolist())
    elif scale == 'log10':
        return tune.grid_search(np.logspace(np.log10(start), np.log10(end), num_points).round(10).tolist())
    else:
        raise ValueError("Invalid scale type. Choose 'linear' or 'log10'.")

# YAML에서 정의한 검색 공간을 Ray Tune 객체로 매핑하는 함수
def map_search_space(search_space_config):
    search_space = {}

    for key, value in search_space_config.items():
        # 중첩된 딕셔너리인 경우 재귀적으로 처리
        if isinstance(value, dict) and 'type' not in value:
            # 'type' 키가 없으면 중첩된 구조로 간주하고 재귀 호출
            search_space[key] = map_search_space(value)
        else:
            # 'type' 키가 있으면 Ray Tune 파라미터로 처리
            param_type = value.get('type')
            
            if param_type == "fixed":
                # Fixed value, no tuning required
                search_space[key] = value['value']
            elif param_type == "uniform":
                # Uniform distribution
                search_space[key] = tune.uniform(float(value['low']), float(value['high']))
            
            elif param_type == "loguniform":
                # Log-uniform distribution
                search_space[key] = tune.loguniform(float(value['low']), float(value['high']))
            
            elif param_type == "choice":
                # Choice from a list of values
                search_space[key] = tune.choice(value['values'])
            
            elif param_type == "grid_search":
                # Choice from a list of values
                search_space[key] = tune.grid_search(value['values'])
            elif param_type == "linear_search":
                # Choice from a list of values
                search_space[key] = create_search_space(start=value['values'][0], end=value['values'][1], num_points=value['values'][2], scale='linear')
            elif param_type == "log10_search":
                # Choice from a list of values
                search_space[key] = create_search_space(start=value['values'][0], end=value['values'][1], num_points=value['values'][2], scale='log10')
            else:
                raise ValueError(f"Unsupported search space type: {param_type}")
    return search_space

def setup_path(config):
    default_path = config['search_space'].get('default_path').get('value')
    print(default_path)
    config['search_space']['data_config']['value'] = \
        os.path.join(default_path,config['search_space'].get('data_config').get('value'))
    if config['search_space'].get('skip_time_list',None):
        config['search_space']['skip_time_list']['value'] = \
            os.path.join(default_path,config['search_space'].get('skip_time_list').get('value'))
    config['tune_parameters']['run']['storage_path'] = \
        os.path.join(default_path,config['tune_parameters']['run'].get('storage_path'))