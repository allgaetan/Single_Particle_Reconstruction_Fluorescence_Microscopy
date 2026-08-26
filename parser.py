from mains.synthetic_data.main_pixel_representation_synthetic_data import main_pixel_representation
from mains.real_data.main_pixel_representation_real_data import main_pixel_representation_real_data
import argparse
from inspect import signature

func_to_test = main_pixel_representation
sign = signature(func_to_test)
params = list(sign.parameters.keys())

parser = argparse.ArgumentParser()
for param in params:
    parser.add_argument(f'--{param}', dest=param, default=None)

args = parser.parse_args()
dict_param = {}

for param in params:
    arg_val = getattr(args, param)
    if arg_val is not None:
        dict_param[param] = arg_val

print(dict_param)
main_pixel_representation(**dict_param)




