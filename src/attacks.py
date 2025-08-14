import numpy as np
import copy
import torch
from collections import OrderedDict
from scipy.stats import norm
import math

def LIT_attack(w_locals, global_model, args):
    delta_ws = []
    for w_local in w_locals:
        delta_w = OrderedDict()
        for key in w_local.keys():
            delta_w[key] = w_local[key] - global_model[key]
        delta_ws.append(delta_w)

    delta_mean = OrderedDict()
    delta_std = OrderedDict()
    
    for key in global_model.keys():
        stacked_tensors = torch.stack([delta_w[key] for delta_w in delta_ws])
        delta_mean[key] = torch.mean(stacked_tensors, dim=0)
        delta_std[key] = torch.std(stacked_tensors, dim=0, unbiased=True)

    n_clients = args.num_users
    n_malicious = args.attack_ratio * n_clients
    
    s = math.floor(n_clients / 2 + 1) - n_malicious
    z = norm.ppf((n_clients - n_malicious - s) / (n_clients - n_malicious))

    for attacker_idx in range(len(w_locals)):
        for key in global_model.keys():
            lower_bound = delta_mean[key] - z * delta_std[key]
            upper_bound = delta_mean[key] + z * delta_std[key]
            w_locals[attacker_idx][key] = torch.clamp(delta_ws[attacker_idx][key], min=lower_bound, max=upper_bound) + global_model[key]

    return w_locals

def sign_flipping_attack(weights, attack_value = -1):
    weights_modified = copy.deepcopy(weights)
    for k in weights_modified.keys():
        weights_modified[k] = weights_modified[k] * attack_value     
    return weights_modified

def additive_noise(weights, args):
    weights_modified = copy.deepcopy(weights)
    for k in weights_modified.keys():
        noise = torch.from_numpy( copy.deepcopy( np.random.normal( scale=0.5, size=weights_modified[k].shape ) ) ).to(args.device) 
        weights_modified[k] = weights_modified[k] + noise     
    return weights_modified

def GaussianNoise(weights, args):
    weights_modified = copy.deepcopy(weights)
    for k, v in weights_modified.items():
        v = v.to(torch.float32).to(args.device) 
        
        noise = torch.randn(v.shape).to(args.device)
        a = torch.mean(v).to(args.device)  
        b = torch.std(v).to(args.device)  
        v_gaussian = a + noise * b * 2
        weights_modified[k] = v_gaussian
    return weights_modified

if __name__ == "__main__":
    # test_input = np.random.rand(10, 10)
    test_input = np.array([ np.random.rand(10,1), np.random.rand(1, 10), np.random.rand(2,3,4), 
                    np.random.rand(5,4,3,2,2) ])
    print("[ORIGINAL]", test_input)
    # print([  same_value_attack(item) for item in test_input ])
    test_input = np.array([ random_attack(item) for item in test_input ])
    print(test_input)
    # print("[SAME VALUE]", same_value_attack(test_input))
    # print("[SIGN FLIP]", sign_flipping_attack(test_input))
    # print("[RANDOM]", random_attack(test_input))