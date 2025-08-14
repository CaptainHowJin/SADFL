import copy
import torch
import numpy as np
import hdmedians as hdm
from scipy.stats import trim_mean
from sklearn.metrics import precision_recall_fscore_support
from models.Nets import VAE
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import sklearn.metrics.pairwise as smp
import math
from sklearn import metrics
from torch.nn.utils import parameters_to_vector
from models.Nets import LogisticRegression, SimpleCNN, ImprovedSimpleCNN
from torch.utils.data import DataLoader, Dataset, TensorDataset
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def get_pca(data):
    data = StandardScaler().fit_transform(data)
    pca = PCA(n_components=2)
    data = pca.fit_transform(data)
    return data

def get_err_threhold(fpr, tpr, threshold):
    differ_tpr_fpr_1=tpr+fpr-1.0
    right_index = np.argmin(np.abs(differ_tpr_fpr_1))
    best_th = threshold[right_index]
    dr = tpr[right_index]
    far = fpr[right_index]
    return dr, far, best_th, right_index

def Metrics(test_y, error):
    auc = metrics.roc_auc_score(test_y, error)
    pr = metrics.average_precision_score(test_y, error)

    fpr, tpr, thresholds = metrics.roc_curve(test_y, error, pos_label=1)
    dr, far, best_th, _ = get_err_threhold(fpr, tpr, thresholds)
    test_labels = np.where(error > best_th, 1, 0)
    f1 = metrics.f1_score(test_y, test_labels)

    return auc, pr, f1




def to_ndarray(w_locals):
    for user_idx in range(len(w_locals)):
        for key in w_locals[user_idx]:
            if isinstance(w_locals[user_idx][key], torch.Tensor):
                w_locals[user_idx][key] = w_locals[user_idx][key].cpu().numpy()
    return w_locals

def reshape_from_oneD(one_d_vector, shape_size_dict, args=None):
	weight_dict = {}
	one_d_vector_idx = 0
	for key in shape_size_dict:
		weight_dict[key] =  copy.deepcopy( one_d_vector[one_d_vector_idx: (one_d_vector_idx + shape_size_dict[key][0])] ).reshape(shape_size_dict[key][1])

		if args != None:
			weight_dict[key] = torch.tensor(weight_dict[key]).to(args.device)

		one_d_vector_idx += shape_size_dict[key][0]
	return weight_dict

def trimmed_mean_user_one_d(user_one_d, trim_ratio):

    if trim_ratio == 0:
        return np.mean(user_one_d)
    
    assert trim_ratio < 0.5, 'Trim ratio is {}, but it should be less than 0.5'.format(trim_ratio)
    
    trim_num = int(trim_ratio * len(user_one_d))
    
    user_one_d_t = user_one_d.T

    sorted_user_weights = np.sort(user_one_d_t, axis=1)

    trimmed_weights = sorted_user_weights[:, trim_num:-trim_num]

    trimmed_means = np.mean(trimmed_weights, axis=1).T
    
    return trimmed_means

def repeated_median_for_user_one_d(user_one_d, SHARD_SIZE=100000):

    num_users, total_num = user_one_d.shape
    
    device = "cuda" if torch.cuda.is_available() else "cpu" 
    y = torch.tensor(user_one_d, dtype=torch.float).to(device) 
    
    # 判断是否需要分片处理
    if total_num < SHARD_SIZE:
        slopes, intercepts = repeated_median(y)
        y = intercepts + slopes * (num_users - 1) / 2.0
    else:
        y_result = torch.FloatTensor(total_num).to(device)
        num_shards = int(math.ceil(total_num / SHARD_SIZE))
        for i in range(num_shards):
            start_index = i * SHARD_SIZE
            end_index = min((i + 1) * SHARD_SIZE, total_num)
            y_shard = y[:, start_index:end_index]
            slopes_shard, intercepts_shard = repeated_median(y_shard)
            y_shard = intercepts_shard + slopes_shard * (num_users - 1) / 2.0
            y_result[start_index:end_index] = y_shard
        y = y_result

    return y.cpu().numpy() 

def repeated_median(y):
    num_points = y.shape[1]
    slopes = []

    for i in range(num_points):
        for j in range(i + 1, num_points):
            slope = (y[:, j] - y[:, i]) / (j - i)
            slopes.append(slope)

    slopes = torch.stack(slopes)
    median_slope = torch.median(slopes, dim=0)[0]

    intercepts = []
    for i in range(num_points):
        intercept = y[:, i] - median_slope * i
        intercepts.append(intercept)

    intercepts = torch.stack(intercepts)
    median_intercept = torch.median(intercepts, dim=0)[0]

    return median_slope, median_intercept

def convert_to_model_updates(w_locals):
    model_updates = {}
    
    # For each parameter in the model (e.g., 'conv1.weight', 'fc2.bias')
    for name in w_locals[0].keys():
        # Convert numpy arrays to torch tensors and flatten all the updates for this parameter (from all participants)
        updates = [torch.tensor(w[name], dtype=torch.float32).flatten() for w in w_locals]  # Convert to Tensor and flatten
        
        # Stack them into a tensor of shape [num_participants, param_size]
        stacked_updates = torch.stack(updates)  # Now it should be 2D
        
        # Save the stacked updates in the model_updates dictionary
        model_updates[name] = stacked_updates
    
    return model_updates

def aggregation(w_locals, user_weights, args, attacker_idx, w_glob, dataset_test=None):

    layer_shape_size = {}
    for key in w_locals[0]:
        layer_shape_size[key] = ( w_locals[0][key].numel(), list(w_locals[0][key].shape) )
    print(layer_shape_size)

    w_dict = {i: value for i, value in enumerate(w_locals)}

    # to numpy array
    w_locals = to_ndarray(w_locals)

    keys_of_interest = ['linear.weight', 'linear.bias']

    user_one_d = []
    for user_idx in range(len(w_locals)):
        user_weights_of_interest = np.concatenate([w_locals[user_idx][key].flatten() for key in keys_of_interest if key in w_locals[user_idx]])
        user_one_d.append(user_weights_of_interest)

    if args.aggregation == "MKrum":
        print("Using {} aggregation".format(args.aggregation))
        num_machines = len(w_locals) 
        num_byz = int(args.attack_ratio * num_machines) 
        num_near = num_machines - num_byz - 2 

        values = []
        for user_idx in range(len(w_locals)):
            tmp = np.array([])
            for key in w_locals[user_idx]:
                data_idx_key = np.array(w_locals[user_idx][key]).flatten()
                tmp = np.hstack((tmp, data_idx_key))
            values.append(tmp)

        scores = []
        for i, w_i in enumerate(values):
            dist = []
            for j, w_j in enumerate(values):
                if i != j:
                    dist.append(np.linalg.norm(w_i - w_j) ** 2) 
            dist.sort() 
            scores.append(sum(dist[:num_near])) 

        k = num_machines - num_byz - 2  
        selected_indices = np.argsort(scores)[:k]  

        aggregated = np.zeros_like(values[0])
        for idx in selected_indices:
            aggregated += values[idx]
        aggregated /= len(selected_indices) 

        total_samples = len(w_locals) 
        total_malicious = len(attacker_idx) 
        total_benign = total_samples - total_malicious  

        malicious_detected = total_malicious - len(set(attacker_idx) & set(selected_indices))
        benign_detected = len(selected_indices) - len(set(attacker_idx) & set(selected_indices))

        DAR = (malicious_detected + benign_detected) / total_samples
        DPR = malicious_detected / total_malicious if total_malicious > 0 else 0
        RR = malicious_detected / (total_samples - len(selected_indices)) if (malicious_detected + benign_detected) > 0 else 0

        print(f"DAR: {DAR:.4f}, DPR: {DPR:.4f}, RR: {RR:.4f}")

        return reshape_from_oneD(aggregated, layer_shape_size, args), DAR, DPR, RR
    
    elif args.aggregation == "defend":
        print("Using {} aggregation".format(args.aggregation))
        

        global_model = list(w_glob.values()) 
        last_g = global_model[-2].clone().detach().cpu().numpy()  

        m = len(w_locals)
        f_grads = [None for i in range(m)]
        
        # for i in range(m):
        #     grad= (last_g - \
        #             list(w_locals[i].values())[-2])
        #     f_grads[i] = grad.reshape(-1)
        
        for i in range(m):
            local_param = list(w_locals[i].values())[-2]
        
            if isinstance(local_param, torch.Tensor):
                local_param = local_param.detach().cpu().numpy()
        
            grad = last_g - local_param

            if grad.shape[0] == 0:
                raise ValueError(f"Gradient for client {i} is empty!")

            f_grads[i] = grad.reshape(-1)

        if len(f_grads) == 0 or np.all(np.array(f_grads) == 0):
            raise ValueError("f_grads is empty or all zeros, check local updates.")
        
        cs = smp.cosine_similarity(f_grads) - np.eye(m)
        cs = get_pca(cs)
        centroid = np.median(cs, axis = 0)
        scores = smp.cosine_similarity([centroid], cs)[0]
        
        # trust = scores/scores.max()
        trust = scores / (scores.max() + 1e-8)  
        trust[(trust < 0)] = 0
        # print(f"Trust scores: {trust}")
        
        selected_indices = np.where(trust > 0)[0].tolist()
        

        values = []
        for user_idx in range(len(w_locals)):
            tmp = np.array([])
            for key in w_locals[user_idx]:
                data_idx_key = np.array(w_locals[user_idx][key]).flatten()
                tmp = np.hstack((tmp, data_idx_key))  
            values.append(tmp)
        
        aggregated = np.zeros_like(values[0])
        for idx in selected_indices:
            aggregated += values[idx] * trust[idx]
        aggregated /= len(selected_indices) 
        

        total_samples = len(w_locals) 
        total_malicious = len(attacker_idx)  
        total_benign = total_samples - total_malicious  

        malicious_detected = total_malicious - len(set(attacker_idx) & set(selected_indices))
        benign_detected = len(selected_indices) - len(set(attacker_idx) & set(selected_indices))

        DAR = (malicious_detected + benign_detected) / total_samples
        DPR = malicious_detected / total_malicious if total_malicious > 0 else 0
        RR = malicious_detected / (total_samples - len(selected_indices)) if (malicious_detected + benign_detected) > 0 else 0

        print(f"DAR: {DAR:.4f}, DPR: {DPR:.4f}, RR: {RR:.4f}")
        
        return reshape_from_oneD(aggregated, layer_shape_size, args), DAR, DPR, RR
        
    
    elif args.aggregation == "trust":
        print("Using {} aggregation".format(args.aggregation))

        param_updates = []
    
        # FIX: Convert w_glob to a vector representation
        glob = parameters_to_vector([v.clone().detach().to(args.device) for v in w_glob.values()])

        for trained_local_model in w_locals:
            local_params = parameters_to_vector([torch.tensor(v).to(args.device) for v in trained_local_model.values()])
            param_updates.append(local_params - glob)

        input_size = 784
        num_classes = 10
        model = LogisticRegression(input_size, num_classes).to(args.device)
        model.load_state_dict(w_glob)
    
        optimizer = torch.optim.Adam(model.parameters())
        epochs = 1
        criterion = torch.nn.CrossEntropyLoss()
    
        root_dataset = DataLoader(dataset_test, batch_size=args.local_bs, shuffle=True)
    
        for _ in range(epochs):
            for inputs, labels in root_dataset:
                inputs, labels = inputs.to(args.device), labels.to(args.device)
                optimizer.zero_grad()
                loss = criterion(model(inputs), labels)
                loss.backward()
                optimizer.step()

        # FIX: Compute clean_param_update using correctly processed glob
        clean_param_update = parameters_to_vector(model.parameters()) - glob

        model_updates = convert_to_model_updates(w_locals)

        cos = torch.nn.CosineSimilarity(dim=0)
        g0_norm = torch.norm(clean_param_update)
        weights = []
    
        for param_update in param_updates:
            sim = cos(param_update.flatten(), clean_param_update.flatten())
            weights.append(F.relu(sim))  
        # for param_update in param_updates:
        #     weights.append(F.relu(cos(param_update.view(-1, 1), clean_param_update.view(-1, 1))))
    
        weights = torch.tensor(weights).to(args.device).view(1, -1)
        if weights.sum() == 0:
            weights = torch.ones_like(weights) / weights.shape[1] 
        else:
            weights = weights / weights.sum()

        # weights = weights / weights.sum()
        weights = torch.where(weights[0].isnan(), torch.zeros_like(weights), weights)
    
        nonzero_weights = torch.count_nonzero(weights.flatten())
        nonzero_indices = torch.nonzero(weights.flatten()).flatten()

        print(f'g0_norm: {g0_norm}, '
            f'weights_sum: {weights.sum()}, '
            f'*** {nonzero_weights} *** model updates are considered to be aggregated!')

        normalize_weights = []
        for param_update in param_updates:
            normalize_weights.append(g0_norm / torch.norm(param_update))

        global_update = dict()
        for name, params in model_updates.items():
            if 'num_batches_tracked' in name or 'running_mean' in name or 'running_var' in name:
                global_update[name] = 1 / nonzero_weights * params[nonzero_indices].sum(dim=0, keepdim=True)
            else:
                global_update[name] = torch.matmul(
                    weights,
                    params.to(args.device) * torch.tensor(normalize_weights).to(args.device).view(-1, 1))
    
        selected = torch.cat([torch.flatten(value) for value in global_update.values()])
        selected = selected.cpu().numpy()
        
        detected_malicious = set(nonzero_indices.cpu().numpy())  
        true_malicious = set(attacker_idx)  
        benign_clients = set(range(len(w_locals))) - true_malicious  

        true_positive = len(detected_malicious & true_malicious)  
        false_positive = len(detected_malicious & benign_clients)  
        false_negative = len(true_malicious - detected_malicious)  
        true_negative = len(benign_clients - detected_malicious)  

        total_clients = len(w_locals)
        total_malicious = len(true_malicious)

        # 计算指标
        DAR = (true_positive + true_negative) / total_clients  
        DPR = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0  # 精确率
        RR = true_positive / len(detected_malicious) if len(detected_malicious) > 0 else 0  

        print(f"DAR: {DAR:.4f}, DPR: {DPR:.4f}, RR: {RR:.4f}")

        return reshape_from_oneD(selected, layer_shape_size, args), DAR, DPR, RR
    
    elif args.aggregation == "RLR":
        print( "Using {} aggregation".format(args.aggregation) )
        # user_one_d = np.array(user_one_d).astype(float)
        
        model_updates = convert_to_model_updates(w_locals)
        
        global_update = dict()
        for name, param in model_updates.items():
            signs = torch.sign(model_updates[name])
            sm_of_signs = torch.abs(torch.sum(signs, dim=0, keepdim=True))
            sm_of_signs[sm_of_signs < 10] = -1
            sm_of_signs[sm_of_signs >= 10] = 1
            global_update[name] = 1 / args.sample_users * \
                                (sm_of_signs * model_updates[name].sum(dim=0, keepdim=True))
                                
        malicious_clients = []
        selected_indices = []
        
        # Track which clients are likely malicious or benign based on sm_of_signs
        for idx, sign in enumerate(sm_of_signs.flatten()):
            if sign == -1:
                malicious_clients.append(idx)
            else:
                selected_indices.append(idx)
                

        total_samples = len(w_locals)  
        total_malicious = len(attacker_idx) 
        total_benign = total_samples - total_malicious  


        malicious_detected = total_malicious - len(set(attacker_idx) & set(selected_indices))
        benign_detected = len(selected_indices) - len(set(attacker_idx) & set(selected_indices))

        DAR = (malicious_detected + benign_detected) / total_samples
        DPR = malicious_detected / total_malicious if total_malicious > 0 else 0
        RR = malicious_detected / (total_samples - len(selected_indices)) if (malicious_detected + benign_detected) > 0 else 0

        print(f"DAR: {DAR:.4f}, DPR: {DPR:.4f}, RR: {RR:.4f}")
        
        selected = torch.cat([torch.flatten(value) for value in global_update.values()])
        
        selected = selected.cpu().numpy()

        return reshape_from_oneD(selected, layer_shape_size, args), DAR, DPR, RR
    
    elif args.aggregation == "Fools":
        print( "Using {} aggregation".format(args.aggregation) )
        # user_one_d = np.array(user_one_d).astype(float)
        
        values = []
        for user_idx in range(len(w_locals)):
            tmp = np.array([])
            for key in w_locals[user_idx]:
                data_idx_key = np.array(w_locals[user_idx][key]).flatten()
                tmp = np.hstack((tmp, data_idx_key))
            values.append(tmp)

        n_clients = len(w_locals)

        cs = smp.cosine_similarity(values) - np.eye(n_clients)


        maxcs = np.max(cs, axis=1)
        
        pardoned_cs = np.zeros_like(cs)
        for i in range(n_clients):
            for j in range(n_clients):
                if i == j:
                    continue
                if maxcs[i] < maxcs[j]:
                    pardoned_cs[i, j] = cs[i, j] * (maxcs[i] / maxcs[j])
                else:
                    pardoned_cs[i, j] = cs[i, j] 
        
        wv = 1 - (np.max(cs, axis=1))
        
        wv = np.clip(wv, 0, 1)  
        wv = wv / np.max(wv)  
        wv = np.where(wv == 1, 0.99, wv)  
        
        wv = np.log(wv / (1 - wv + 1e-8)) + 0.5
        wv = np.clip(wv, 0, 1)  
        
        total_weight = np.sum(wv)
        aggregated = np.zeros_like(values[0])
        for idx in range(n_clients):
            aggregated += values[idx] * wv[idx]
        if total_weight > 0:
            aggregated /= total_weight
        else:
            aggregated = np.mean(values, axis=0)  
        
        selected_indices = np.where(wv > 0)[0]
        

        total_samples = len(w_locals)  
        total_malicious = len(attacker_idx)  
        total_benign = total_samples - total_malicious  

        malicious_detected = total_malicious - len(set(attacker_idx) & set(selected_indices))
        benign_detected = len(selected_indices) - len(set(attacker_idx) & set(selected_indices))

        DAR = (malicious_detected + benign_detected) / total_samples
        DPR = malicious_detected / total_malicious if total_malicious > 0 else 0
        RR = malicious_detected / (total_samples - len(selected_indices)) if (total_samples - len(selected_indices)) > 0 else 0

        print(f"DAR: {DAR:.4f}, DPR: {DPR:.4f}, RR: {RR:.4f}")

        return reshape_from_oneD(aggregated, layer_shape_size, args), DAR, DPR, RR
        

    elif args.aggregation == "GeoMed":
        print( "Using {} aggregation".format(args.aggregation) )
        user_one_d = np.array(user_one_d).astype(float)

        selected = np.asarray(hdm.geomedian(user_one_d, axis=0))

        return reshape_from_oneD(selected, layer_shape_size, args)
    
    elif args.aggregation == "Tmean":
        print( "Using {} aggregation".format(args.aggregation) )
        user_one_d = np.array(user_one_d).astype(float)
        trim_ratio = 0.2  # attacker ratio
        
        selected = trimmed_mean_user_one_d(user_one_d, trim_ratio)
        
        return reshape_from_oneD(selected, layer_shape_size, args)

    elif args.aggregation == "atten":
        true = [False] * args.sample_users
        for idx in attacker_idx:
            true[idx] = True

        user_one_d_test = copy.deepcopy(user_one_d)
        
        model = VAE( input_dim = user_one_d_test[0].shape[0] )
        model.load_state_dict( torch.load(args.vae_model) )
        model.eval()
        
        scores = model.test(user_one_d_test)
        score_avg = np.mean(scores)
        print("scores", scores)
        print("score_avg", score_avg)
        predicted = scores > score_avg
        

        total_samples = len(true)
        total_malicious = sum(true)  
        total_benign = total_samples - total_malicious  
        
        malicious_detected = sum([t and p for t, p in zip(true, predicted)])
        false_positive = sum([not t and p for t, p in zip(true, predicted)])
        benign_detected = sum([not t and not p for t, p in zip(true, predicted)])  

        DAR = (malicious_detected + benign_detected) / total_samples  
        DPR = malicious_detected / total_malicious if total_malicious > 0 else 0  
        RR = malicious_detected / (malicious_detected + false_positive) if (malicious_detected + false_positive) > 0 else 0 
        
        print(f"DAR: {DAR:.4f}, DPR: {DPR:.4f}, RR: {RR:.4f}")
        
        keys = []

        new_weights = copy.deepcopy(user_weights)
        for _ in range(len(predicted)):
            if predicted[_]:
                new_weights[_] = 0.0
            if not predicted[_]:
                keys.append(_)
        new_weights = new_weights / sum(new_weights)

        # user_one_d = np.array(user_one_d)

        values_of_interest = [w_dict[key] for key in keys]
        values = []
        for user_idx in range(len(w_locals)):
            tmp = np.array([])
            for key in w_locals[user_idx]:
                data_idx_key = np.array(w_locals[user_idx][key]).flatten()
                tmp = copy.deepcopy( np.hstack((tmp, data_idx_key)) )  
            values.append(tmp)
        # to numpy array
        selected_a = to_ndarray(values_of_interest)
        # print(values_of_interest[0])
        selected = np.zeros(values[0].shape)
        for _ in range(len(new_weights)):
            selected += values[_] * new_weights[_]



        return reshape_from_oneD(selected, layer_shape_size, args), DAR, DPR, RR
    
    elif args.aggregation == "vaegan":
        print("Using {} aggregation".format(args.aggregation))
        # this is our method
        true = [False] * args.sample_users
        for idx in attacker_idx:
            true[idx] = True

        user_one_d_test = copy.deepcopy(user_one_d)
        
        
        netg = Generator(input_dim = user_one_d_test[0].shape[0]).to(args.device)
        # netd = Discriminator(input_dim = user_one_d_test[0].shape[0]).to(device)
        
        netg.load_state_dict( torch.load(args.vae_model) )
        # netd.load_state_dict( torch.load("./VAE_data/netd_imdb.pth") )
        
        netg.eval()
        # netd.eval()
        
        running_loss = []
        for data in user_one_d_test:
            data = torch.tensor(data, dtype=torch.float)  
            data = data.to(args.device)  
            
            x_in = Variable(data)
        
            x_out, z_mean_1, z_logvar_1, z_mean_2, z_logvar_2 = netg(data)
        
            x_out = x_out.view(-1)
            x_in = x_in.view(-1)

            # cuda = True if torch.cuda.is_available() else False
            
            # real_label = Variable(torch.ones(1)).cuda() if cuda else Variable(torch.ones(1))
            # bce_criterion = nn.BCELoss()
            # g_loss_output = netd(x_out)
            # g_loss = bce_criterion(g_loss_output, real_label.view(-1, 1))
            
            lambda_rec = 0.4
            beta1 = 0.15
            beta2 = 0.15
            lambda_enc = 0.3
            
            recon_loss = F.mse_loss(x_out, x_in, reduction='sum')
            kld_loss_1 = -0.5 * torch.sum(1 + z_logvar_1 - z_mean_1.pow(2) - z_logvar_1.exp())
            kld_loss_2 = -0.5 * torch.sum(1 + z_logvar_2 - z_mean_2.pow(2) - z_logvar_2.exp())
            enc_diff_loss = F.mse_loss(z_mean_1, z_mean_2) 

            loss = lambda_rec * recon_loss + beta1 * kld_loss_1 + beta2 * kld_loss_2 + lambda_enc * enc_diff_loss
            
            running_loss.append(loss)

        running_loss_np = [x.detach().cpu().numpy() for x in running_loss]
        score_avg = np.mean(running_loss_np)
        print("scores", running_loss_np)
        print("score_avg", score_avg)
        predicted = running_loss_np > score_avg
        

        total_samples = len(true)
        total_malicious = sum(true) 
        total_benign = total_samples - total_malicious  
        
        malicious_detected = sum([t and p for t, p in zip(true, predicted)])  
        false_positive = sum([not t and p for t, p in zip(true, predicted)]) 
        benign_detected = sum([not t and not p for t, p in zip(true, predicted)]) 

        DAR = (malicious_detected + benign_detected) / total_samples  
        DPR = malicious_detected / total_malicious if total_malicious > 0 else 0  
        RR = malicious_detected / (malicious_detected + false_positive) if (malicious_detected + false_positive) > 0 else 0 
        
        print(f"DAR: {DAR:.4f}, DPR: {DPR:.4f}, RR: {RR:.4f}")
        
        keys = []

        new_weights = copy.deepcopy(user_weights)
        for _ in range(len(predicted)):
            if predicted[_]:
                new_weights[_] = 0.0
            if not predicted[_]:
                keys.append(_)
        new_weights = new_weights / sum(new_weights)

        # user_one_d = np.array(user_one_d)
        
        values_of_interest = [w_dict[key] for key in keys]
        values = []
        for user_idx in range(len(w_locals)):
            tmp = np.array([])
            for key in w_locals[user_idx]:
                data_idx_key = np.array(w_locals[user_idx][key]).flatten()
                tmp = copy.deepcopy( np.hstack((tmp, data_idx_key)) )  
            values.append(tmp)
        # to numpy array
        selected_a = to_ndarray(values_of_interest)
        # print(values_of_interest[0])
        selected = np.zeros(values[0].shape)
        for _ in range(len(new_weights)):
            selected += values[_] * new_weights[_]

        return reshape_from_oneD(selected, layer_shape_size, args), DAR, DPR, RR
    
if __name__ == "__main__":

    test = {"a": np.array([[1,2],[3,4]]), "b": np.array([5,6,7,8]), "c": np.array([[[9,10], [11,12]], [[13,14], [15,16]]])}
    print(test)

    layer_shape_size = {}
    for key in test:
        layer_shape_size[key] = ( test[key].size, list(test[key].shape) )

    tmp = np.array([])
    for key in test:
        print(key)
        data_idx_key = np.array(test[key]).flatten()
        tmp = copy.deepcopy( np.hstack((tmp, data_idx_key)) )
    print(tmp)
    print(reshape_from_oneD(tmp, layer_shape_size))



