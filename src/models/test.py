import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

def test_imgim(net_g, datatest, args):
    net_g.eval()
    # testing
    test_loss = 0
    correct = 0
    total = 0
    data_loader = DataLoader(datatest, batch_size=args.bs)
    l = len(data_loader)
    for idx, (data, target) in enumerate(data_loader):
        if args.gpu != -1:
            data, target = data.to(args.device), target.to(args.device)
        log_probs = net_g(data)
        # sum up batch loss
        # test_loss += F.cross_entropy(log_probs, target, reduction='sum').item()
        # get the index of the max log-probability
        # y_pred = log_probs.data.max(1, keepdim=True)[1]
        y_pred = torch.round(torch.sigmoid(log_probs.squeeze()))
        total += target.size(0)
        correct += (y_pred == target).sum().item()
        # correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()

    # test_loss /= len(data_loader.dataset)
    accuracy = 100.00 * correct / total
    if args.verbose:
        print('\nTest set: Average loss: {:.4f} \nAccuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, correct, len(data_loader.dataset), accuracy))
    return accuracy, test_loss

def test_imdb_poison(net_g, datatest, args):
    net_g.eval()
    net_g.to(args.device)
    # testing
    test_loss = 0.0
    correct = 0
    total = 0

    poison_correct = 0.0

    data_loader = DataLoader(datatest, batch_size=args.bs)
    l = len(data_loader)
    for idx, (data, target) in enumerate(data_loader):
        if args.gpu != -1:
            data, target = data.cuda(), target.cuda()
        log_probs = net_g(data)
        # sum up batch loss
        # test_loss += F.cross_entropy(log_probs.squeeze(), target, reduction='sum').item()
        # get the index of the max log-probability
        y_pred = torch.round(torch.sigmoid(log_probs.squeeze()))
        total += target.size(0)
        correct += (y_pred == target).sum().item()
        # y_pred = log_probs.data.max(1, keepdim=True)[1]

        # y_gold = target.data.view_as(y_pred).squeeze(1).long()
        # y_pred = y_pred.squeeze(1)

        # y_gold = y_gold.cpu()  
        # y_pred = y_pred.cpu()  

        for pred_idx in range(len(y_pred)):
            # gold_all[ y_gold[pred_idx] ] += 1
            # if y_pred[pred_idx] == target[pred_idx]:
            #     correct[y_pred[pred_idx]] += 1
            if y_pred[pred_idx] == 0 and target[pred_idx] == 1:  # poison attack
                poison_correct += 1






        # for pred_idx in range(len(y_pred)):
        #     gold_all[ y_gold[pred_idx] ] += 1
        #     if y_pred[pred_idx] == y_gold[pred_idx]:
        #         correct[y_pred[pred_idx]] += 1
        #     elif y_pred[pred_idx] == 1 and y_gold[pred_idx] == 0:  # poison attack
        #         poison_correct += 1

    # test_loss /= len(data_loader.dataset)

    # accuracy = 100.00 * (sum(correct) / sum(gold_all)).item()
    accuracy = 100.00 * correct / total
    if args.verbose:
        print('\nTest set: Average loss: {:.4f} \nAccuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, correct, len(data_loader.dataset), accuracy))
        
    # acc_per_label = correct / gold_all

    return accuracy, test_loss, poison_correct/total*100

def test_img_poison1(net_g, datatest, args):
    net_g.eval()
    # testing
    test_loss = 0
    if args.dataset == "mnist":
        correct  = torch.tensor([0.0] * 10)
        gold_all = torch.tensor([0.0] * 10)
    elif args.dataset == "femnist":
        correct  = torch.tensor([0.0] * 62)
        gold_all = torch.tensor([0.0] * 62)
    else:
        print("Unknown dataset")
        exit(0)

    poison_correct = 0.0

    data_loader = DataLoader(datatest, batch_size=args.bs)
    l = len(data_loader)
    for idx, (data, target) in enumerate(data_loader):
        if args.gpu != -1:
            data, target = data.to(args.device), target.to(args.device)
        log_probs = net_g(data)
        # sum up batch loss
        test_loss += F.cross_entropy(log_probs, target, reduction='sum').item()
        # get the index of the max log-probability
        y_pred = log_probs.data.max(1, keepdim=True)[1]

        y_gold = target.data.view_as(y_pred).squeeze(1)
        y_pred = y_pred.squeeze(1)

        for pred_idx in range(len(y_pred)):
            gold_all[ y_gold[pred_idx] ] += 1
            if y_pred[pred_idx] == y_gold[pred_idx]:
                correct[y_pred[pred_idx]] += 1
            elif y_pred[pred_idx] == 5 and y_gold[pred_idx] == 3:  # poison attack
                poison_correct += 1

    test_loss /= len(data_loader.dataset)

    accuracy = 100.00 * (sum(correct) / sum(gold_all)).item()
    acc_per_label = correct / gold_all

    return accuracy, test_loss, acc_per_label.tolist(), poison_correct/gold_all[7].item()


def test_img_poison(net_g, datatest, args, attacker_idxs=None):
    net_g.eval()
    # testing
    test_loss = 0
    if args.dataset == "mnist":
        correct  = torch.tensor([0.0] * 10)
        gold_all = torch.tensor([0.0] * 10)
    elif args.dataset == "femnist":
        correct  = torch.tensor([0.0] * 62)
        gold_all = torch.tensor([0.0] * 62)
    else:
        print("Unknown dataset")
        exit(0)

    poison_correct = 0.0

    data_loader = DataLoader(datatest, batch_size=args.bs)
    l = len(data_loader)
    for idx, (data, target) in enumerate(data_loader):
        if args.gpu != -1:
            data, target = data.to(args.device), target.to(args.device)
        log_probs = net_g(data)
        # sum up batch loss
        test_loss += F.cross_entropy(log_probs, target, reduction='sum').item()
        # get the index of the max log-probability
        y_pred = log_probs.data.max(1, keepdim=True)[1]

        y_gold = target.data.view_as(y_pred).squeeze(1)
        y_pred = y_pred.squeeze(1)

        for pred_idx in range(len(y_pred)):
            gold_all[y_gold[pred_idx]] += 1
            if y_pred[pred_idx] == y_gold[pred_idx]:
                correct[y_pred[pred_idx]] += 1
            elif y_pred[pred_idx] == 5 and y_gold[pred_idx] == 3:  # poison attack
                poison_correct += 1

    test_loss /= len(data_loader.dataset)

    accuracy = 100.00 * (sum(correct) / sum(gold_all)).item()
    acc_per_label = correct / gold_all
    
    ASR = poison_correct / gold_all[3].item() if gold_all[3].item() > 0 else 0  # 攻击成功率

    return accuracy, test_loss, acc_per_label.tolist(), ASR



def test_img_poisonds(net_g, datatest, args, attacker_idxs=None, attack_target_label=0, attack_source_label=None):
    net_g.eval()
    test_loss = 0
    num_classes = 10 if args.dataset == "mnist" else 62
    correct = torch.zeros(num_classes).to(args.device)
    gold_all = torch.zeros(num_classes).to(args.device)
    poison_correct = 0.0
    total_attacked = 0.0

    data_loader = DataLoader(datatest, batch_size=args.bs)
    for data, target in data_loader:
        data, target = data.to(args.device), target.to(args.device)
        log_probs = net_g(data)
        test_loss += F.cross_entropy(log_probs, target, reduction='sum').item()
        y_pred = log_probs.argmax(dim=1)
        y_gold = target

        # 统计常规准确率
        for cls in range(num_classes):
            cls_mask = (y_gold == cls)
            correct[cls] += (y_pred[cls_mask] == cls).sum()
            gold_all[cls] += cls_mask.sum()


        if attack_source_label is None:

            source_mask = (y_gold != attack_target_label)
        else:

            source_mask = (y_gold == attack_source_label)
        

        attack_mask = (y_pred == attack_target_label) & source_mask
        poison_correct += attack_mask.sum().item()
        total_attacked += source_mask.sum().item()

    test_loss /= len(data_loader.dataset)
    accuracy = 100.0 * correct.sum().item() / gold_all.sum().item()
    acc_per_label = (correct / gold_all).cpu().numpy()
    ASR = 100.0 * poison_correct / total_attacked if total_attacked > 0 else 0.0

    return accuracy, test_loss, acc_per_label.tolist(), ASR

    # if args.verbose:
    #     print('\nTest set: Average loss: {:.4f} \nAccuracy: {}/{} ({:.2f}%)\n'.format(
    #         test_loss, correct, len(data_loader.dataset), accuracy))
    # return accuracy, test_loss

def test_img(net_g, datatest, args):
    net_g.eval()
    # testing
    test_loss = 0
    correct = 0
    data_loader = DataLoader(datatest, batch_size=args.bs)
    l = len(data_loader)
    for idx, (data, target) in enumerate(data_loader):
        if args.gpu != -1:
            data, target = data.to(args.device), target.to(args.device)
        log_probs = net_g(data)
        

        target = target.long()
        
        # sum up batch loss
        test_loss += F.cross_entropy(log_probs, target, reduction='sum').item()
        # get the index of the max log-probability
        y_pred = log_probs.data.max(1, keepdim=True)[1]
        correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()

    test_loss /= len(data_loader.dataset)
    accuracy = 100.00 * correct / len(data_loader.dataset)
    if args.verbose:
        print('\nTest set: Average loss: {:.4f} \nAccuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, correct, len(data_loader.dataset), accuracy))
    return accuracy, test_loss




def test_img1213212(net_g, datatest, args):
    net_g.eval()
    # testing
    test_loss = 0
    correct = 0
    total = 0
    data_loader = DataLoader(datatest, batch_size=args.bs)
    l = len(data_loader)
    for idx, (data, target) in enumerate(data_loader):
        if args.gpu != -1:
            data, target = data.cuda(), target.cuda()
        log_probs = net_g(data)
        # sum up batch loss
        # test_loss += F.cross_entropy(log_probs, target, reduction='sum').item()
        # get the index of the max log-probability
        # y_pred = log_probs.data.max(1, keepdim=True)[1]
        y_pred = torch.round(torch.sigmoid(log_probs.squeeze()))
        total += target.size(0)
        correct += (y_pred == target).sum().item()
        # correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()

    # test_loss /= len(data_loader.dataset)
    accuracy = 100.00 * correct / total
    if args.verbose:
        print('\nTest set: Average loss: {:.4f} \nAccuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, correct, len(data_loader.dataset), accuracy))
    return accuracy, test_loss