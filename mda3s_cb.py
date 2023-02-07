import argparse
import os, sys
import os.path as osp
import torchvision
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import network, loss
from torch.utils.data import DataLoader
from data_list import ImageList
import random, pdb, math, copy
from tqdm import tqdm
from loss import CrossEntropyLabelSmooth_cb
from scipy.spatial.distance import cdist
from sklearn.metrics import confusion_matrix
from sklearn.cluster import KMeans

def op_copy(optimizer):
    for param_group in optimizer.param_groups:
        param_group['lr0'] = param_group['lr']
    return optimizer

def lr_scheduler(optimizer, iter_num, max_iter, gamma=10, power=0.75):
    decay = (1 + gamma * iter_num / max_iter) ** (-power)
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr0'] * decay
        param_group['weight_decay'] = 1e-3
        param_group['momentum'] = 0.9
        param_group['nesterov'] = True
    return optimizer

def image_train(resize_size=256, crop_size=224, alexnet=False):
  if not alexnet:
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
  else:
    normalize = Normalize(meanfile='./ilsvrc_2012_mean.npy')
  return  transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.RandomCrop(crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])

def image_test(resize_size=256, crop_size=224, alexnet=False):
  if not alexnet:
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
  else:
    normalize = Normalize(meanfile='./ilsvrc_2012_mean.npy')
  return  transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        normalize
    ])

def data_load(args): 
    ## prepare data
    dsets = {}
    dset_loaders = {}
    train_bs = args.batch_size
    txt_src1 = open(args.s1_dset_path).readlines()
    txt_src2 = open(args.s2_dset_path).readlines()
    txt_src3 = open(args.s3_dset_path).readlines()
    txt_test = open(args.test_dset_path).readlines()

    if not args.da == 'uda':
        label_map_s = {}
        for i in range(len(args.src_classes)):
            label_map_s[args.src_classes[i]] = i
        
        new_src1 = []
        for i in range(len(txt_src1)):
            rec = txt_src1[i]
            reci = rec.strip().split(' ')
            if int(reci[1]) in args.src_classes:
                line = reci[0] + ' ' + str(label_map_s[int(reci[1])]) + '\n'   
                new_src1.append(line)
        txt_src1 = new_src1.copy()
        
        new_src2 = []
        for i in range(len(txt_src2)):
            rec = txt_src2[i]
            reci = rec.strip().split(' ')
            if int(reci[1]) in args.src_classes:
                line = reci[0] + ' ' + str(label_map_s[int(reci[1])]) + '\n'   
                new_src2.append(line)
        txt_src2 = new_src2.copy()
        
        new_src3 = []
        for i in range(len(txt_src3)):
            rec = txt_src3[i]
            reci = rec.strip().split(' ')
            if int(reci[1]) in args.src_classes:
                line = reci[0] + ' ' + str(label_map_s[int(reci[1])]) + '\n'   
                new_src3.append(line)
        txt_src3 = new_src3.copy()
        
        
        new_tar = []
        for i in range(len(txt_test)):
            rec = txt_test[i]
            reci = rec.strip().split(' ')
            if int(reci[1]) in args.tar_classes:
                if int(reci[1]) in args.src_classes:
                    line = reci[0] + ' ' + str(label_map_s[int(reci[1])]) + '\n'   
                    new_tar.append(line)
                else:
                    line = reci[0] + ' ' + str(len(label_map_s)) + '\n'   
                    new_tar.append(line)
        txt_test = new_tar.copy()

    if args.trte == "val":
        dsize = len(txt_src1)
        tr_size = int(0.9*dsize)
        # print(dsize, tr_size, dsize - tr_size)
        tr_txt1, te_txt1 = torch.utils.data.random_split(txt_src1, [tr_size, dsize - tr_size])
        dsize = len(txt_src2)
        tr_size = int(0.9*dsize)
        # print(dsize, tr_size, dsize - tr_size)
        tr_txt2, te_txt2 = torch.utils.data.random_split(txt_src2, [tr_size, dsize - tr_size])
        dsize = len(txt_src3)
        tr_size = int(0.9*dsize)
        # print(dsize, tr_size, dsize - tr_size)
        tr_txt3, te_txt3 = torch.utils.data.random_split(txt_src3, [tr_size, dsize - tr_size])
    else:
        dsize = len(txt_src1)
        tr_size = int(0.9*dsize)
        _, te_txt1 = torch.utils.data.random_split(txt_src1, [tr_size, dsize - tr_size])
        tr_txt1 = txt_src1
        dsize = len(txt_src2)
        tr_size = int(0.9*dsize)
        _, te_txt2 = torch.utils.data.random_split(txt_src2, [tr_size, dsize - tr_size])
        tr_txt2 = txt_src2
        dsize = len(txt_src3)
        tr_size = int(0.9*dsize)
        _, te_txt3 = torch.utils.data.random_split(txt_src3, [tr_size, dsize - tr_size])
        tr_txt3 = txt_src3

    dsets["source1_tr"] = ImageList(tr_txt1, transform=image_train())
    dset_loaders["source1_tr"] = DataLoader(dsets["source1_tr"], batch_size=train_bs, shuffle=True, num_workers=args.worker, drop_last=False)
    dsets["source1_te"] = ImageList(te_txt1, transform=image_test())
    dset_loaders["source1_te"] = DataLoader(dsets["source1_te"], batch_size=train_bs, shuffle=True, num_workers=args.worker, drop_last=False)
    
    dsets["source2_tr"] = ImageList(tr_txt2, transform=image_train())
    dset_loaders["source2_tr"] = DataLoader(dsets["source2_tr"], batch_size=train_bs, shuffle=True, num_workers=args.worker, drop_last=False)
    dsets["source2_te"] = ImageList(te_txt2, transform=image_test())
    dset_loaders["source2_te"] = DataLoader(dsets["source2_te"], batch_size=train_bs, shuffle=True, num_workers=args.worker, drop_last=False)
    
    dsets["source3_tr"] = ImageList(tr_txt3, transform=image_train())
    dset_loaders["source3_tr"] = DataLoader(dsets["source3_tr"], batch_size=train_bs, shuffle=True, num_workers=args.worker, drop_last=False)
    dsets["source3_te"] = ImageList(te_txt3, transform=image_test())
    dset_loaders["source3_te"] = DataLoader(dsets["source3_te"], batch_size=train_bs, shuffle=True, num_workers=args.worker, drop_last=False)
    
    
    dsets["test"] = ImageList(txt_test, transform=image_test())
    dset_loaders["test"] = DataLoader(dsets["test"], batch_size=train_bs*2, shuffle=True, num_workers=args.worker, drop_last=False)

    return dset_loaders

def data_load_src(args, source_name): 
    ## prepare data
    dsets = {}
    dset_loaders = {}
    train_bs = args.batch_size
    txt_src = open(source_name).readlines()

    if not args.da == 'uda':
        label_map_s = {}
        for i in range(len(args.src_classes)):
            label_map_s[args.src_classes[i]] = i
        
        new_src = []
        for i in range(len(txt_src)):
            rec = txt_src[i]
            reci = rec.strip().split(' ')
            if int(reci[1]) in args.src_classes:
                line = reci[0] + ' ' + str(label_map_s[int(reci[1])]) + '\n'   
                new_src.append(line)
        txt_src = new_src.copy()
        
    if args.trte == "val":
        dsize = len(txt_src)
        tr_size = int(0.9*dsize)
        # print(dsize, tr_size, dsize - tr_size)
        tr_txt, te_txt = torch.utils.data.random_split(txt_src, [tr_size, dsize - tr_size])
    else:
        dsize = len(txt_src)
        tr_size = int(0.9*dsize)
        _, te_txt = torch.utils.data.random_split(txt_src, [tr_size, dsize - tr_size])
        tr_txt = txt_src

    dsets["source_tr"] = ImageList(tr_txt, transform=image_train())
    dset_loaders["source_tr"] = DataLoader(dsets["source_tr"], batch_size=train_bs, shuffle=True, num_workers=args.worker, drop_last=False)
    dsets["source_te"] = ImageList(te_txt, transform=image_test())
    dset_loaders["source_te"] = DataLoader(dsets["source_te"], batch_size=train_bs, shuffle=True, num_workers=args.worker, drop_last=False)

    return dset_loaders

def data_load_tst(args): 
    ## prepare data
    dsets = {}
    dset_loaders = {}
    train_bs = args.batch_size
    txt_test = open(args.test_dset_path).readlines()

    if not args.da == 'uda':
        label_map_s = {}
        for i in range(len(args.src_classes)):
            label_map_s[args.src_classes[i]] = i       
        
        new_tar = []
        for i in range(len(txt_test)):
            rec = txt_test[i]
            reci = rec.strip().split(' ')
            if int(reci[1]) in args.tar_classes:
                if int(reci[1]) in args.src_classes:
                    line = reci[0] + ' ' + str(label_map_s[int(reci[1])]) + '\n'   
                    new_tar.append(line)
                else:
                    line = reci[0] + ' ' + str(len(label_map_s)) + '\n'   
                    new_tar.append(line)
        txt_test = new_tar.copy()
    
    dsets["test"] = ImageList(txt_test, transform=image_test())
    dset_loaders["test"] = DataLoader(dsets["test"], batch_size=train_bs*2, shuffle=True, num_workers=args.worker, drop_last=False)

    return dset_loaders

def cal_globe(loader, netF, netB1, netC1, netB2, netC2, netB3, netC3, netG, flag=False):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = iter_test.next()
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            outputs1 = netC1(netB1(netF(inputs)))
            outputs2 = netC2(netB2(netF(inputs)))
            outputs3 = netC3(netB3(netF(inputs)))
            outputsg = torch.stack([outputs1, outputs2, outputs3], 2)
            del outputs1
            del outputs2
            del outputs3
            outputsg = netG(outputsg)
            outputsg = torch.squeeze(outputsg,2)
            if start_test:
                all_outputg = outputsg.float().cpu()
                all_label = labels.float()
                start_test = False
            else:              
                all_outputg = torch.cat((all_outputg, outputsg.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)
    
    all_outputg = nn.Softmax(dim=1)(all_outputg)
    _, predictg = torch.max(all_outputg, 1)
    accuracyg = torch.sum(torch.squeeze(predictg).float() == all_label).item() / float(all_label.size()[0])
    mean_entg = torch.mean(loss.Entropy(all_outputg)).cpu().data.item()
   
   
    if flag:
        matrix = confusion_matrix(all_label, torch.squeeze(predict).float())
        acc = matrix.diagonal()/matrix.sum(axis=1) * 100
        aacc = acc.mean()
        aa = [str(np.round(i, 2)) for i in acc]
        acc = ' '.join(aa)
        return aacc, acc
    else:
        return accuracyg*100, mean_entg


def cal_acc(loader, netF, netB, netC, flag=False):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = iter_test.next()
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            outputs = netC(netB(netF(inputs)))
            if start_test:
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)

    all_output = nn.Softmax(dim=1)(all_output)
    _, predict = torch.max(all_output, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    mean_ent = torch.mean(loss.Entropy(all_output)).cpu().data.item()
   
    if flag:
        matrix = confusion_matrix(all_label, torch.squeeze(predict).float())
        acc = matrix.diagonal()/matrix.sum(axis=1) * 100
        aacc = acc.mean()
        aa = [str(np.round(i, 2)) for i in acc]
        acc = ' '.join(aa)
        return aacc, acc
    else:
        return accuracy*100, mean_ent


def cal_globe_oda(loader, netF, netB1, netC1, netB2, netC2, netB3, netC3, netG, args):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = iter_test.next()
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            
            outputs1 = netC1(netB1(netF(inputs)))
            outputs2 = netC2(netB2(netF(inputs)))
            outputs3 = netC3(netB3(netF(inputs)))
            outputsg = torch.stack([outputs1, outputs2, outputs3], 2)
            del outputs1
            del outputs2
            del outputs3
            outputsg = netG(outputsg)
            outputsg = torch.squeeze(outputsg,2)
            if start_test:
                all_outputg = outputsg.float().cpu()
                all_label = labels.float()
                start_test = False
            else:              
                all_outputg = torch.cat((all_outputg, outputsg.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)
    
    all_outputg = nn.Softmax(dim=1)(all_outputg)
    _, predict = torch.max(all_outputg, 1)
    ent = torch.sum(-all_outputg * torch.log(all_outputg + args.epsilon), dim=1) / np.log(args.class_num)
    ent = ent.float().cpu()
    initc = np.array([[0], [1]])
    kmeans = KMeans(n_clusters=2, random_state=0, init=initc, n_init=1).fit(ent.reshape(-1,1))
    threshold = (kmeans.cluster_centers_).mean()

    predict[ent>threshold] = args.class_num
    matrix = confusion_matrix(all_label, torch.squeeze(predict).float())
    matrix = matrix[np.unique(all_label).astype(int),:]

    acc = matrix.diagonal()/matrix.sum(axis=1) * 100
    unknown_acc = acc[-1:].item()

    return np.mean(acc[:-1]), np.mean(acc), unknown_acc

def cal_acc_oda(loader, netF, netB, netC):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = iter_test.next()
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            outputs = netC(netB(netF(inputs)))
            if start_test:
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)

    all_output = nn.Softmax(dim=1)(all_output)
    _, predict = torch.max(all_output, 1)
    ent = torch.sum(-all_output * torch.log(all_output + args.epsilon), dim=1) / np.log(args.class_num)
    ent = ent.float().cpu()
    initc = np.array([[0], [1]])
    kmeans = KMeans(n_clusters=2, random_state=0, init=initc, n_init=1).fit(ent.reshape(-1,1))
    threshold = (kmeans.cluster_centers_).mean()

    predict[ent>threshold] = args.class_num
    matrix = confusion_matrix(all_label, torch.squeeze(predict).float())
    matrix = matrix[np.unique(all_label).astype(int),:]

    acc = matrix.diagonal()/matrix.sum(axis=1) * 100
    unknown_acc = acc[-1:].item()

    return np.mean(acc[:-1]), np.mean(acc), unknown_acc
    # return np.mean(acc), np.mean(acc[:-1])

def obtain_cls(loader, args):
    start_count = True
    with torch.no_grad():
        iter_count = iter(loader)
        for _ in range(len(loader)):
            data = iter_count.next()
            labels = data[1] # ground truth           
            if start_count:
                all_label = labels.float().int()
                start_count = False
            else:
                all_label = torch.cat((all_label, labels.float().int()), 0)

    predict = all_label.cpu()

    K = args.class_num
    cls_count = np.eye(K)[predict].sum(axis=0)
    cls_count = torch.from_numpy(cls_count)

    return cls_count


def train_source(args):
    #dset_loaders = data_load(args)
    dset_loaders_s1 = data_load_src(args, args.s1_dset_path)
    dset_loaders_s2 = data_load_src(args, args.s2_dset_path)
    dset_loaders_s3 = data_load_src(args, args.s3_dset_path)
    dset_loaders = data_load_tst(args)
    ## set base network
    if args.net[0:3] == 'res':
        netF = network.ResBase(res_name=args.net).cuda()
    elif args.net[0:3] == 'vgg':
        netF = network.VGGBase(vgg_name=args.net).cuda()  

    netB1 = network.feat_bootleneck(type=args.classifier, feature_dim=netF.in_features, bottleneck_dim=args.bottleneck).cuda()
    netC1 = network.feat_classifier(type=args.layer, class_num = args.class_num, bottleneck_dim=args.bottleneck).cuda()
    netB2 = network.feat_bootleneck(type=args.classifier, feature_dim=netF.in_features, bottleneck_dim=args.bottleneck).cuda()
    netC2 = network.feat_classifier(type=args.layer, class_num = args.class_num, bottleneck_dim=args.bottleneck).cuda()
    netB3 = network.feat_bootleneck(type=args.classifier, feature_dim=netF.in_features, bottleneck_dim=args.bottleneck).cuda()
    netC3 = network.feat_classifier(type=args.layer, class_num = args.class_num, bottleneck_dim=args.bottleneck).cuda()
        
    netG = nn.Linear(args.sk,1).cuda()

    param_group = []
    learning_rate = args.lr
    for k, v in netF.named_parameters():
        param_group += [{'params': v, 'lr': learning_rate*0.1}]
    for k, v in netB1.named_parameters():
        param_group += [{'params': v, 'lr': learning_rate}]
    for k, v in netC1.named_parameters():
        param_group += [{'params': v, 'lr': learning_rate}]   
    for k, v in netB2.named_parameters():
        param_group += [{'params': v, 'lr': learning_rate}]
    for k, v in netC2.named_parameters():
        param_group += [{'params': v, 'lr': learning_rate}]
    for k, v in netB3.named_parameters():
        param_group += [{'params': v, 'lr': learning_rate}]
    for k, v in netC3.named_parameters():
        param_group += [{'params': v, 'lr': learning_rate}]
    for k, v in netG.named_parameters():
        param_group += [{'params': v, 'lr': learning_rate}]  
    optimizer = optim.SGD(param_group)
    optimizer = op_copy(optimizer)

    acc_init = 0
    max_iter = args.max_epoch * max(len(dset_loaders_s1["source_tr"]),len(dset_loaders_s2["source_tr"]),len(dset_loaders_s3["source_tr"]))
    interval_iter = max_iter // 10
    iter_num = 0

    netF.train()
    netB1.train()
    netC1.train()
    netB2.train()
    netC2.train()
    netB3.train()
    netC3.train()
    netG.train()
    
    samples_per_cls1 = obtain_cls(dset_loaders_s1["source_tr"], args)
    samples_per_cls2 = obtain_cls(dset_loaders_s2["source_tr"], args)
    samples_per_cls3 = obtain_cls(dset_loaders_s3["source_tr"], args)
    s_num = [sum(samples_per_cls1), sum(samples_per_cls2), sum(samples_per_cls3)]
    s_num = np.array(s_num)
    
    effective_num = 1.0 - np.power(args.beta, s_num)
    weights = (1.0 - args.beta) / np.array(effective_num)
    weights = weights / np.sum(weights)# * s_num

    while iter_num < max_iter:
        try:
            inputs_source1, labels_source1 = iter_source1.next()
        except:
            iter_source1 = iter(dset_loaders_s1["source_tr"])
            inputs_source1, labels_source1 = iter_source1.next()

        try:
            inputs_source2, labels_source2 = iter_source2.next()
        except:
            iter_source2 = iter(dset_loaders_s2["source_tr"])
            inputs_source2, labels_source2 = iter_source2.next()
        
        try:
            inputs_source3, labels_source3 = iter_source3.next()
        except:
            iter_source3 = iter(dset_loaders_s3["source_tr"])
            inputs_source3, labels_source3 = iter_source3.next()
        
        if inputs_source1.size(0) == 1 or inputs_source2.size(0) == 1 or inputs_source3.size(0) == 1:
            continue

        iter_num += 1
        lr_scheduler(optimizer, iter_num=iter_num, max_iter=max_iter)

        inputs_source1, labels_source1 = inputs_source1.cuda(), labels_source1.cuda()
        inputs_source2, labels_source2 = inputs_source2.cuda(), labels_source2.cuda()
        inputs_source3, labels_source3 = inputs_source3.cuda(), labels_source3.cuda()
        
        outputs_source = netC1(netB1(netF(inputs_source1)))
        outputs_source_m1 = netC2(netB2(netF(inputs_source1)))  
        outputs_source_m2 = netC3(netB3(netF(inputs_source1)))       
        #classifier_loss1 = CrossEntropyLabelSmooth(num_classes=args.class_num, epsilon=args.smooth)(outputs_source, labels_source1)
        classifier_loss1 = CrossEntropyLabelSmooth_cb(num_classes=args.class_num, epsilon=args.smooth)(outputs_source, labels_source1, samples_per_cls1, args.beta, args.gamma)       
        outputs_sourceg = torch.stack([outputs_source, outputs_source_m1, outputs_source_m2], 2)
        outputs_sourceg = netG(outputs_sourceg)
        outputs_sourceg = torch.squeeze(outputs_sourceg,2)
        classifier_loss1 += CrossEntropyLabelSmooth_cb(num_classes=args.class_num, epsilon=args.smooth)(outputs_sourceg, labels_source1, samples_per_cls1, args.beta, args.gamma)
        
        
        outputs_source = netC2(netB2(netF(inputs_source2)))
        outputs_source_m1 = netC1(netB1(netF(inputs_source2)))
        outputs_source_m2 = netC3(netB3(netF(inputs_source2)))
        classifier_loss2 = CrossEntropyLabelSmooth_cb(num_classes=args.class_num, epsilon=args.smooth)(outputs_source, labels_source2, samples_per_cls2, args.beta, args.gamma)  
        outputs_sourceg = torch.stack([outputs_source_m1, outputs_source, outputs_source_m2], 2)
        outputs_sourceg = netG(outputs_sourceg)
        outputs_sourceg = torch.squeeze(outputs_sourceg,2)
        classifier_loss2 += CrossEntropyLabelSmooth_cb(num_classes=args.class_num, epsilon=args.smooth)(outputs_sourceg, labels_source2, samples_per_cls2, args.beta, args.gamma)
        
        outputs_source = netC3(netB3(netF(inputs_source3)))
        outputs_source_m1 = netC1(netB1(netF(inputs_source3)))
        outputs_source_m2 = netC2(netB2(netF(inputs_source3)))
        classifier_loss3 = CrossEntropyLabelSmooth_cb(num_classes=args.class_num, epsilon=args.smooth)(outputs_source, labels_source3, samples_per_cls3, args.beta, args.gamma)  
        outputs_sourceg = torch.stack([outputs_source_m1, outputs_source_m2, outputs_source], 2)
        outputs_sourceg = netG(outputs_sourceg)
        outputs_sourceg = torch.squeeze(outputs_sourceg,2)
        classifier_loss3 += CrossEntropyLabelSmooth_cb(num_classes=args.class_num, epsilon=args.smooth)(outputs_sourceg, labels_source3, samples_per_cls3, args.beta, args.gamma)
                       
        optimizer.zero_grad()
        classifier_loss1.backward()
        classifier_loss2.backward()
        classifier_loss3.backward()
        optimizer.step()

        if iter_num % interval_iter == 0 or iter_num == max_iter:
            netF.eval()
            netB1.eval()
            netC1.eval()
            netB2.eval()
            netC2.eval()
            netB3.eval()
            netC3.eval()
            netG.eval()

            if args.da == 'oda':
                acc_s1_te, _ = cal_acc(dset_loaders_s1['source_te'], netF, netB1, netC1, False)
                acc_os1, acc_os2, acc_unknown = cal_acc_oda(dset_loaders['test'], netF, netB1, netC1)
                log_str1 = 'Source: {}, Acc-src = {:.2f}%, Accuracy = {:.2f}% / {:.2f}% / {:.2f}%'.format(args.name_src1, acc_s1_te, acc_os2, acc_os1, acc_unknown)
  
                acc_s2_te, _ = cal_acc(dset_loaders_s2['source_te'], netF, netB2, netC2, False)                
                acc_os1, acc_os2, acc_unknown = cal_acc_oda(dset_loaders['test'], netF, netB2, netC2)
                log_str2 = 'Source: {}, Acc-src = {:.2f}%, Accuracy = {:.2f}% / {:.2f}% / {:.2f}%'.format(args.name_src2, acc_s2_te, acc_os2, acc_os1, acc_unknown)
                
                acc_s3_te, _ = cal_acc(dset_loaders_s3['source_te'], netF, netB3, netC3, False)                
                acc_os1, acc_os2, acc_unknown = cal_acc_oda(dset_loaders['test'], netF, netB3, netC3)
                log_str3 = 'Source: {}, Acc-src = {:.2f}%, Accuracy = {:.2f}% / {:.2f}% / {:.2f}%'.format(args.name_src3, acc_s3_te, acc_os2, acc_os1, acc_unknown)
                
                acc_os1, acc_os2, acc_unknown = cal_globe_oda(dset_loaders['test'], netF, netB1, netC1, netB2, netC2, netB3, netC3, netG, args)                
                log_strg = '\Task: {}, Iter:{}/{}; Accuracy = {:.2f}% / {:.2f}% / {:.2f}%'.format(args.task, iter_num, max_iter, acc_os2, acc_os1, acc_unknown)
            else:
                acc_s1_te, _ = cal_acc(dset_loaders_s1['source_te'], netF, netB1, netC1, False)
                acc_g1_te, _ = cal_globe(dset_loaders_s1['source_te'], netF, netB1, netC1, netB2, netC2, netB3, netC3, netG, False)                
                log_str1 = 'Task: {}, Iter:{}/{}; Accuracy = {:.2f}%\tGlobe = {:.2f}%'.format(args.name_src1, iter_num, max_iter, acc_s1_te, acc_g1_te)
                
                acc_s2_te, _ = cal_acc(dset_loaders_s2['source_te'], netF, netB2, netC2, False)                
                acc_g2_te, _ = cal_globe(dset_loaders_s2['source_te'], netF, netB1, netC1, netB2, netC2, netB3, netC3, netG, False)
                log_str2 = 'Task: {}, Iter:{}/{}; Accuracy = {:.2f}%\tGlobe = {:.2f}%'.format(args.name_src2, iter_num, max_iter, acc_s2_te, acc_g2_te)
                
                acc_s3_te, _ = cal_acc(dset_loaders_s3['source_te'], netF, netB3, netC3, False)                
                acc_g3_te, _ = cal_globe(dset_loaders_s3['source_te'], netF, netB1, netC1, netB2, netC2, netB3, netC3, netG, False)
                log_str3 = 'Task: {}, Iter:{}/{}; Accuracy = {:.2f}%\tGlobe = {:.2f}%'.format(args.name_src3, iter_num, max_iter, acc_s3_te, acc_g3_te)
            
            
            args.out_file.write(log_str1 + '\n')
            args.out_file.flush()
            print(log_str1+'\n')
            args.out_file.write(log_str2 + '\n')
            args.out_file.flush()
            print(log_str2+'\n')
            args.out_file.write(log_str3 + '\n')
            args.out_file.flush()
            print(log_str3+'\n')

            if acc_s1_te + acc_s2_te + acc_s3_te >= acc_init:
                acc_init = acc_s1_te + acc_s2_te + acc_s3_te 
                best_netF = netF.state_dict()
                best_netB1 = netB1.state_dict()
                best_netC1 = netC1.state_dict()
                best_netB2 = netB2.state_dict()
                best_netC2 = netC2.state_dict()
                best_netB3 = netB3.state_dict()
                best_netC3 = netC3.state_dict()
                best_netG = netG.state_dict()
                
                torch.save(best_netF, osp.join(args.output_dir_src, "source_F.pt"))
                torch.save(best_netB1, osp.join(args.output_dir_src, args.name_src1+"_B.pt"))
                torch.save(best_netC1, osp.join(args.output_dir_src, args.name_src1+"_C.pt"))
                torch.save(best_netB2, osp.join(args.output_dir_src, args.name_src2+"_B.pt"))
                torch.save(best_netC2, osp.join(args.output_dir_src, args.name_src2+"_C.pt"))
                torch.save(best_netB3, osp.join(args.output_dir_src, args.name_src3+"_B.pt"))
                torch.save(best_netC3, osp.join(args.output_dir_src, args.name_src3+"_C.pt"))
                torch.save(best_netG, osp.join(args.output_dir_src, "source_G.pt"))

            netF.train()
            netB1.train()
            netC1.train()
            netB2.train()
            netC2.train()
            netB3.train()
            netC3.train()
            netG.train()


    return netF, netB1, netC1, netB2, netC2, netB3, netC3, netG
'''
def test_target(args):
    #dset_loaders = data_load(args)
    dset_loaders_t = data_load_tst(args)
    ## set base network
    if args.net[0:3] == 'res':
        netF = network.ResBase(res_name=args.net).cuda()
    elif args.net[0:3] == 'vgg':
        netF = network.VGGBase(vgg_name=args.net).cuda()  

    netB1 = network.feat_bootleneck(type=args.classifier, feature_dim=netF.in_features, bottleneck_dim=args.bottleneck).cuda()
    netC1 = network.feat_classifier(type=args.layer, class_num = args.class_num, bottleneck_dim=args.bottleneck).cuda()
    netB2 = network.feat_bootleneck(type=args.classifier, feature_dim=netF.in_features, bottleneck_dim=args.bottleneck).cuda()
    netC2 = network.feat_classifier(type=args.layer, class_num = args.class_num, bottleneck_dim=args.bottleneck).cuda()
    netB3 = network.feat_bootleneck(type=args.classifier, feature_dim=netF.in_features, bottleneck_dim=args.bottleneck).cuda()
    netC3 = network.feat_classifier(type=args.layer, class_num = args.class_num, bottleneck_dim=args.bottleneck).cuda()
    
    netG = nn.Linear(args.sk,1).cuda()

    args.modelpath = args.output_dir_src + '/source_F.pt'   
    netF.load_state_dict(torch.load(args.modelpath))
    
    args.modelpath = args.output_dir_src + '/' + args.name_src1+'_B.pt'   
    netB1.load_state_dict(torch.load(args.modelpath))
    args.modelpath = args.output_dir_src + '/' + args.name_src1+'_C.pt'   
    netC1.load_state_dict(torch.load(args.modelpath))
    
    args.modelpath = args.output_dir_src + '/' + args.name_src2+'_B.pt'   
    netB2.load_state_dict(torch.load(args.modelpath))
    args.modelpath = args.output_dir_src + '/' + args.name_src2+'_C.pt'   
    netC2.load_state_dict(torch.load(args.modelpath))
    
    args.modelpath = args.output_dir_src + '/' + args.name_src3+'_B.pt'   
    netB3.load_state_dict(torch.load(args.modelpath))
    args.modelpath = args.output_dir_src + '/' + args.name_src3+'_C.pt'   
    netC3.load_state_dict(torch.load(args.modelpath))
    
    args.modelpath = args.output_dir_src + '/source_G.pt'   
    netG.load_state_dict(torch.load(args.modelpath))
    
    netF.eval()
    netB1.eval()
    netC1.eval()
    netB2.eval()
    netC2.eval()
    netB3.eval()
    netC3.eval()
    netG.eval()

    if args.da == 'oda':
        acc_os1, acc_os2, acc_unknown = cal_acc_oda(dset_loaders['test'], netF, netB1, netC1)
        log_str = '\nTraining: {}, Task: {}, Accuracy = {:.2f}% / {:.2f}% / {:.2f}%'.format(args.trte, args.task, acc_os2, acc_os1, acc_unknown)
    else:
        if args.dset=='VISDA-C':
            acc, acc_list = cal_acc(dset_loaders_t['test'], netF, netB1, netC1, True)
            log_str = '\nTraining: {}, Task: {}, Accuracy = {:.2f}%'.format(args.trte, args.task, acc) + '\n' + acc_list
        else:
            acc1, _ = cal_acc(dset_loaders_t['test'], netF, netB1, netC1, False)            
            acc2, _ = cal_acc(dset_loaders_t['test'], netF, netB2, netC2, False)                        
            acc3, _ = cal_acc(dset_loaders_t['test'], netF, netB3, netC3, False)                                    
            accg, _ = cal_globe(dset_loaders_t['test'], netF, netB1, netC1, netB2, netC2, netB3, netC3, netG, False)
            log_str = '\nTraining: {}, Task: {}, Accuracy_s1 = {:.2f}%\tAccuracy_s2 = {:.2f}%\tAccuracy_s3 = {:.2f}%\tGlobe = {:.2f}%'.format(args.trte, args.task, acc1, acc2, acc3, accg)

    args.out_file.write(log_str)
    args.out_file.flush()
    print(log_str)
'''
def print_args(args):
    s = "==========================================\n"
    for arg, content in args.__dict__.items():
        s += "{}:{}\n".format(arg, content)
    return s

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SHOT')
    parser.add_argument('--gpu_id', type=str, nargs='?', default='1', help="device id to run")
    parser.add_argument('--s1', type=int, default=0, help="source")
    parser.add_argument('--s2', type=int, default=1, help="source")
    parser.add_argument('--s3', type=int, default=2, help="source")
    parser.add_argument('--t', type=int, default=3, help="target")
    parser.add_argument('--max_epoch', type=int, default=20, help="max iterations")
    parser.add_argument('--batch_size', type=int, default=32, help="batch_size")
    parser.add_argument('--worker', type=int, default=4, help="number of workers")
    parser.add_argument('--dset', type=str, default='OfficeHome', choices=['VISDA-C', 'office31', 'OfficeHome', 'offcal', 'CLEF', 'DomainNet'])
    parser.add_argument('--lr', type=float, default=1e-2, help="learning rate")
    parser.add_argument('--net', type=str, default='resnet50', help="vgg16, resnet50, resnet101")
    parser.add_argument('--seed', type=int, default=2020, help="random seed")
    parser.add_argument('--bottleneck', type=int, default=256)
    parser.add_argument('--epsilon', type=float, default=1e-5)
    parser.add_argument('--layer', type=str, default="wn", choices=["linear", "wn"])
    parser.add_argument('--classifier', type=str, default="bn", choices=["ori", "bn"])
    parser.add_argument('--smooth', type=float, default=0.1)   
    parser.add_argument('--output', type=str, default='san')
    parser.add_argument('--da', type=str, default='oda', choices=['uda', 'pda', 'oda'])
    parser.add_argument('--trte', type=str, default='val', choices=['full', 'val'])
    args = parser.parse_args()

    if args.dset == 'OfficeHome':
        names = ['Art', 'Clipart', 'Product', 'RealWorld']
        args.class_num = 65 
        task = [0,1,2,3] 
        #task = [0,1,3,2]
        #task = [0,2,3,1] 
        #task = [1,2,3,0]
    if args.dset == 'office31':
        names = ['amazon', 'dslr', 'webcam']
        args.class_num = 31
        #task = [0,1,2] 
        #task = [2,1,0]
        task = [0,2,1]
    if args.dset == 'VISDA-C':
        names = ['train', 'validation']
        args.class_num = 12
    if args.dset == 'offcal':
        names = ['amazon', 'caltech', 'dslr', 'webcam']
        #task = [0,1,2,3] 
        #task = [0,1,3,2]
        #task = [0,2,3,1] 
        task = [1,2,3,0]
        args.class_num = 10
    if args.dset == 'DomainNet':
        names = ['clipart','infograph','painting', 'quickdraw', 'real', 'sketch'] 
        args.class_num = 345
        #task = [1,2,3,4,5,0] 
        task = [0,2,3,4,5,1]
        #task = [0,1,3,4,5,2] 
        #task = [0,1,2,4,5,3]
        #task = [0,1,2,3,5,4]
        #task = [0,1,2,3,4,5]
    
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    '''
    SEED = args.seed
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    '''
    # torch.backends.cudnn.deterministic = True

    args.beta = 0.9999
    args.gamma = 2.0
    
    args.sk = 3
    folder = './dataset/'
    args.s1_dset_path = folder + args.dset + '/' + names[task[args.s1]] + 'List.txt'
    args.s2_dset_path = folder + args.dset + '/' + names[task[args.s2]] + 'List.txt'
    args.s3_dset_path = folder + args.dset + '/' + names[task[args.s3]] + 'List.txt'
    args.test_dset_path = folder + args.dset + '/' + names[task[args.t]] + 'List.txt'     

    if args.dset == 'OfficeHome':
        if args.da == 'pda':
            args.class_num = 65
            args.src_classes = [i for i in range(65)]
            args.tar_classes = [i for i in range(25)]
        if args.da == 'oda':
            args.class_num = 25
            args.src_classes = [i for i in range(25)]
            args.tar_classes = [i for i in range(65)]

    traepo = 'sonly_cb'
    args.task = names[task[args.s1]][0].upper() + names[task[args.s2]][0].upper() + names[task[args.s3]][0].upper() + '2' + names[task[args.t]][0].upper()
    args.output_dir_src = osp.join(args.output, args.da, args.dset, args.task, traepo)
    args.name_src = names[task[args.s1]][0].upper() + names[task[args.s2]][0].upper() + names[task[args.s3]][0].upper()
    args.name_src1 = names[task[args.s1]][0].upper()
    args.name_src2 = names[task[args.s2]][0].upper()
    args.name_src3 = names[task[args.s3]][0].upper()

    if not osp.exists(args.output_dir_src):
        os.system('mkdir -p ' + args.output_dir_src)
    if not osp.exists(args.output_dir_src):
        os.mkdir(args.output_dir_src)
    
    args.out_file = open(osp.join(args.output_dir_src, 'log.txt'), 'w')
    args.out_file.write(print_args(args)+'\n')
    args.out_file.flush()
    train_source(args)

    #args.out_file = open(osp.join(args.output_dir_src, 'log_test.txt'), 'w')

    #test_target(args)
