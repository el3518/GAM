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
from data_list import ImageList, ImageList_idx
import random, pdb, math, copy
from tqdm import tqdm
from scipy.spatial.distance import cdist
from sklearn.metrics import confusion_matrix
from loss import CB_loss_tar, CrossEntropy_cb

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
    txt_tar = open(args.t_dset_path).readlines()
    txt_test = open(args.test_dset_path).readlines()

    if not args.da == 'uda':
        label_map_s = {}
        for i in range(len(args.src_classes)):
            label_map_s[args.src_classes[i]] = i

        new_tar = []
        for i in range(len(txt_tar)):
            rec = txt_tar[i]
            reci = rec.strip().split(' ')
            if int(reci[1]) in args.tar_classes:
                if int(reci[1]) in args.src_classes:
                    line = reci[0] + ' ' + str(label_map_s[int(reci[1])]) + '\n'   
                    new_tar.append(line)
                else:
                    line = reci[0] + ' ' + str(len(label_map_s)) + '\n'   
                    new_tar.append(line)
        txt_tar = new_tar.copy()
        txt_test = txt_tar.copy()

    dsets["target"] = ImageList_idx(txt_tar, transform=image_train())
    dset_loaders["target"] = DataLoader(dsets["target"], batch_size=train_bs, shuffle=True, num_workers=args.worker, drop_last=False)
    dsets["test"] = ImageList_idx(txt_test, transform=image_test())
    dset_loaders["test"] = DataLoader(dsets["test"], batch_size=train_bs*3, shuffle=False, num_workers=args.worker, drop_last=False)

    return dset_loaders

def cal_acc(loader, netF, netB, netC, flag=False):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = iter_test.next()
            inputs = data[0]
            labels = data[1] #ground truth
            inputs = inputs.cuda()
            outputs = netC(netB(netF(inputs))) # predicted
            if start_test:
                all_output = outputs.float().cpu() # predicted
                all_label = labels.float() #ground truth
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)
    _, predict = torch.max(all_output, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    mean_ent = torch.mean(loss.Entropy(nn.Softmax(dim=1)(all_output))).cpu().data.item()

    if flag:
        matrix = confusion_matrix(all_label, torch.squeeze(predict).float())
        acc = matrix.diagonal()/matrix.sum(axis=1) * 100
        aacc = acc.mean()
        aa = [str(np.round(i, 2)) for i in acc]
        acc = ' '.join(aa)
        return aacc, acc
    else:
        return accuracy*100, mean_ent

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

def cal_globe_oda(loader, netF, netB1, netC1, netB2, netC2, netB3, netC3, netG, args, flag=False, threshold1=0.1, threshold2=0.1, threshold3=0.1):
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
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    mean_ent = torch.mean(loss.Entropy(nn.Softmax(dim=1)(all_outputg))).cpu().data.item()

    if flag:
        all_outputg = nn.Softmax(dim=1)(all_outputg)
        ent = torch.sum(-all_outputg * torch.log(all_outputg + args.epsilon), dim=1) / np.log(args.class_num)

        from sklearn.cluster import KMeans
        kmeans = KMeans(2, random_state=0).fit(ent.reshape(-1,1))
        labels = kmeans.predict(ent.reshape(-1,1))

        idx = np.where(labels==1)[0]
        iidx = 0
        if ent[idx].mean() > ent.mean():
            iidx = 1
        predict[np.where(labels==iidx)[0]] = args.class_num

        matrix = confusion_matrix(all_label, torch.squeeze(predict).float())
        matrix = matrix[np.unique(all_label).astype(int),:]

        acc = matrix.diagonal()/matrix.sum(axis=1) * 100
        unknown_acc = acc[-1:].item()
        return np.mean(acc[:-1]), np.mean(acc), unknown_acc
    else:
        return accuracy*100, mean_ent

def cal_acc_oda(loader, netF, netB, netC, flag=False, threshold=0.1):
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
    mean_ent = torch.mean(loss.Entropy(nn.Softmax(dim=1)(all_output))).cpu().data.item()

    if flag:
        all_output = nn.Softmax(dim=1)(all_output)
        ent = torch.sum(-all_output * torch.log(all_output + args.epsilon), dim=1) / np.log(args.class_num)

        from sklearn.cluster import KMeans
        kmeans = KMeans(2, random_state=0).fit(ent.reshape(-1,1))
        labels = kmeans.predict(ent.reshape(-1,1))

        idx = np.where(labels==1)[0]
        iidx = 0
        if ent[idx].mean() > ent.mean():
            iidx = 1
        predict[np.where(labels==iidx)[0]] = args.class_num

        matrix = confusion_matrix(all_label, torch.squeeze(predict).float())
        matrix = matrix[np.unique(all_label).astype(int),:]

        acc = matrix.diagonal()/matrix.sum(axis=1) * 100
        unknown_acc = acc[-1:].item()
        return np.mean(acc[:-1]), np.mean(acc), unknown_acc
    else:
        return accuracy*100, mean_ent

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

def train_target(args):
    dset_loaders = data_load(args)
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
        
    netC1.eval()
    netC2.eval()
    netC3.eval()
    
    for k, v in netC1.named_parameters():
        v.requires_grad = False
    for k, v in netC2.named_parameters():
        v.requires_grad = False   
    for k, v in netC3.named_parameters():
        v.requires_grad = False
    ########################
    #'''
    netG.eval()
    for k, v in netG.named_parameters():
        v.requires_grad = False
    #'''
    
    param_group = []
    for k, v in netF.named_parameters():
        if args.lr_decay1 > 0:
            param_group += [{'params': v, 'lr': args.lr * args.lr_decay1}]
        else:
            v.requires_grad = False
    for k, v in netB1.named_parameters():
        if args.lr_decay2 > 0:
            param_group += [{'params': v, 'lr': args.lr * args.lr_decay2}]
        else:
            v.requires_grad = False
    for k, v in netB2.named_parameters():
        if args.lr_decay2 > 0:
            param_group += [{'params': v, 'lr': args.lr * args.lr_decay2}]
        else:
            v.requires_grad = False
    for k, v in netB3.named_parameters():
        if args.lr_decay2 > 0:
            param_group += [{'params': v, 'lr': args.lr * args.lr_decay2}]
        else:
            v.requires_grad = False
    ###############################
    '''
    for k, v in netG.named_parameters():
        if args.lr_decay2 > 0:
            param_group += [{'params': v, 'lr': args.lr * args.lr_decay2}]
        else:
            v.requires_grad = False    
    '''
    
    optimizer = optim.SGD(param_group)
    optimizer = op_copy(optimizer)

    max_iter = args.max_epoch * len(dset_loaders["target"])
    interval_iter = max_iter // args.interval
    iter_num = 0
    
    #samples_per_cls = obtain_cls(dset_loaders["target"], args)
    
    while iter_num < max_iter:
        try:
            inputs_test, _, tar_idx = iter_test.next()
        except:
            iter_test = iter(dset_loaders["target"])
            inputs_test, _, tar_idx = iter_test.next()

        if inputs_test.size(0) == 1:
            continue

        if iter_num % interval_iter == 0 and args.cls_par > 0:
            netF.eval()
            netB1.eval()
            netB2.eval()
            netB3.eval()
            #netG.eval()   ###########
            if args.da == 'oda':           
                mem_label1, mem_label2, mem_label3, ENT_THRESHOLD1, ENT_THRESHOLD2, ENT_THRESHOLD3 = obtain_label(dset_loaders['test'], netF, netB1, netC1, netB2, netC2, netB3, netC3, netG, args) #
            else:
                mem_label1, mem_label2, mem_label3 = obtain_label(dset_loaders['test'], netF, netB1, netC1, netB2, netC2, netB3, netC3, netG, args) ##pseudo
            samples_per_cls = torch.from_numpy(np.eye(args.class_num)[mem_label1].sum(axis=0))
            mem_label1 = torch.from_numpy(mem_label1).cuda()
            mem_label2 = torch.from_numpy(mem_label2).cuda()
            mem_label3 = torch.from_numpy(mem_label3).cuda()
            netF.train()
            netB1.train()
            netB2.train()
            netB3.train()
            #netG.train()   ############

        inputs_test = inputs_test.cuda()

        iter_num += 1
        lr_scheduler(optimizer, iter_num=iter_num, max_iter=max_iter)

        #features_test = netB1(netF(inputs_test))
        outputs_test1 = netC1(netB1(netF(inputs_test)))
        #features_test = netB2(netF(inputs_test))
        outputs_test2 = netC2(netB2(netF(inputs_test)))
        outputs_test3 = netC3(netB3(netF(inputs_test)))
        outputs_test = torch.stack([outputs_test1, outputs_test2, outputs_test3], 2)
        outputs_test = netG(outputs_test)
        outputs_test = torch.squeeze(outputs_test,2)

        
        if args.da == 'oda':
            pred = mem_label1[tar_idx]

            softmax_out = nn.Softmax(dim=1)(outputs_test)
            outputs_known = outputs_test[pred < args.class_num, :]
            pred = pred[pred < args.class_num]

            if len(pred) != 0:
                classifier_loss1 = nn.CrossEntropyLoss()(outputs_known, pred)
                classifier_loss1 *= args.cls_par
            else:
                classifier_loss1 = torch.tensor(0.0).cuda()

            if args.ent:
                softmax_out_known = nn.Softmax(dim=1)(outputs_known)
                entropy_loss = torch.mean(loss.Entropy(softmax_out_known))
            if args.gent:
                msoftmax = softmax_out.mean(dim=0)
                gentropy_loss = torch.sum(-msoftmax * torch.log(msoftmax + args.epsilon))
                entropy_loss -= gentropy_loss
                classifier_loss1 += entropy_loss * args.ent_par
            
            pred = mem_label2[tar_idx]

            softmax_out = nn.Softmax(dim=1)(outputs_test)
            outputs_known = outputs_test[pred < args.class_num, :]
            pred = pred[pred < args.class_num]

            if len(pred) != 0:
                classifier_loss2 = nn.CrossEntropyLoss()(outputs_known, pred)
                classifier_loss2 *= args.cls_par
            else:
                classifier_loss2 = torch.tensor(0.0).cuda()

            if args.ent:
                softmax_out_known = nn.Softmax(dim=1)(outputs_known)
                entropy_loss = torch.mean(loss.Entropy(softmax_out_known))
            if args.gent:
                msoftmax = softmax_out.mean(dim=0)
                gentropy_loss = torch.sum(-msoftmax * torch.log(msoftmax + args.epsilon))
                entropy_loss -= gentropy_loss
                classifier_loss2 += entropy_loss * args.ent_par
            
            pred = mem_label3[tar_idx]

            softmax_out = nn.Softmax(dim=1)(outputs_test)
            outputs_known = outputs_test[pred < args.class_num, :]
            pred = pred[pred < args.class_num]

            if len(pred) != 0:
                classifier_loss3 = nn.CrossEntropyLoss()(outputs_known, pred)
                classifier_loss3 *= args.cls_par
            else:
                classifier_loss3 = torch.tensor(0.0).cuda()

            if args.ent:
                softmax_out_known = nn.Softmax(dim=1)(outputs_known)
                entropy_loss = torch.mean(loss.Entropy(softmax_out_known))
            if args.gent:
                msoftmax = softmax_out.mean(dim=0)
                gentropy_loss = torch.sum(-msoftmax * torch.log(msoftmax + args.epsilon))
                entropy_loss -= gentropy_loss
                classifier_loss3 += entropy_loss * args.ent_par      
        else:
            if args.cls_par > 0:
                pred1 = mem_label1[tar_idx]
                #classifier_loss1 = nn.CrossEntropyLoss()(outputs_test, pred1)
                classifier_loss1 = CB_loss_tar(pred1, outputs_test, samples_per_cls, args.class_num, args.beta, args.gamma)                       
                #classifier_loss1 = CrossEntropy_cb(num_classes=args.class_num, epsilon=args.smooth)(outputs_test, pred1, samples_per_cls, args.beta, args.gamma)       
        
                classifier_loss1 *= args.cls_par
                pred2 = mem_label2[tar_idx]
                #classifier_loss2 = nn.CrossEntropyLoss()(outputs_test, pred2)
                classifier_loss2 = CB_loss_tar(pred2, outputs_test, samples_per_cls, args.class_num, args.beta, args.gamma)                      
                #classifier_loss2 = CrossEntropy_cb(num_classes=args.class_num, epsilon=args.smooth)(outputs_test, pred2, samples_per_cls, args.beta, args.gamma)       
        
                classifier_loss2 *= args.cls_par
                pred3 = mem_label3[tar_idx]           
                #classifier_loss3 = nn.CrossEntropyLoss()(outputs_test, pred3)
                classifier_loss3 = CB_loss_tar(pred3, outputs_test, samples_per_cls, args.class_num, args.beta, args.gamma)                       
                #classifier_loss3 = CrossEntropy_cb(num_classes=args.class_num, epsilon=args.smooth)(outputs_test, pred3, samples_per_cls, args.beta, args.gamma)       
        
                classifier_loss3 *= args.cls_par
                if iter_num < interval_iter and args.dset == "VISDA-C":
                    classifier_loss1 *= 0
                    classifier_loss2 *= 0
                    classifier_loss3 *= 0
            else:
                classifier_loss1 = torch.tensor(0.0).cuda()
                classifier_loss2 = torch.tensor(0.0).cuda()
                classifier_loss3 = torch.tensor(0.0).cuda()

            if args.ent:
                softmax_out = nn.Softmax(dim=1)(outputs_test)
                entropy_loss = torch.mean(loss.Entropy(softmax_out))
                if args.gent:
                    msoftmax = softmax_out.mean(dim=0)
                    gentropy_loss = torch.sum(-msoftmax * torch.log(msoftmax + args.epsilon))
                    entropy_loss -= gentropy_loss
                im_loss = entropy_loss * args.ent_par
                classifier_loss1 += im_loss
                classifier_loss2 += im_loss
                classifier_loss3 += im_loss

        classifier_loss = classifier_loss1 + classifier_loss2 + classifier_loss3
        optimizer.zero_grad()
        classifier_loss.backward()
        #classifier_loss2.backward()
        optimizer.step()

        if iter_num % interval_iter == 0 or iter_num == max_iter:
            netF.eval()
            netB1.eval()
            netC1.eval()
            netB2.eval()
            netC2.eval()
            netB3.eval()
            netC3.eval()
            netG.eval()  ############
            if args.da == 'oda':
                acc_os1, acc_os2, acc_unknown = cal_acc_oda(dset_loaders['test'], netF, netB1, netC1, True, ENT_THRESHOLD1)
                log_str1 = 'Source: {}, Accuracy = {:.2f}% / {:.2f}% / {:.2f}%'.format(args.name_src1, acc_os2, acc_os1, acc_unknown)
  
                acc_os1, acc_os2, acc_unknown = cal_acc_oda(dset_loaders['test'], netF, netB2, netC2, True, ENT_THRESHOLD2)
                log_str2 = 'Source: {}, Accuracy = {:.2f}% / {:.2f}% / {:.2f}%'.format(args.name_src2, acc_os2, acc_os1, acc_unknown)
                
                acc_os1, acc_os2, acc_unknown = cal_acc_oda(dset_loaders['test'], netF, netB3, netC3, True, ENT_THRESHOLD3)
                log_str3 = 'Source: {}, Accuracy = {:.2f}% / {:.2f}% / {:.2f}%'.format(args.name_src3, acc_os2, acc_os1, acc_unknown)
                
                acc_os1, acc_os2, acc_unknown = cal_globe_oda(dset_loaders['test'], netF, netB1, netC1, netB2, netC2, netB3, netC3, netG, args, True, ENT_THRESHOLD1, ENT_THRESHOLD2, ENT_THRESHOLD3)                
                log_str = '\Task: {}, Iter:{}/{}; Accuracy = {:.2f}% / {:.2f}% / {:.2f}%'.format(args.task, iter_num, max_iter, acc_os2, acc_os1, acc_unknown)
            else:
                acc1, _  = cal_acc(dset_loaders['test'], netF, netB1, netC1, False)
                acc2, _  = cal_acc(dset_loaders['test'], netF, netB2, netC2, False)
                acc3, _  = cal_acc(dset_loaders['test'], netF, netB3, netC3, False)

                accg, _ = cal_globe(dset_loaders['test'], netF, netB1, netC1, netB2, netC2, netB3, netC3, netG, False)               
                log_str = 'Task: {}, Iter:{}/{}; Accuracy_s1 = {:.2f}%\tAccuracy_s2 = {:.2f}%\tAccuracy_s3 = {:.2f}%\tGlobe = {:.2f}%'.format(args.task, iter_num, max_iter, acc1, acc2, acc3, accg)

            
            if args.issave:   
                torch.save(netF.state_dict(), osp.join(args.output_dir, "target_F_" + args.savename + ".pt"))
                torch.save(netB1.state_dict(), osp.join(args.output_dir, "target_B1_" + args.savename + ".pt"))
                torch.save(netC1.state_dict(), osp.join(args.output_dir, "target_C1_" + args.savename + ".pt"))
                torch.save(netB2.state_dict(), osp.join(args.output_dir, "target_B2_" + args.savename + ".pt"))
                torch.save(netC2.state_dict(), osp.join(args.output_dir, "target_C2_" + args.savename + ".pt"))
                torch.save(netB3.state_dict(), osp.join(args.output_dir, "target_B3_" + args.savename + ".pt"))
                torch.save(netC3.state_dict(), osp.join(args.output_dir, "target_C3_" + args.savename + ".pt"))
        
                torch.save(netG.state_dict(), osp.join(args.output_dir, "target_G_" + args.savename + ".pt"))
        
            
            args.out_file.write(log_str + '\n')
            args.out_file.flush()
            print(log_str+'\n')
            netF.train()
            netB1.train()
            netB2.train()
            netB3.train()
            #netG.train() ########

    
        
    return netF, netB1, netC1, netB2, netC2, netB3, netC3, netG

def print_args(args):
    s = "==========================================\n"
    for arg, content in args.__dict__.items():
        s += "{}:{}\n".format(arg, content)
    return s

def cal_fea_oda(all_fea, all_output, predict, all_label, ent):
    if args.distance == 'cosine':
            all_fea = torch.cat((all_fea, torch.ones(all_fea.size(0), 1)), 1)
            all_fea = (all_fea.t() / torch.norm(all_fea, p=2, dim=1)).t()
        
    from sklearn.cluster import KMeans
    kmeans = KMeans(2, random_state=0).fit(ent.reshape(-1,1))
    labels = kmeans.predict(ent.reshape(-1,1))

    idx = np.where(labels==1)[0]
    iidx = 0
    if ent[idx].mean() > ent.mean():
        iidx = 1
    known_idx = np.where(kmeans.labels_ != iidx)[0]

    all_fea = all_fea[known_idx,:]
    all_output = all_output[known_idx,:]
    predict = predict[known_idx]
    all_label_idx = all_label[known_idx]
    ENT_THRESHOLD = (kmeans.cluster_centers_).mean()
 
    all_fea = all_fea.float().cpu().numpy()
        
    return all_fea, all_output, predict, known_idx, ENT_THRESHOLD

def cal_lab_oda(all_label, pred_label, known_idx, ENT_THRESHOLD):
    guess_label = args.class_num * np.ones(len(all_label), )
    guess_label[known_idx] = pred_label    
        
    return guess_label

def obtain_label(loader, netF, netB1, netC1, netB2, netC2, netB3, netC3, netG, args):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for _ in range(len(loader)):
            data = iter_test.next()
            inputs = data[0]
            labels = data[1] # ground truth
            inputs = inputs.cuda()
            feas1 = netB1(netF(inputs))
            outputs1 = netC1(feas1) #predicted
            feas2 = netB2(netF(inputs))
            outputs2 = netC2(feas2) 
            feas3 = netB3(netF(inputs))
            outputs3 = netC3(feas3) 
            outputsg = torch.stack([outputs1, outputs2, outputs3], 2)
            outputsg = netG(outputsg)
            outputsg = torch.squeeze(outputsg,2)
            if start_test:
                all_fea1 = feas1.float().cpu()
                all_fea2 = feas2.float().cpu()
                all_fea3 = feas3.float().cpu()
                all_output = outputsg.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_fea1 = torch.cat((all_fea1, feas1.float().cpu()), 0)
                all_fea2 = torch.cat((all_fea2, feas2.float().cpu()), 0)
                all_fea3 = torch.cat((all_fea3, feas3.float().cpu()), 0)
                all_output = torch.cat((all_output, outputsg.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)

    all_output = nn.Softmax(dim=1)(all_output)
    ent = torch.sum(-all_output * torch.log(all_output + args.epsilon), dim=1)
    unknown_weight = 1 - ent / np.log(args.class_num)
    _, predict = torch.max(all_output, 1)

    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    
    if args.da == 'oda':
        ent = torch.sum(-all_output * torch.log(all_output + args.epsilon), dim=1) / np.log(args.class_num)
        ent = ent.float().cpu()
        
        all_fea1, all_output1, predict1, known_idx1, ENT_THRESHOLD1= cal_fea_oda(all_fea1, all_output, predict, all_label, ent)
        all_fea2, all_output2, predict2, known_idx2, ENT_THRESHOLD2= cal_fea_oda(all_fea2, all_output, predict, all_label, ent)
        all_fea3, all_output3, predict3, known_idx3, ENT_THRESHOLD3= cal_fea_oda(all_fea3, all_output, predict, all_label, ent)

        
    else:
        if args.distance == 'cosine':
            all_fea1 = torch.cat((all_fea1, torch.ones(all_fea1.size(0), 1)), 1)
            all_fea1 = (all_fea1.t() / torch.norm(all_fea1, p=2, dim=1)).t()
            all_fea2 = torch.cat((all_fea2, torch.ones(all_fea2.size(0), 1)), 1)
            all_fea2 = (all_fea2.t() / torch.norm(all_fea2, p=2, dim=1)).t()
            all_fea3 = torch.cat((all_fea3, torch.ones(all_fea3.size(0), 1)), 1)
            all_fea3 = (all_fea3.t() / torch.norm(all_fea3, p=2, dim=1)).t()

        all_fea1 = all_fea1.float().cpu().numpy()
        all_fea2 = all_fea2.float().cpu().numpy()
        all_fea3 = all_fea3.float().cpu().numpy()
        
    
    '''
    if args.distance == 'cosine':
        all_fea1 = torch.cat((all_fea1, torch.ones(all_fea1.size(0), 1)), 1)
        all_fea1 = (all_fea1.t() / torch.norm(all_fea1, p=2, dim=1)).t()
        all_fea2 = torch.cat((all_fea2, torch.ones(all_fea2.size(0), 1)), 1)
        all_fea2 = (all_fea2.t() / torch.norm(all_fea2, p=2, dim=1)).t()
        all_fea3 = torch.cat((all_fea3, torch.ones(all_fea3.size(0), 1)), 1)
        all_fea3 = (all_fea3.t() / torch.norm(all_fea3, p=2, dim=1)).t()
    '''


    if not args.da == 'oda':
        all_output1 = all_output
        all_output2 = all_output
        all_output3 = all_output
        
        predict1 = predict
        predict2 = predict
        predict3 = predict
		
    
    K = all_output.size(1)
    aff = all_output1.float().cpu().numpy()
    
    initc1 = aff.transpose().dot(all_fea1)
    initc1 = initc1 / (1e-8 + aff.sum(axis=0)[:,None])
    cls_count = np.eye(K)[predict1].sum(axis=0)
    labelset1 = np.where(cls_count>args.threshold)
    labelset1 = labelset1[0]
    # print(labelset)

    dd = cdist(all_fea1, initc1[labelset1], args.distance)
    pred_label1 = dd.argmin(axis=1)
    pred_label1 = labelset1[pred_label1]
    
    for round in range(1):
        aff = np.eye(K)[pred_label1]
        initc1 = aff.transpose().dot(all_fea1)
        initc1 = initc1 / (1e-8 + aff.sum(axis=0)[:,None])
        dd = cdist(all_fea1, initc1[labelset1], args.distance)
        pred_label1 = dd.argmin(axis=1)
        pred_label1 = labelset1[pred_label1]
    
    aff = all_output2.float().cpu().numpy()
    
    initc2 = aff.transpose().dot(all_fea2)
    initc2 = initc2 / (1e-8 + aff.sum(axis=0)[:,None])
    cls_count = np.eye(K)[predict2].sum(axis=0)
    labelset2 = np.where(cls_count>args.threshold)
    labelset2 = labelset2[0]
    
    dd = cdist(all_fea2, initc2[labelset2], args.distance)
    pred_label2 = dd.argmin(axis=1)
    pred_label2 = labelset2[pred_label2]
    
    for round in range(1):        
        aff = np.eye(K)[pred_label2]
        initc2 = aff.transpose().dot(all_fea2)
        initc2 = initc2 / (1e-8 + aff.sum(axis=0)[:,None])
        dd = cdist(all_fea2, initc2[labelset2], args.distance)
        pred_label2 = dd.argmin(axis=1)
        pred_label2 = labelset2[pred_label2]
    
    aff = all_output3.float().cpu().numpy()
    
    initc3 = aff.transpose().dot(all_fea3)
    initc3 = initc3 / (1e-8 + aff.sum(axis=0)[:,None])
    cls_count = np.eye(K)[predict3].sum(axis=0)
    labelset3 = np.where(cls_count>args.threshold)
    labelset3 = labelset3[0]
    
    dd = cdist(all_fea3, initc3[labelset3], args.distance)
    pred_label3 = dd.argmin(axis=1)
    pred_label3 = labelset3[pred_label3]

    for round in range(1):
        '''
        aff = np.eye(K)[pred_label1]
        initc1 = aff.transpose().dot(all_fea1)
        initc1 = initc1 / (1e-8 + aff.sum(axis=0)[:,None])
        dd = cdist(all_fea1, initc1[labelset1], args.distance)
        pred_label1 = dd.argmin(axis=1)
        pred_label1 = labelset1[pred_label1]
        
        aff = np.eye(K)[pred_label2]
        initc2 = aff.transpose().dot(all_fea2)
        initc2 = initc2 / (1e-8 + aff.sum(axis=0)[:,None])
        dd = cdist(all_fea2, initc2[labelset2], args.distance)
        pred_label2 = dd.argmin(axis=1)
        pred_label2 = labelset2[pred_label2]
        '''
        
        aff = np.eye(K)[pred_label3]
        initc3 = aff.transpose().dot(all_fea3)
        initc3 = initc3 / (1e-8 + aff.sum(axis=0)[:,None])
        dd = cdist(all_fea3, initc3[labelset3], args.distance)
        pred_label3 = dd.argmin(axis=1)
        pred_label3 = labelset3[pred_label3]

    if args.da == 'oda':
        guess_label1 = cal_lab_oda(all_label, pred_label1, known_idx1, ENT_THRESHOLD1)
        guess_label2 = cal_lab_oda(all_label, pred_label2, known_idx2, ENT_THRESHOLD2)
        guess_label3 = cal_lab_oda(all_label, pred_label3, known_idx3, ENT_THRESHOLD3)
        
        acc = np.sum(guess_label1 == all_label.float().numpy()) / len(all_label)
        log_str1 = 'Threshold = {:.2f}, Accuracy = {:.2f}% -> {:.2f}%'.format(ENT_THRESHOLD1, accuracy*100, acc*100)
        acc = np.sum(guess_label2 == all_label.float().numpy()) / len(all_label)
        log_str2 = 'Threshold = {:.2f}, Accuracy = {:.2f}% -> {:.2f}%'.format(ENT_THRESHOLD2, accuracy*100, acc*100)
        acc = np.sum(guess_label3 == all_label.float().numpy()) / len(all_label)
        log_str3 = 'Threshold = {:.2f}, Accuracy = {:.2f}% -> {:.2f}%'.format(ENT_THRESHOLD3, accuracy*100, acc*100)
        
        args.out_file.write(log_str1 + '\n')
        args.out_file.flush()
        print(log_str1+'\n')
        args.out_file.write(log_str2 + '\n')
        args.out_file.flush()
        print(log_str2+'\n')
        args.out_file.write(log_str3 + '\n')
        args.out_file.flush()
        print(log_str3+'\n')

        return guess_label1.astype('int'), guess_label2.astype('int'), guess_label3.astype('int'), ENT_THRESHOLD1, ENT_THRESHOLD2, ENT_THRESHOLD3
    else:        
        acc = np.sum(pred_label1 == all_label.float().numpy()) / len(all_fea1)
        log_str1 = 'Accuracy = {:.2f}% -> {:.2f}%'.format(accuracy * 100, acc * 100)
        acc = np.sum(pred_label2 == all_label.float().numpy()) / len(all_fea2)
        log_str2 = 'Accuracy = {:.2f}% -> {:.2f}%'.format(accuracy * 100, acc * 100)
        acc = np.sum(pred_label3 == all_label.float().numpy()) / len(all_fea3)
        log_str3 = 'Accuracy = {:.2f}% -> {:.2f}%'.format(accuracy * 100, acc * 100)
        
        args.out_file.write(log_str1 + '\n')
        args.out_file.flush()
        print(log_str1+'\n')
        args.out_file.write(log_str2 + '\n')
        args.out_file.flush()
        print(log_str2+'\n')
        args.out_file.write(log_str3 + '\n')
        args.out_file.flush()
        print(log_str3+'\n')

        return pred_label1.astype('int'), pred_label2.astype('int'), pred_label3.astype('int')
    
    
    '''
    acc = np.sum(pred_label1 == all_label.float().numpy()) / len(all_fea1)
    log_str1 = 'Accuracy = {:.2f}% -> {:.2f}%'.format(accuracy * 100, acc * 100)
    acc = np.sum(pred_label2 == all_label.float().numpy()) / len(all_fea2)
    log_str2 = 'Accuracy = {:.2f}% -> {:.2f}%'.format(accuracy * 100, acc * 100)
    acc = np.sum(pred_label3 == all_label.float().numpy()) / len(all_fea3)
    log_str3 = 'Accuracy = {:.2f}% -> {:.2f}%'.format(accuracy * 100, acc * 100)
        
    args.out_file.write(log_str1 + '\n')
    args.out_file.flush()
    print(log_str1+'\n')
    args.out_file.write(log_str2 + '\n')
    args.out_file.flush()
    print(log_str2+'\n')
    args.out_file.write(log_str3 + '\n')
    args.out_file.flush()
    print(log_str3+'\n')

    return pred_label1.astype('int'), pred_label2.astype('int'), pred_label3.astype('int')
    '''


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SHOT')
    parser.add_argument('--gpu_id', type=str, nargs='?', default='1', help="device id to run")
    parser.add_argument('--s1', type=int, default=0, help="source")
    parser.add_argument('--s2', type=int, default=1, help="source")
    parser.add_argument('--s3', type=int, default=2, help="source")
    parser.add_argument('--t', type=int, default=3, help="target")
    parser.add_argument('--max_epoch', type=int, default=15, help="max iterations")
    parser.add_argument('--interval', type=int, default=15)
    parser.add_argument('--batch_size', type=int, default=64, help="batch_size")
    parser.add_argument('--worker', type=int, default=4, help="number of workers")
    parser.add_argument('--dset', type=str, default='OfficeHome', choices=['VISDA-C', 'office31', 'OfficeHome', 'offcal'])
    parser.add_argument('--lr', type=float, default=1e-2, help="learning rate")
    parser.add_argument('--net', type=str, default='resnet50', help="alexnet, vgg16, resnet50, res101")
    parser.add_argument('--seed', type=int, default=2020, help="random seed")
 
    parser.add_argument('--gent', type=bool, default=True)
    parser.add_argument('--ent', type=bool, default=True)
    parser.add_argument('--threshold', type=int, default=0)
    parser.add_argument('--cls_par', type=float, default=0.3)
    parser.add_argument('--ent_par', type=float, default=1.0)
    parser.add_argument('--lr_decay1', type=float, default=0.1)
    parser.add_argument('--lr_decay2', type=float, default=1.0)

    parser.add_argument('--bottleneck', type=int, default=256)
    parser.add_argument('--epsilon', type=float, default=1e-5)
    parser.add_argument('--layer', type=str, default="wn", choices=["linear", "wn"])
    parser.add_argument('--classifier', type=str, default="bn", choices=["ori", "bn"])
    parser.add_argument('--distance', type=str, default='euclidean', choices=["euclidean", "cosine"])  
    parser.add_argument('--output', type=str, default='san')
    parser.add_argument('--output_src', type=str, default='san')
    parser.add_argument('--da', type=str, default='uda', choices=['uda', 'pda', 'oda'])
    parser.add_argument('--issave', type=bool, default=False)
    args = parser.parse_args()

    if args.dset == 'OfficeHome':
        names = ['Art', 'Clipart', 'Product', 'RealWorld']
        args.class_num = 65 
        #task = [0,1,2,3] 
        #task = [0,1,3,2]
        #task = [0,2,3,1] 
        task = [1,2,3,0]
    if args.dset == 'office31':
        names = ['amazon', 'dslr', 'webcam']
        args.class_num = 31
        task = [0,1,2] 
        #task = [2,1,0]
        #task = [0,2,1]
    if args.dset == 'VISDA-C':
        names = ['train', 'validation']
        args.class_num = 12
    if args.dset == 'offcal':
        names = ['amazon', 'caltech', 'dslr', 'webcam']
        #task = [0,1,2,3] 
        task = [0,1,3,2]
        #task = [0,2,3,1] 
        #task = [1,2,3,0]
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
    args.t_dset_path = folder + args.dset + '/' + names[task[args.t]] + 'List.txt'
    args.test_dset_path = folder + args.dset + '/' + names[task[args.t]] + 'List.txt'
    args.name_src1 = names[task[args.s1]][0].upper()
    args.name_src2 = names[task[args.s2]][0].upper()
    args.name_src3 = names[task[args.s3]][0].upper()


    if args.dset == 'OfficeHome':
        if args.da == 'pda':
            args.class_num = 65
            args.src_classes = [i for i in range(65)]
            args.tar_classes = [i for i in range(25)]
        if args.da == 'oda':
            args.class_num = 25
            args.src_classes = [i for i in range(25)]
            args.tar_classes = [i for i in range(65)]
    
    load_tag = 'sonly_cb'
    args.task = names[task[args.s1]][0].upper() + names[task[args.s2]][0].upper() + names[task[args.s3]][0].upper() + '2' + names[task[args.t]][0].upper()   
    args.output_dir_src = osp.join(args.output, args.da, args.dset, args.task, load_tag)
    traepo = 0
    save_tag = 'epo_cb_eu' + str(traepo)
    args.output_dir = osp.join(args.output, args.da, args.dset, args.task, save_tag)

    if not osp.exists(args.output_dir):
        os.system('mkdir -p ' + args.output_dir)
    if not osp.exists(args.output_dir):
        os.mkdir(args.output_dir)

    args.savename = 'par_' + str(args.cls_par)
    if args.da == 'pda':
        args.gent = ''
        args.savename = 'par_' + str(args.cls_par) + '_thr' + str(args.threshold)
    args.out_file = open(osp.join(args.output_dir, 'log_' + args.savename + '.txt'), 'w')
    args.out_file.write(print_args(args)+'\n')
    args.out_file.flush()
    train_target(args)
