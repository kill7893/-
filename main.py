import os
import random
import argparse
import yaml
from tqdm import tqdm
import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision.transforms as transforms
from datasets import build_dataset
from datasets.utils import build_data_loader
import clip
from utils import *

def reading_parameter():
    pas = argparse.ArgumentParser()
    pas.add_argument('--config', dest='config', default='configs/OOH.yaml')
    arguments = pas.parse_args()
    return arguments

def Unlearnable(cfg, cache_keys, cache_values, val_features, val_labels, test_features, test_labels, clip_weights):

    # A
    clip_logits = 100. * val_features @ clip_weights
    zero_pre=cls_precision(clip_logits, val_labels)
    print("\n**** A val precision: {:.2f}. ****\n".format(zero_pre))

    # Tip-Adapter
    beta, alpha = cfg['init_beta'], cfg['init_alpha']
    affinity = val_features @ cache_keys
    cache_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values
    tip_logits = clip_logits + cache_logits * alpha
    cache_pre=cls_precision(tip_logits, val_labels)
    print("**** Tip-Adapter's val precision: {:.2f}. ****\n".format(cache_pre))

    # Search Hyperparameters
    best_beta, best_alpha = search_hp(cfg, cache_keys, cache_values, val_features, val_labels, clip_weights)

    print("\n-------- Evaluating on the test set. --------")

    # A
    clip_logits = 100. * test_features @ clip_weights
    zt_pree = cls_precision(clip_logits, test_labels)
    print("\n**** A test precision: {:.2f}. ****\n".format(zt_pree))

    # A AND B
    affinity = test_features @ cache_keys
    cache_logits = ((-1) * (best_beta - best_beta * affinity)).exp() @ cache_values
    tip_logits = clip_logits + cache_logits * best_alpha
    tc_pree = cls_precision(tip_logits, test_labels)
    print("**** A AND B test precision: {:.2f}. ****\n".format(tc_pree))

def Learnable(cfg, cache_keys, cache_values, val_features, val_labels,test_features, test_labels, clip_weights, clip_model, train_loader_F):

    # Enable the cached keys to be learnable
    adapter = nn.Linear(cache_keys.shape[0], cache_keys.shape[1], bias=False).to(clip_model.dtype).cuda()
    adapter.weight = nn.Parameter(cache_keys.t())
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=cfg['lr'], eps=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg['train_epoch'] * len(train_loader_F))
    beta, alpha = cfg['init_beta'], cfg['init_alpha']
    best_pre, best_epoch = 0.0, 0

    train_loss = []
    train_preacc = []
    test_preacc = []

    for train_idx in range(cfg['train_epoch']):
        # Train
        adapter.train()
        correct_samples, all_samples = 0, 0
        loss_list = []
        print('Train Epoch: {:} / {:}'.format(train_idx, cfg['train_epoch']))

        for i, (images, target) in enumerate(tqdm(train_loader_F)):
            images, target = images.cuda(), target.cuda()
            with torch.no_grad():
                image_features = clip_model.encode_image(images)
                image_features /= image_features.norm(dim=-1, keepdim=True)

            affinity = adapter(image_features)
            cache_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values
            clip_logits = 100. * image_features @ clip_weights
            tip_logits = clip_logits + cache_logits * alpha
            loss = F.cross_entropy(tip_logits, target)
            train_pre = cls_precision(tip_logits, target)
            train_preacc.append(train_pre)
            correct_samples += train_pre / 100 * len(tip_logits)
            all_samples += len(tip_logits)
            loss_list.append(loss.item())
            train_loss.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        print('LR: {:.6f},Loss: {:.4f}'.format(current_lr,sum(loss_list)/len(loss_list)))

        # Eval
        adapter.eval()
        affinity = adapter(test_features)
        cache_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values
        clip_logits = 100. * test_features @ clip_weights
        tip_logits = clip_logits + cache_logits * alpha
        #测试集精确率
        pre=cls_precision(tip_logits, test_labels)
        test_preacc.append(pre)
        print("**** A AND C test precision: {:.2f}. ****\n".format(pre))
        if pre > best_pre:
            best_pre = pre
            best_epoch = train_idx
            torch.save(adapter.weight, cfg['data_dir'] + "/best_F_" + str(cfg['shots']) + "shots.pt")
    adapter.weight = torch.load(cfg['data_dir'] + "/best_F_" + str(cfg['shots']) + "shots.pt")
    print(f"**** After fine-tuning, A ADN C  best test precision: {best_pre:.2f}, at epoch: {best_epoch}. ****\n")

    with open("./train_loss.txt", 'w') as train_los:
        train_los.write(str(train_loss))

    with open("./train_preacc.txt", 'w') as val_ac:
        val_ac.write(str(train_preacc))

    with open("./test_preacc.txt", 'w') as test_pre:
        test_pre.write(str(test_preacc))
    print("\n-------- Searching hyperparameters on the val set. --------")
    # Search Hyperparameters
    best_beta, best_alpha = search_hp(cfg, cache_keys, cache_values, val_features, val_labels, clip_weights, adapter=adapter)
    print("\n-------- Evaluating on the test set. --------")
    affinity = adapter(test_features)
    cache_logits = ((-1) * (best_beta - best_beta * affinity)).exp() @ cache_values
    tip_logits = clip_logits + cache_logits * best_alpha
    last_pre = cls_precision(tip_logits, test_labels)
    print("**** A ADN C  test precision: {:.2f}. ****\n".format(max(best_pre, last_pre)))

def main():

    args = reading_parameter()
    assert (os.path.exists(args.config))    #args.config== configs/xx.yaml，assert断言检查：检测路径是否存在
    cfg = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)     #yaml.load是读取文件转换成参数赋值给cfg，loader是选择加载器来加载文件
    data_dir = os.path.join('./data', cfg['dataset'])
    os.makedirs(data_dir, exist_ok=True)
    cfg['data_dir'] = data_dir

    clip_model, preprocess = clip.load(cfg['backbone']) #两个参数--加载模型和预处理函数：对输入图像预处理
    clip_model.eval() #评估模式：模型将不会进行梯度计算和参数更新

    #保证在每次运行程序时获得相同的随机数序列，从而提高可复现性
    random.seed(1)
    torch.manual_seed(1)

    print("Prepare a dataset of offending outdoor advertising images.")
    #读取数据集文件x.json
    dataset = build_dataset(cfg['dataset'], cfg['root_path'], cfg['shots'])
    #划分训练集和测试集，其中is_train=False：为推理/测试模式（不进行参数更新），shuffle=True适用于训练模式
    val_loader = build_data_loader(data_source=dataset.val, batch_size=64, is_train=False, tfm=preprocess, shuffle=False)
    test_loader = build_data_loader(data_source=dataset.test, batch_size=64, is_train=False, tfm=preprocess, shuffle=False)
    #数据预处理，transforms.Compose用于将多个变换函数组合成一个串联的变换操作。
    train_tranform = transforms.Compose([
        transforms.RandomResizedCrop(size=224, scale=(0.5, 1), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
    ])

    #用于构建缓存模型
    train_loader_cache = build_data_loader(data_source=dataset.train_x, batch_size=64, tfm=train_tranform, is_train=True, shuffle=False)
    #随机打乱可以增加模型的泛化能力，避免模型对输入数据的顺序产生依赖，从而更好地学习到数据的整体分布。
    train_loader_F = build_data_loader(data_source=dataset.train_x, batch_size=64, tfm=train_tranform, is_train=True, shuffle=True)

    # clip_weights指的是文本对应的图像特征，dataset.template记录了文本描述的模板，里面占位符在此处为图像的类别：这是一个XX的图片
    print("\nGetting textual features as CLIP's classifier.")
    clip_weights = clip_classifier(dataset.classnames, dataset.template, clip_model)

    # 缓存模型构建
    print("\nConstructing cache model by few-shot visual features and labels.")
    cache_keys, cache_values = build_cache_model(cfg, clip_model, train_loader_cache)

    # Pre-load val features
    print("\nLoading visual features and labels from val set.")
    val_features, val_labels = pre_load_features(cfg, "val", clip_model, val_loader)
    # Pre-load test features
    print("\nLoading visual features and labels from test set.")
    test_features, test_labels = pre_load_features(cfg, "test", clip_model, test_loader)

    #Non-updatable cache model
    Unlearnable(cfg, cache_keys, cache_values, val_features, val_labels, test_features, test_labels, clip_weights)
    #Updatable cache model
    Learnable(cfg, cache_keys, cache_values, val_features, val_labels, test_features, test_labels, clip_weights, clip_model, train_loader_F)

if __name__ == '__main__':
    main()