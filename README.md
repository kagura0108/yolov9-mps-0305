# yolov9-0305
ver0305

暂时无法在飞书文档外展示此内容

---
1. b 站-qiong_学海无涯_
https://www.bilibili.com/video/BV1mH4y1L7GX/?spm_id_from=333.337.search-card.all.click&vd_source=cb4f9bfc1bb86b6e63059c47d737907e
1.1 github
1.1.1 下载 zip
https://github.com/WongKinYiu/yolov9
1.1.2 下载预训练权重
up 主选择了 -c
[图片]
放入/yolov9-main/里面
[图片]
1.1.3 下载官方数据集
[图片]
后面没有用到
1.1.4 下载 up 主 dataset
up 主只需要 labels images
https://www.bilibili.com/video/BV12K411a7X4/?spm_id_from=333.999.0.0&vd_source=cb4f9bfc1bb86b6e63059c47d737907e
评论区
https://pan.baidu.com/s/1l0Uszrfp9lN89fRwh0mL_w 
提取码：pa7n 
放入对应的目录下。
[图片]

---
1.2 配置
1.2.1 修改 coco.yaml
本来
[图片]
path: ../datasets/coco  # dataset root dir
train: train2017.txt  # train images (relative to 'path') 118287 images
val: val2017.txt  # val images (relative to 'path') 5000 images
test: test-dev2017.txt  # 20288 of 40670 images, submit to https://competitions.codalab.org/competitions/20794
改成
# path: ../datasets/coco  # dataset root dir
train: /Users/myk/PycharmProjects/pyPjct-yolov9-0305/yolov9-main/data/data-self/train2017  # train images (relative to 'path') 118287 images
val: /Users/myk/PycharmProjects/pyPjct-yolov9-0305/yolov9-main/data/data-self/val2017  # val images (relative to 'path') 5000 images
test: /Users/myk/PycharmProjects/pyPjct-yolov9-0305/yolov9-main/data/data-self/test2017  # 20288 of 40670 images, submit to https://competitions.codalab.org/competitions/20794
#Classes 改成（up 主更改了）
# path: ../datasets/coco  # dataset root dir
train: /Users/myk/PycharmProjects/pyPjct-yolov9-0305/yolov9-main/data/data-2/images/train  # train images (relative to 'path') 118287 images
val: /Users/myk/PycharmProjects/pyPjct-yolov9-0305/yolov9-main/data/data-2/images/val  # val images (relative to 'path') 5000 images
test: /Users/myk/PycharmProjects/pyPjct-yolov9-0305/yolov9-main/data/data-2/images/val  # 20288 of 40670 images, submit to https://competitions.codalab.org/competitions/20794

# Classes
names:
  0: no mask
  1: mask
  2: not good
Question 这样是不是会快一点
up 主：test 没有，就把 val 放过来。
1.2.2 修改 train_dual.py
train_dual.py Code 442 本来
def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    # parser.add_argument('--weights', type=str, default=ROOT / 'yolo.pt', help='initial weights path')
    # parser.add_argument('--cfg', type=str, default='', help='model.yaml path')
    parser.add_argument('--weights', type=str, default='', help='initial weights path')
    parser.add_argument('--cfg', type=str, default='yolo.yaml', help='model.yaml path')
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco.yaml', help='dataset.yaml path')
    parser.add_argument('--hyp', type=str, default=ROOT / 'data/hyps/hyp.scratch-high.yaml', help='hyperparameters path')
改成 yolov9-c-converted.pt 的路径
改成/yolov9-main/models/detect/yolov9-c.yaml 的路径
改成/yolov9-main/data/coco.yaml 的路径
改成/yolov9-main/data/hyps/hyp.scratch-high.yaml 的路径
    parser.add_argument('--weights', type=str, default='/Users/myk/PycharmProjects/pyPjct-yolov9-0305/yolov9-main/yolov9-c-converted.pt', help='initial weights path')
    parser.add_argument('--cfg', type=str, default='/Users/myk/PycharmProjects/pyPjct-yolov9-0305/yolov9-main/models/detect/yolov9-c.yaml', help='model.yaml path')
    parser.add_argument('--data', type=str, default='/Users/myk/PycharmProjects/pyPjct-yolov9-0305/yolov9-main/data/coco.yaml', help='model.yaml path')
    parser.add_argument('--hyp', type=str, default='/Users/myk/PycharmProjects/pyPjct-yolov9-0305/yolov9-main/data/hyps/hyp.scratch-high.yaml', help='hyperparameters path')
question 地址 双斜杠要不要加
Answer 不用
[图片]
接下来的几行
    parser.add_argument('--epochs', type=int, default=100, help='total training epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='total batch size for all GPUs, -1 for autobatch')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='train, val image size (pixels)')
    parser.add_argument('--rect', action='store_true', help='rectangular training')
epochs 如果自己电脑，看看效果，无所谓
batch-size 一定设置 1 很考验 gpu（原来是default=16）
1.3 虚拟环境 yolov9-0305
(yolov9-0305) myk@miyuki-M2Max pyPjct-yolov9-0305 % cd /Users/myk/PycharmProjects/pyPjct-yolov9-0305/yolov9-main
(yolov9-0305) myk@miyuki-M2Max yolov9-main % pip install -r requirements.txt
[图片]
如果跑不通，up 主建议去 audl 租一台服务器

---
1.4 train_dual.py exp-01
全文
/Users/myk/anaconda3/envs/yolov9-0305/bin/python /Users/myk/PycharmProjects/pyPjct-yolov9-0305/yolov9-main/train_dual.py 
train_dual: weights=/Users/myk/PycharmProjects/pyPjct-yolov9-0305/yolov9-main/yolov9-c-converted.pt, cfg=/Users/myk/PycharmProjects/pyPjct-yolov9-0305/yolov9-main/models/detect/yolov9-c.yaml, data=/Users/myk/PycharmProjects/pyPjct-yolov9-0305/yolov9-main/data/coco.yaml, hyp=/Users/myk/PycharmProjects/pyPjct-yolov9-0305/yolov9-main/data/hyps/hyp.scratch-high.yaml, epochs=10, batch_size=1, imgsz=640, rect=False, resume=False, nosave=False, noval=False, noautoanchor=False, noplots=False, evolve=None, bucket=, cache=None, image_weights=False, device=, multi_scale=False, single_cls=False, optimizer=SGD, sync_bn=False, workers=8, project=runs/train, name=exp, exist_ok=False, quad=False, cos_lr=False, flat_cos_lr=False, fixed_lr=False, label_smoothing=0.0, patience=100, freeze=[0], save_period=-1, seed=0, local_rank=-1, min_items=0, close_mosaic=0, entity=None, upload_dataset=False, bbox_interval=-1, artifact_alias=latest
YOLOv5 🚀 2024-3-4 Python-3.8.18 torch-2.2.1 CPU

hyperparameters: lr0=0.01, lrf=0.01, momentum=0.937, weight_decay=0.0005, warmup_epochs=3.0, warmup_momentum=0.8, warmup_bias_lr=0.1, box=7.5, cls=0.5, cls_pw=1.0, obj=0.7, obj_pw=1.0, dfl=1.5, iou_t=0.2, anchor_t=5.0, fl_gamma=0.0, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, degrees=0.0, translate=0.1, scale=0.9, shear=0.0, perspective=0.0, flipud=0.0, fliplr=0.5, mosaic=1.0, mixup=0.15, copy_paste=0.3
ClearML: run 'pip install clearml' to automatically track, visualize and remotely train YOLO 🚀 in ClearML
Comet: run 'pip install comet_ml' to automatically track and visualize YOLO 🚀 runs in Comet
TensorBoard: Start with 'tensorboard --logdir runs/train', view at http://localhost:6006/
Overriding model.yaml nc=80 with nc=3

                 from  n    params  module                                  arguments                     
  0                -1  1         0  models.common.Silence                   []                            
  1                -1  1      1856  models.common.Conv                      [3, 64, 3, 2]                 
  2                -1  1     73984  models.common.Conv                      [64, 128, 3, 2]               
  3                -1  1    212864  models.common.RepNCSPELAN4              [128, 256, 128, 64, 1]        
  4                -1  1    164352  models.common.ADown                     [256, 256]                    
  5                -1  1    847616  models.common.RepNCSPELAN4              [256, 512, 256, 128, 1]       
  6                -1  1    656384  models.common.ADown                     [512, 512]                    
  7                -1  1   2857472  models.common.RepNCSPELAN4              [512, 512, 512, 256, 1]       
  8                -1  1    656384  models.common.ADown                     [512, 512]                    
  9                -1  1   2857472  models.common.RepNCSPELAN4              [512, 512, 512, 256, 1]       
 10                -1  1    656896  models.common.SPPELAN                   [512, 512, 256]               
 11                -1  1         0  torch.nn.modules.upsampling.Upsample    [None, 2, 'nearest']          
 12           [-1, 7]  1         0  models.common.Concat                    [1]                           
 13                -1  1   3119616  models.common.RepNCSPELAN4              [1024, 512, 512, 256, 1]      
 14                -1  1         0  torch.nn.modules.upsampling.Upsample    [None, 2, 'nearest']          
 15           [-1, 5]  1         0  models.common.Concat                    [1]                           
 16                -1  1    912640  models.common.RepNCSPELAN4              [1024, 256, 256, 128, 1]      
 17                -1  1    164352  models.common.ADown                     [256, 256]                    
 18          [-1, 13]  1         0  models.common.Concat                    [1]                           
 19                -1  1   2988544  models.common.RepNCSPELAN4              [768, 512, 512, 256, 1]       
 20                -1  1    656384  models.common.ADown                     [512, 512]                    
 21          [-1, 10]  1         0  models.common.Concat                    [1]                           
 22                -1  1   3119616  models.common.RepNCSPELAN4              [1024, 512, 512, 256, 1]      
 23                 5  1    131328  models.common.CBLinear                  [512, [256]]                  
 24                 7  1    393984  models.common.CBLinear                  [512, [256, 512]]             
 25                 9  1    656640  models.common.CBLinear                  [512, [256, 512, 512]]        
 26                 0  1      1856  models.common.Conv                      [3, 64, 3, 2]                 
 27                -1  1     73984  models.common.Conv                      [64, 128, 3, 2]               
 28                -1  1    212864  models.common.RepNCSPELAN4              [128, 256, 128, 64, 1]        
 29                -1  1    164352  models.common.ADown                     [256, 256]                    
 30  [23, 24, 25, -1]  1         0  models.common.CBFuse                    [[0, 0, 0]]                   
 31                -1  1    847616  models.common.RepNCSPELAN4              [256, 512, 256, 128, 1]       
 32                -1  1    656384  models.common.ADown                     [512, 512]                    
 33      [24, 25, -1]  1         0  models.common.CBFuse                    [[1, 1]]                      
 34                -1  1   2857472  models.common.RepNCSPELAN4              [512, 512, 512, 256, 1]       
 35                -1  1    656384  models.common.ADown                     [512, 512]                    
 36          [25, -1]  1         0  models.common.CBFuse                    [[2]]                         
 37                -1  1   2857472  models.common.RepNCSPELAN4              [512, 512, 512, 256, 1]       
 38[31, 34, 37, 16, 19, 22]  1  21547442  models.yolo.DualDDetect                 [3, [512, 512, 512, 256, 512, 512]]
yolov9-c summary: 962 layers, 51004210 parameters, 51004178 gradients, 238.9 GFLOPs

Transferred 18/1460 items from /Users/myk/PycharmProjects/pyPjct-yolov9-0305/yolov9-main/yolov9-c-converted.pt
optimizer: SGD(lr=0.01) with parameter groups 238 weight(decay=0.0), 255 weight(decay=0.0005), 253 bias
albumentations: Blur(p=0.01, blur_limit=(3, 7)), MedianBlur(p=0.01, blur_limit=(3, 7)), ToGray(p=0.01), CLAHE(p=0.01, clip_limit=(1, 4.0), tile_grid_size=(8, 8))
train: Scanning /Users/myk/PycharmProjects/pyPjct-yolov9-0305/yolov9-main/data/data-2/labels/train... 653 images, 0 backgrounds, 0 corrupt: 100%|██████████| 653/653 00:02
train: New cache created: /Users/myk/PycharmProjects/pyPjct-yolov9-0305/yolov9-main/data/data-2/labels/train.cache
val: Scanning /Users/myk/PycharmProjects/pyPjct-yolov9-0305/yolov9-main/data/data-2/labels/val... 200 images, 0 backgrounds, 0 corrupt: 100%|██████████| 200/200 00:02
val: New cache created: /Users/myk/PycharmProjects/pyPjct-yolov9-0305/yolov9-main/data/data-2/labels/val.cache
Plotting labels to runs/train/exp/labels.jpg... 
Image sizes 640 train, 640 val
Using 0 dataloader workers
Logging results to runs/train/exp
Starting training for 10 epochs...

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        0/9         0G      4.355      6.132      5.364          6        640:   0%|          | 0/653 00:01Exception in thread Thread-9:
Traceback (most recent call last):
  File "/Users/myk/anaconda3/envs/yolov9-0305/lib/python3.8/threading.py", line 932, in _bootstrap_inner
    self.run()
  File "/Users/myk/anaconda3/envs/yolov9-0305/lib/python3.8/threading.py", line 870, in run
    self._target(*self._args, **self._kwargs)
  File "/Users/myk/PycharmProjects/pyPjct-yolov9-0305/yolov9-main/utils/plots.py", line 300, in plot_images
    annotator.box_label(box, label, color=color)
  File "/Users/myk/PycharmProjects/pyPjct-yolov9-0305/yolov9-main/utils/plots.py", line 86, in box_label
    w, h = self.font.getsize(label)  # text width, height
AttributeError: 'FreeTypeFont' object has no attribute 'getsize'
WARNING ⚠️ TensorBoard graph visualization failure Only tensors, lists, tuples of tensors, or dictionary of tensors can be output from traced functions
        0/9         0G       3.47      6.101      5.275          1        640:   0%|          | 2/653 00:04Exception in thread Thread-10:
Traceback (most recent call last):
  File "/Users/myk/anaconda3/envs/yolov9-0305/lib/python3.8/threading.py", line 932, in _bootstrap_inner
    self.run()
  File "/Users/myk/anaconda3/envs/yolov9-0305/lib/python3.8/threading.py", line 870, in run
    self._target(*self._args, **self._kwargs)
  File "/Users/myk/PycharmProjects/pyPjct-yolov9-0305/yolov9-main/utils/plots.py", line 300, in plot_images
    annotator.box_label(box, label, color=color)
  File "/Users/myk/PycharmProjects/pyPjct-yolov9-0305/yolov9-main/utils/plots.py", line 86, in box_label
    w, h = self.font.getsize(label)  # text width, height
AttributeError: 'FreeTypeFont' object has no attribute 'getsize'
        0/9         0G      3.718      6.064      5.412          4        640:   0%|          | 3/653 00:05Exception in thread Thread-11:
Traceback (most recent call last):
  File "/Users/myk/anaconda3/envs/yolov9-0305/lib/python3.8/threading.py", line 932, in _bootstrap_inner
    self.run()
  File "/Users/myk/anaconda3/envs/yolov9-0305/lib/python3.8/threading.py", line 870, in run
    self._target(*self._args, **self._kwargs)
  File "/Users/myk/PycharmProjects/pyPjct-yolov9-0305/yolov9-main/utils/plots.py", line 300, in plot_images
    annotator.box_label(box, label, color=color)
  File "/Users/myk/PycharmProjects/pyPjct-yolov9-0305/yolov9-main/utils/plots.py", line 86, in box_label
    w, h = self.font.getsize(label)  # text width, height
AttributeError: 'FreeTypeFont' object has no attribute 'getsize'
        0/9         0G        4.8      7.161      5.014          5        640:  26%|██▋       | 172/653 04:10libpng warning: iCCP: Not recognizing known sRGB profile that has been edited
        0/9         0G      4.938      7.138      5.024         38        640:  33%|███▎      | 216/653 05:13libpng warning: iCCP: Not recognizing known sRGB profile that has been edited
        0/9         0G      4.967      7.048      5.048          6        640:  42%|████▏     | 274/653 06:35libpng warning: iCCP: Not recognizing known sRGB profile that has been edited
        0/9         0G      4.844      6.957      4.928         30        640:  93%|█████████▎| 605/653 14:24libpng warning: iCCP: Not recognizing known sRGB profile that has been edited
        0/9         0G      4.861      6.954      4.937         20        640: 100%|██████████| 653/653 15:31
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100%|██████████| 100/100 01:30
                   all        200        867    0.00145     0.0107   0.000443   0.000125

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        1/9         0G      5.161      6.719      5.133          6        640:  19%|█▊        | 121/653 02:49libpng warning: iCCP: Not recognizing known sRGB profile that has been edited
        1/9         0G      5.192      6.637       5.02         17        640:  50%|█████     | 328/653 07:39libpng warning: iCCP: Not recognizing known sRGB profile that has been edited
        1/9         0G      5.061      6.668      4.933          8        640:  92%|█████████▏| 602/653 14:06libpng warning: iCCP: Not recognizing known sRGB profile that has been edited
        1/9         0G      5.027      6.648      4.924          9        640: 100%|██████████| 653/653 15:21
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100%|██████████| 100/100 01:23
                   all        200        867    0.00074     0.0189   0.000335   7.93e-05

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        2/9         0G      5.267      6.304      5.227          4        640:   6%|▌         | 36/653 00:51libpng warning: iCCP: Not recognizing known sRGB profile that has been edited
        2/9         0G      4.937      6.306      5.045         24        640:  19%|█▉        | 124/653 02:55libpng warning: iCCP: Not recognizing known sRGB profile that has been edited
        2/9         0G      4.845      6.462      4.876          3        640:  24%|██▎       | 155/653 03:40
Traceback (most recent call last):
  File "/Users/myk/PycharmProjects/pyPjct-yolov9-0305/yolov9-main/train_dual.py", line 644, in <module>
    main(opt)
  File "/Users/myk/PycharmProjects/pyPjct-yolov9-0305/yolov9-main/train_dual.py", line 538, in main
    train(opt.hyp, opt, device, callbacks)
  File "/Users/myk/PycharmProjects/pyPjct-yolov9-0305/yolov9-main/train_dual.py", line 322, in train
    scaler.scale(loss).backward()
  File "/Users/myk/anaconda3/envs/yolov9-0305/lib/python3.8/site-packages/torch/_tensor.py", line 522, in backward
    torch.autograd.backward(
  File "/Users/myk/anaconda3/envs/yolov9-0305/lib/python3.8/site-packages/torch/autograd/__init__.py", line 266, in backward
    Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass
KeyboardInterrupt

进程已结束，退出代码为 130 (interrupted by signal 2:SIGINT)

截图
[图片]
[图片]
[图片]
1.4.1 温度
[图片]
回归正常
[图片]
1.5 train_dual-mps.py exp02 - 尝试改 mps
难点 在哪里寻找更改 gpu 的选择





