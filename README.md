# SSD-tensorflow-mask

一个`Tensorflow SSD进行佩戴口罩识别`（目标检测）任务

# 复现我的模型

## 1.下载SSD源码

https://github.com/balancap/SSD-Tensorflow

## 2.解压源码目录

## 3.在解压目录下新建文件夹

创建`tfrecords_`、`train_model`、`VOC2007`文件夹

1. `VOC2007`存放数据；
2. `tfrecords_`文件夹是用来存储`.tfrecords`文件（后面有程序可以直接生成）
3. `train_model`文件夹是用来存储模型的记录与参数的

再将【mask-dataset】中【mask】中制作的三个文件夹`Annotations`、`ImageSets`、`JPEGImages`全都拖入`VOC2007`文件夹内；

## 4.生成.tfrecords文件的代码微调

- 修改标签项——打开`datasets`文件夹中`pascalvoc_common.py`文件，将自己的标签项填入。`

```html
VOC_LABELS = {
    'none': (0, 'Background'),
    'mask': (1, 'mask'),
    'no-mask': (2, 'no-mask')
}
```

- 修改读取个数、读取方式——打开`datasets`文件夹中的`pascalvoc_to_tfrecords.py`文件。

(1)修改67行`SAMPLES_PER_FILES`的个数（几张图生成一个`.tfrecords`）；

(2)修改83行读取方式为'`rb`'；

(3)如果你的文件不是`.jpg`格式，也可以修改图片的类型；

## 5.生成.tfrecords文件

```python
python tf_convert_data.py --dataset_name=pascalvoc --dataset_dir=./VOC2007/ --output_name=voc_2007_train --output_dir=./tfrecords_
```

此时，会在`tfrecords_`文件夹中生成相应格式的文件。

## 6.训练模型的代码微调

- 修改训练数据shape——打开`datasets`文件夹中的`pascalvoc_2007.py`文件，

根据自己训练数据修改：`NUM_CLASSES` = **类别数**

```html
TRAIN_STATISTICS = {
    'none': (0, 0),
    'mask': (149, 804),
    'no-mask': (54, 148),
    'total': (203, 952),  # （图片数量，框数）
}
TEST_STATISTICS = {
    'none': (0, 0),
    'mask': (1, 1),
    'no-mask': (1, 1),
    'total': (2, 2),
}
SPLITS_TO_SIZES = {
    'train': 149,
    'test': 54,
}
SPLITS_TO_STATISTICS = {
    'train': TRAIN_STATISTICS,
    'test': TEST_STATISTICS,
}
NUM_CLASSES = 2
```

- 修改类别个数——打开`nets`文件夹中的`ssd_vgg_300.py`文件

根据自己训练类别数修改96 和97行：**等于类别数+1**

> 比如我做戴口罩和未戴口罩的2个类别，那我应该写的是3！

- 修改类别个数——打开`eval_ssd_network.py`文件

修改66行的类别个数：**等于类别数+1**

- 修改训练步数epoch——打开train_ssd_network.py文件，

(1)修改27行的数据格式，改为'`NHWC`'；

(2)修改135行的类别个数：**等于类别数+1**；

(3)修改154行训练总步数，None会无限训练下去；

(4)说明：60行、63行是关于模型保存的参数；

## 7.加载VGG_16

进入到`checkpoint`文件夹，运行下载指令：
```html
https://www.flyai.com/m/vgg_16_2016_08_28.tar.gz
```

当然你也可以直接去百度搜【vgg_16_2016_08_28.tar.gz】，有很多人下好了给你用。

## 8.训练模型

```python
python train_ssd_network.py --train_dir=./train_model/ --dataset_dir=./tfrecords_/ --dataset_name=pascalvoc_2007 --dataset_split_name=train --model_name=ssd_300_vgg --checkpoint_path=./checkpoints/vgg_16.ckpt --checkpoint_model_scope=vgg_16 --checkpoint_exclude_scopes=ssd_300_vgg/conv6,ssd_300_vgg/conv7,ssd_300_vgg/block8,ssd_300_vgg/block9,ssd_300_vgg/block10,ssd_300_vgg/block11,ssd_300_vgg/block4_box,ssd_300_vgg/block7_box,ssd_300_vgg/block8_box,ssd_300_vgg/block9_box,ssd_300_vgg/block10_box,ssd_300_vgg/block11_box --trainable_scopes=ssd_300_vgg/conv6,ssd_300_vgg/conv7,ssd_300_vgg/block8,ssd_300_vgg/block9,ssd_300_vgg/block10,ssd_300_vgg/block11,ssd_300_vgg/block4_box,ssd_300_vgg/block7_box,ssd_300_vgg/block8_box,ssd_300_vgg/block9_box,ssd_300_vgg/block10_box,ssd_300_vgg/block11_box --save_summaries_secs=60 --save_interval_secs=50 --weight_decay=0.0005 --optimizer=adam --learning_rate=0.001 --learning_rate_decay_factor=0.94 --batch_size=256 --gpu_memory_fraction=0.95
```

- `save_interval_secs`是训练多少次保存参数的步长；
- `optimizer`是优化器；
- `learning_rate`是学习率；
- `learning_rate_decay_factor`是学习率衰减因子；
- 如果你的机器比较强大，可以适当增大`--batch_size`的数值，以及调高GPU的占比`--gpu_memory_fraction`；
- `model_name`；

## 9.训练完成

训练结束可以在`train_model`文件夹下看到生成的参数文件；

# 后续工作

- 模型ckpt转化为pb；
- 模型测试；
- 模型嵌入；

# 参考

- https://blog.csdn.net/zzz_cming/article/details/81128460


