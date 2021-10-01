# Script_for_PointNet-_used_in_MasterThesis
This is the Python script for PiointNet++, which has been used in master thesis to train the network and test it.
## Environments Requirement
···
CUDA = 10.2
Python = 3.7.0
PyTorch = 1.6
open3d
opencv
scikit-learn

## How to Run
### Train the Model
You can use the following command to train the model:
```
CUDA_VISIBLE_DEVICES=4,5 python train_semseg.py --model pointnet2_sem_seg --epoch 100 --log_dir pointnet2_sem_seg_0914_weights1 --root /home/ies/hyu/data/Training_set_0814_noise/ --bolt_weight 1
```
### Test the Mdel
You can use the following command to test your model after trained with our synthetic dataset:
```
CUDA_VISIBLE_DEVICES=0,1 python test_semseg_new.py --log_dir pointnet2_sem_seg_0831 --root /home/ies/hyu/data/Training_set_0827_noiseNoCover/ --visual --test_area Validation
```
You can use following command to test the model with Zivid dataset:
```
CUDA_VISIBLE_DEVICES=2 python test_semseg_new.py --log_dir pointnet2_sem_seg_0814 --root /home/ies/hyu/data/Zivid_Testset/labeled_transformed_0909/ --visual --test_area Test
```
Note that the `--test_area` is a string from the dataset. And you need to check the data before testing. If there is no 'Test' or 'Validation' work in the name of data, please add it and then start to test.

### Pretrained Model
It's already placed in ./log/sem_seg/pointnet2_sem_seg.

## Reference by
[halimacc/pointnet3](https://github.com/halimacc/pointnet3)<br>
[fxia22/pointnet.pytorch](https://github.com/fxia22/pointnet.pytorch)<br>
[charlesq34/PointNet](https://github.com/charlesq34/pointnet) <br>
[charlesq34/PointNet++](https://github.com/charlesq34/pointnet2)
