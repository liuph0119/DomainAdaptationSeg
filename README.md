# DomainAdaptationSeg
Domain Adaptation for semantic segmentation.

Domain Adaptation is of significant importance in model transferring for different datasets. Usually, initializing the model with pre-trained weights and fine-tuning is useful in transfer learning. However, adequate labels are needed in such a strategy. We know that in the field of semantic segmentation, labeling is a laborious task and thus pixel-wise labels are hard to obtain. In our implementation, no labels in the target domain are needed but a quite excellent model can be trained to apply to the target domain dataset.

Suppose we have trained a base model, i.e., G0, in the source domain. Thereafter, the domain adaptation training includes two parts:

0. Initialize the G with the weights of G0. Input the prediction of G to a discriminator, i.e., D, and produce the domain label of the inputs.
1. Set D trainable, and train D to be discriminative.
2. Set D untrainable, and train the G to optimize the segmentation performance in the source domain, meanwhile produce predictions that make D been cheated

## Usage

### 1. Train G on source domain 

```powershell
python.exe ./train_adda_seg.py ./data/inria_test source_image source_label_index target_image adda_deeplab_v3p.h5 --optimizer adam --base_learning_rate 1e-4 --min_learning_rate 1e-7 --image_width 256 --image_height 256 --image_channel 3 --image_suffix .png --label_suffix .png --n_class 2 --batch_size 2 --iterations 50 --weight_decay 1e-4 --initializer he_normal --bn_epsilon 1e-3 --bn_momentum 0.99 --pre_trained_model ./logs/checkpoints/deeplab_v3p_base.h5 --source_fname_file ./data/inria_test/source.txt --target_fname_file ./data/inria_test/target.txt --logs_dir ./logs --augmentations flip_x,flip_y,random_crop --display 1 --snapshot 5
```

### 2. Train G&D considering both the source and target domain

```powershell
python.exe ./train_adda_seg.py ./data/inria_test source_image source_label_index target_image adda_deeplab_v3p.h5 --optimizer adam --base_learning_rate 1e-4 --min_learning_rate 1e-7 --image_width 256 --image_height 256 --image_channel 3 --image_suffix .png --label_suffix .png --n_class 2 --batch_size 2 --iterations 50 --weight_decay 1e-4 --initializer he_normal --bn_epsilon 1e-3 --bn_momentum 0.99 --pre_trained_model ./logs/checkpoints/deeplab_v3p_base.h5 --source_fname_file ./data/inria_test/source.txt --target_fname_file ./data/inria_test/target.txt --logs_dir ./logs --augmentations flip_x,flip_y,random_crop --display 1 --snapshot 5
```

