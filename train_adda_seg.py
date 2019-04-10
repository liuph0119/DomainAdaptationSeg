import numpy as np
import os
from keras.optimizers import Adam, SGD
from models.adda_seg_model import set_trainability, ADDA_Seg_Model
from models.discriminator import Discriminator
from models.deeplab_v3p import Deeplab_v3p
from models.net_utils import load_custom_model
from argparse import ArgumentParser
from utils.image_generator import get_image_label, get_images


def parse_args():
    ap = ArgumentParser()
    ap.add_argument("data_dir", type=str, help="the data set directory")
    ap.add_argument("source_image_dir", type=str)
    ap.add_argument("source_label_dir", type=str)
    ap.add_argument("target_image_dir", type=str)
    ap.add_argument("model_name", type=str)
    ap.add_argument("--optimizer", default="adam")
    ap.add_argument("--base_learning_rate", default=1e-4, type=float)
    ap.add_argument("--min_learning_rate", default=1e-7, type=float)
    ap.add_argument("--image_width", default=256, type=int)
    ap.add_argument("--image_height", default=256, type=int)
    ap.add_argument("--image_channel", default=3, type=int)
    ap.add_argument("--color_mode", default="color", type=str)
    ap.add_argument("--image_suffix", default=".png", type=str)
    ap.add_argument("--label_suffix", default=".png", type=str)
    ap.add_argument("--n_class", default=2, type=int)
    ap.add_argument("--batch_size", default=2, type=int)
    ap.add_argument("--iterations", default=500, type=int)
    ap.add_argument("--weight_decay", default=1e-4, type=float)
    ap.add_argument("--initializer", default="he_normal", type=str)
    ap.add_argument("--bn_epsilon", default=1e-3, type=float)
    ap.add_argument("--bn_momentum", default=0.99, type=float)
    ap.add_argument("--pre_trained_model", default="", type=str)
    ap.add_argument("--source_fname_file", default="", type=str)
    ap.add_argument("--target_fname_file", default="", type=str)
    ap.add_argument("--logs_dir", default="./logs", type=str)
    ap.add_argument("--augmentations", default="", type=str)
    ap.add_argument("--display", default=1, type=int)
    ap.add_argument("--snapshot", default=10, type=int)
    return ap.parse_args()


def train_adda_seg_main(args):
    if os.path.exists(args.pre_trained_model):
        print(">>>>>>>> load generator model from ", args.pre_trained_model)
        G = load_custom_model(args.pre_trained_model)
    else:
        G = Deeplab_v3p(input_shape=(args.image_height, args.image_width, args.image_channel),
                        n_class=args.n_class,
                        encoder_weights=None,
                        weight_decay=args.weight_decay,
                        kernel_initializer=args.initializer,
                        bn_epsilon=args.bn_epsilon,
                        bn_momentum=args.bn_momentum
                        )
    D = Discriminator((args.image_height, args.image_width, args.n_class), n_filters=64, activation_fn="relu")

    optimizer = Adam(args.base_learning_rate) if args.optimizer.lower() is "adam" else SGD(args.base_learning_rate)
    G.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])
    D.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])
    adda_seg_model = ADDA_Seg_Model((args.image_height, args.image_width, args.image_channel), G, D)
    adda_seg_model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])

    source_urls = [line.strip() for line in open(args.source_fname_file, "r", encoding="utf-8")]
    target_urls = [line.strip() for line in open(args.target_fname_file, "r", encoding="utf-8")]

    for i in range(args.iterations):
        # train D
        # 设置D可训练，判别 source 和 target.txt 的images经过G映射后得到的P，经过D能区分开是0/1
        set_trainability(D, True)

        source_images, source_labels = get_image_label(args.data_dir+"/"+args.source_image_dir,
                                                       args.data_dir+"/"+args.source_label_dir,
                                                       source_urls, args, shuffle=True)
        source_P = G.predict(source_images)

        target_images = get_images(args.data_dir+"/"+args.target_image_dir, target_urls, args, shuffle=True)
        target_P = G.predict(target_images)

        combined_Ps = np.concatenate((target_P, source_P))
        combined_labels = np.concatenate((np.ones((args.batch_size, args.image_height//32, args.image_width//32, 1)),
                                          np.zeros((args.batch_size, args.image_height//32, args.image_width//32, 1))))

        d_loss = D.train_on_batch(combined_Ps, combined_labels)


        # 固定住D，训练G和Gan
        set_trainability(D, False)
        combined_images = np.concatenate((target_images, source_images))
        # G要让D认为两者都是source label的，所以两类标签都是0
        misleading_targets = np.zeros((args.batch_size*2, args.image_height//32, args.image_width//32, 1))

        g_loss = G.train_on_batch(source_images, source_labels)
        a_loss = adda_seg_model.train_on_batch(combined_images, misleading_targets)

        if i>0 and (i+1)%args.display==0:
            print("# iteration{}: generator loss={}, discriminator loss={}, advertiser loss={}".format(i, np.mean(g_loss), np.mean(d_loss), np.mean(a_loss)))

        if i>0 and (i+1)%args.snapshot==0:
            G.save(args.logs_dir+"/checkpoints/"+args.model_name)
            #adda_seg_model.save(args.logs_dir+"/checkpoints/"+args.model_name)


# python.exe ./train_adda_seg.py ./data/inria_test source_image source_label_index target_image adda_deeplab_v3p.h5 --optimizer adam --base_learning_rate 1e-4 --min_learning_rate 1e-7 --image_width 256 --image_height 256 --image_channel 3 --image_suffix .png --label_suffix .png --n_class 2 --batch_size 2 --iterations 50 --weight_decay 1e-4 --initializer he_normal --bn_epsilon 1e-3 --bn_momentum 0.99 --pre_trained_model ./logs/checkpoints/deeplab_v3p_base.h5 --source_fname_file ./data/inria_test/source.txt --target_fname_file ./data/inria_test/target.txt --logs_dir ./logs --augmentations flip_x,flip_y,random_crop --display 1 --snapshot 5
if __name__ == "__main__":
    args = parse_args()
    train_adda_seg_main(args)