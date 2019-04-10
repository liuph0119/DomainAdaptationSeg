from keras.callbacks import LearningRateScheduler, TensorBoard, ModelCheckpoint
from keras.optimizers import Adam, SGD

from argparse import ArgumentParser
from models.deeplab_v3p import Deeplab_v3p
from utils.image_generator import *
from models.net_utils import load_custom_model


def parse_args():
    ap = ArgumentParser()
    ap.add_argument("data_dir", type=str)
    ap.add_argument("image_dir", type=str)
    ap.add_argument("label_dir", type=str)
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
    ap.add_argument("--epoch", default=100, type=int)
    ap.add_argument("--weight_decay", default=1e-4, type=float)
    ap.add_argument("--initializer", default="he_normal", type=str)
    ap.add_argument("--bn_epsilon", default=1e-3, type=float)
    ap.add_argument("--bn_momentum", default=0.99, type=float)
    ap.add_argument("--pre_trained_model", default="", type=str)
    ap.add_argument("--train_fname_file", default="", type=str)
    ap.add_argument("--val_fname_file", default="", type=str)
    ap.add_argument("--logs_dir", default="./logs", type=str)
    ap.add_argument("--augmentations", default="", type=str)
    ap.add_argument("--verbose", default=1, type=int)
    return ap.parse_args()



def train_seg_main(args):
    # build or load model
    if args.pre_trained_model is not None and os.path.exists(args.pre_trained_model):
        print(">>>>>>>> load model from ", args.pre_trained_model)
        seg_model = load_custom_model(args.pre_trained_model)
    else:
        seg_model = Deeplab_v3p(input_shape=(args.image_height, args.image_width, args.image_channel),
                                n_class=args.n_class,
                                encoder_weights=None,
                                weight_decay=args.weight_decay,
                                kernel_initializer=args.initializer,
                                bn_epsilon=args.bn_epsilon,
                                bn_momentum=args.bn_momentum
                                )
    seg_model.summary()

    # optimizer
    def learning_rate_schedule(_epoch):
        lr_base = args.base_learning_rate
        lr = lr_base * ((1 - float(_epoch) / args.epoch) ** 0.9)
        return lr

    optimizer = Adam(args.base_learning_rate) if args.optimizer.lower() is "adam" else SGD(args.base_learning_rate)
    seg_model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])

    # callbacks
    callbacks = []
    log_path = os.path.join(args.logs_dir, "logs")
    ckp_path = os.path.join(args.logs_dir, "checkpoints")
    if not os.path.exists(log_path):
        os.mkdir(log_path)
    if not os.path.exists(ckp_path):
        os.mkdir(ckp_path)
    callbacks.append(ModelCheckpoint(ckp_path+"/"+args.model_name, save_best_only=True, verbose=1))
    callbacks.append(TensorBoard(log_dir=log_path))
    callbacks.append(LearningRateScheduler(schedule=learning_rate_schedule, verbose=1))

    # generator
    train_fnames = [line.strip() for line in open(args.train_fname_file, "r", encoding="utf-8")]
    val_fnames = [line.strip() for line in open(args.val_fname_file, "r", encoding="utf-8")]

    train_gen = segmentation_generator(train_fnames, args, shuffle=True)
    val_gen = segmentation_generator(val_fnames, args)
    seg_model.fit_generator(train_gen,
                            steps_per_epoch=len(train_fnames)//args.batch_size,
                            epochs=args.epoch,
                            verbose=args.verbose,
                            callbacks=callbacks,
                            validation_data=val_gen,
                            validation_steps=len(val_fnames)//args.batch_size
                            )


# python.exe ./train_seg.py ./data/inria_test source_image source_label_index deeplab_v3p_base.h5 --optimizer adam --base_learning_rate 1e-4 --min_learning_rate 1e-7 --image_width 256 --image_height 256 --image_channel 3 --image_suffix .png --label_suffix .png --n_class 2 --batch_size 2 --epoch 50 --weight_decay 1e-4 --initializer he_normal --bn_epsilon 1e-3 --bn_momentum 0.99 --pre_trained_model ./logs/checkpoints/deeplab_v3p_base.h5 --train_fname_file ./data/inria_test/train.txt --val_fname_file ./data/inria_test/val.txt --logs_dir ./logs --augmentations flip_x,flip_y,random_crop --verbose 1
if __name__ == "__main__":
    args = parse_args()
    train_seg_main(args)