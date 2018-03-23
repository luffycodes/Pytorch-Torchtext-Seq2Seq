import argparse
import os
import time
from torch.backends import cudnn
import configparser as con

from prepro import *
from trainer import *
from torchtext import data as dt


def main(args):
    cuda.set_device(int(args.gpu_num))
    cudnn.benchmark = True

    # logging setting
    console_logger = logging.getLogger()
    console_logger.setLevel(logging.DEBUG)
    consoleHandler = logging.StreamHandler()
    logFormatter = logging.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")
    consoleHandler.setFormatter(logFormatter)
    console_logger.addHandler(consoleHandler)

    fileHandler = logging.FileHandler("{0}/{1}.log".format(args.log_path, "console"))
    fileHandler.setFormatter(logFormatter)
    console_logger.addHandler(fileHandler)

    console_logger.propagate = False

    # Load dataset
    train_file = os.path.join(args.data_path, "data_{}_{}_{}_{}.json".format(args.dataset, args.src_lang,
                                                                             args.trg_lang, args.max_len))
    val_file = os.path.join(args.data_path, "data_dev_{}_{}_{}.json".format(args.src_lang, args.trg_lang, args.max_len))

    start_time = time.time()
    if os.path.isfile(train_file) and os.path.isfile(val_file):
        console_logger.debug("Loading data..")
        dp = DataPreprocessor(args.src_lang, args.trg_lang)
        train_dataset, val_dataset, vocabs = dp.load_data(train_file, val_file)
    else:
        console_logger.debug("Preprocessing data..")
        dp = DataPreprocessor(args.src_lang, args.trg_lang)
        train_dataset, val_dataset, vocabs = dp.preprocess(args.train_path, args.val_path, train_file, val_file,
                                                           args.src_lang, args.trg_lang, args.max_len)

    console_logger.debug("Elapsed Time: %1.3f \n" % (time.time() - start_time))

    console_logger.debug("=========== Data Stat ===========")
    console_logger.debug("Train: %d", len(train_dataset))
    console_logger.debug("val: %d", len(val_dataset))
    console_logger.debug("=================================")

    train_loader = dt.BucketIterator(dataset=train_dataset, batch_size=args.batch_size,
                                     repeat=False, shuffle=True, sort_within_batch=True,
                                     sort_key=lambda x: len(x.src), device=args.gpu_num)
    val_loader = dt.BucketIterator(dataset=val_dataset, batch_size=args.batch_size,
                                   repeat=False, shuffle=True, sort_within_batch=True,
                                   sort_key=lambda x: len(x.src), device=args.gpu_num)

    trainer = Trainer(train_loader, val_loader, vocabs, args)
    trainer.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--server', type=bool, default=not os.path.isdir("/home/zoro"))
    server = parser.parse_args().server

    config = con.ConfigParser()
    config.sections()
    if server:
        config.read('/root/pythonProjects/Pytorch-Torchtext-Seq2Seq/configFile.ini')
        MACHINE = "SERVER"
        gpu = 2
    else:
        config.read('/home/zoro/PycharmProjects/OtherGitRepo/Pytorch-Torchtext-Seq2Seq/configFile.ini')
        MACHINE = "DEFAULT"
        gpu = -1

    # Language setting
    parser.add_argument('--dataset', type=str, default='europarl')
    parser.add_argument('--src_lang', type=str, default='en')
    parser.add_argument('--trg_lang', type=str, default='fr')
    parser.add_argument('--max_len', type=int, default=70)

    # Model hyper-parameters
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--grad_clip', type=float, default=5)
    parser.add_argument('--num_layer', type=int, default=1)
    parser.add_argument('--bi_dir', type=bool, default=False)
    parser.add_argument('--embed_dim', type=int, default=512)
    parser.add_argument('--hidden_dim', type=int, default=1024)

    # Training setting
    parser.add_argument('--batch_size', type=int, default=40)
    parser.add_argument('--num_epoch', type=int, default=100)

    # Path
    data = config[MACHINE]['translation_data_location']
    parser.add_argument('--data_path', type=str, default=('%s/' % data))
    args = parser.parse_args()
    parser.add_argument('--train_path', type=str, default=('%s/training/europarl-v7.%s-%s' % (data, args.trg_lang, args.src_lang)))
    parser.add_argument('--val_path', type=str, default=('%s/dev/newstest2013' % data))

    model_results = config[MACHINE]['translation_model_location']
    parser.add_argument('--log_path', type=str, default=('%s/' % model_results))

    # Dir.
    parser.add_argument('--log', type=str, default='tfEvents')
    parser.add_argument('--sample', type=str, default='sample')

    # Misc.
    parser.add_argument('--gpu_num', type=int, default=gpu)

    args = parser.parse_args()
    print(args)
    main(args)
