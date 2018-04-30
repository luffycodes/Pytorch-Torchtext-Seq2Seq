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
    start_time = time.time()
    max_len = args.max_len

    # Read correlation file
    correlation = []
    correlation_file_path = os.path.join(args.data_path, "correlation.txt")
    with open(correlation_file_path) as correlation_file:
        for line in correlation_file:
            correlation.append(float(line))

    # STS dataset
    sts_dp = DataPreprocessor()
    sts_dp.src_field, sts_dp.trg_field = generate_fields(args.sts_src_lang, args.sts_src_lang)
    sts_fields = [("sim", dt.Field(sequential=False, use_vocab=False)),
                  ("src", sts_dp.src_field),
                  ("trg", sts_dp.trg_field),
                  ]
    sts_file = os.path.join(args.data_path, "data_sts_{}_{}.json".format(args.sts_src_lang, args.sts_trg_lang))
    sts_dataset = dt.TabularDataset(path=args.data_path + "stsTab.txt", format="TSV", fields=sts_fields)
    sts_loader = dt.Iterator(dataset=sts_dataset, batch_size=args.batch_size,
                             repeat=False, shuffle=False, device=args.gpu_num)

    src_lang = args.src_lang
    trg_lang = args.trg_lang

    # Validation dataset
    val_dp = DataPreprocessor()
    val_dp.src_field, val_dp.trg_field = generate_fields(args.src_lang, args.trg_lang)
    val_file = os.path.join(args.data_path, "data_dev_{}_{}_{}.json".format(src_lang, trg_lang, max_len))
    val_dataset = val_dp.getOneDataset(args.val_path, val_file, src_lang, trg_lang, max_len)

    # Training dataset
    train_dp = DataPreprocessor()
    train_dp.src_field, train_dp.trg_field = generate_fields(args.src_lang, args.trg_lang)
    train_file = os.path.join(args.data_path, "data_{}_{}_{}_{}.json".format(args.dataset, src_lang, trg_lang, max_len))
    train_dataset = train_dp.getOneDataset(args.train_path, train_file, src_lang, trg_lang, max_len)

    # Building vocab
    vocabs = train_dp.buildVocab(train_dataset, sts_dataset)

    val_dp.src_field.vocab = train_dp.src_field.vocab
    val_dp.trg_field.vocab = train_dp.trg_field.vocab

    sts_dp.src_field.vocab = train_dp.src_field.vocab
    sts_dp.trg_field.vocab = train_dp.src_field.vocab

    console_logger.debug("Elapsed Time: %1.3f \n" % (time.time() - start_time))

    console_logger.debug("=========== Data Stat ===========")
    console_logger.debug("Train: %d", len(train_dataset))
    console_logger.debug("Val: %d", len(val_dataset))
    console_logger.debug("STS: %d", len(sts_dataset))
    console_logger.debug("=================================")

    train_loader = dt.BucketIterator(dataset=train_dataset, batch_size=args.batch_size,
                                     repeat=False, shuffle=True, sort_within_batch=True,
                                     sort_key=lambda x: len(x.src), device=args.gpu_num)
    val_loader = dt.BucketIterator(dataset=val_dataset, batch_size=args.batch_size,
                                   repeat=False, shuffle=True, sort_within_batch=True,
                                   sort_key=lambda x: len(x.src), device=args.gpu_num)

    trainer = Trainer(train_loader, val_loader, sts_loader, vocabs, correlation, args)
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
        gpu = 1
    else:
        config.read('/home/zoro/PycharmProjects/OtherGitRepo/Pytorch-Torchtext-Seq2Seq/configFile.ini')
        MACHINE = "DEFAULT"
        gpu = -1

    # Language setting
    parser.add_argument('--dataset', type=str, default='europarl')
    parser.add_argument('--src_lang', type=str, default='en')
    parser.add_argument('--trg_lang', type=str, default='fr')
    parser.add_argument('--sts_src_lang', type=str, default='en')
    parser.add_argument('--sts_trg_lang', type=str, default='fr')
    parser.add_argument('--max_len', type=int, default=70)

    # Model hyper-parameters
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--grad_clip', type=float, default=5)
    parser.add_argument('--num_layer', type=int, default=1)
    parser.add_argument('--bi_dir', type=bool, default=False)
    parser.add_argument('--embed_dim', type=int, default=512)
    parser.add_argument('--hidden_dim', type=int, default=1024)

    # Training setting
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--num_epoch', type=int, default=100)

    # Path
    data = config[MACHINE]['translation_data_location']
    parser.add_argument('--data_path', type=str, default=('%s/' % data))
    args = parser.parse_args()
    parser.add_argument('--train_path', type=str,
                        default=('%s/training/europarl-v7.%s-%s' % (data, args.trg_lang, args.src_lang)))
    parser.add_argument('--val_path', type=str, default=('%s/dev/newstest2013' % data))
    parser.add_argument('--sts_path', type=str, default=('%s/sts/sts' % data))

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
