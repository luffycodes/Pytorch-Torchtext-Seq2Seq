import codecs

src_sent = open('/home/zoro/Documents/OpenStaxData/Datasets/stsbenchmark/sts.en', 'w')
trg_sent = open('/home/zoro/Documents/OpenStaxData/Datasets/stsbenchmark/sts.fr', 'w')
correlation = open('/home/zoro/Documents/OpenStaxData/Datasets/stsbenchmark/correlation.txt', 'w')
sts_tab = open('/home/zoro/Documents/OpenStaxData/OtherGitRepoData/Pytorch-Torchtext-Seq2Seq/data/stsTab.txt', 'w')
data = codecs.open('/home/zoro/Documents/OpenStaxData/Datasets/stsbenchmark/sts-train.csv', 'r', 'UTF-8')
correlation_arr = []
src_sent_arr = []
trg_sent_arr = []

for i, line in enumerate(data):
    line = line.split('\t')
    if len(line) < 7:
        print(line)

    correlation_arr.append(line[4].strip())
    src_sent_arr.append(line[5].strip())
    trg_sent_arr.append(line[6].strip())

sorted_src_sent_arr_idx_list = sorted(range(len(src_sent_arr)), key=lambda x: len(src_sent_arr[x]), reverse=True)
sorted_src_sent_arr = [src_sent_arr[i] for i in sorted_src_sent_arr_idx_list]
sorted_trg_sent_arr = [trg_sent_arr[i] for i in sorted_src_sent_arr_idx_list]
sorted_correlation_arr = [correlation_arr[i] for i in sorted_src_sent_arr_idx_list]

for i, line in enumerate(src_sent_arr):
    if i != len(src_sent_arr) -1:
        newLine = '\n'
    else:
        newLine = ''
    # correlation.write(sorted_correlation_arr[i]+newLine)
    # src_sent.write(sorted_src_sent_arr[i]+newLine)
    # trg_sent.write(sorted_trg_sent_arr[i]+newLine)

    # For Tab separated data
    sts_tab.write(str(int(float(sorted_correlation_arr[i]) * 100))+'\t')
    sts_tab.write(sorted_src_sent_arr[i]+'\t')
    sts_tab.write(sorted_trg_sent_arr[i]+newLine)

sts_tab.close()
# correlation.close()
# src_sent.close()
# trg_sent.close()
