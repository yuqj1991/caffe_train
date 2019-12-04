import sys
import argparse

alignImgDir = "../../../../../dataset/reid_data/combineData/val/"

# romove 1 when same remove 0 when dissame
def get_all_images(filename):
    file = open(filename)
    lines = file.readlines()
    list = []
    for line in lines:
        line_split = line.strip("\n").split("\t")
        if len(line_split) == 3:
            line_split[-1] = line_split[-1].zfill(4)
            line_split[-2] = line_split[-2].zfill(4)
        elif len(line_split) == 4:
            line_split[-1] = line_split[-1].zfill(4)
            line_split[-3] = line_split[-3].zfill(4)
        list.append(line_split)
    file.close()
    return list

def save2AbspathLabelfile(list, labelpath, gtPath):
    file = open(labelpath, "w")
    gt_file = open(gtPath, "w")
    labellines=[]
    gtlines =[]
    for i in range(len(list)):
        if len(list[i]) == 3:
            labelline = alignImgDir + list[i][0] + "/" + list[i][0] + "_" + list[i][1] + ".jpg" + ' ' \
                        + alignImgDir + list[i][0] + "/" + list[i][0] + "_" + list[i][2] + ".jpg" + "\n"
            labellines.append(labelline)
            gtlines.append('1' + ' ' + labelline)
        elif len(list[i]) == 4:
            labelline = alignImgDir + list[i][0] + "/" + list[i][0] + "_" + list[i][1] + ".jpg" + ' ' \
                        + alignImgDir + list[i][2] + "/" + list[i][2] + "_" + list[i][3] + ".jpg" + "\n"
            labellines.append(labelline)
            gtlines.append('0' + ' ' + labelline)
    file.writelines(labellines)
    file.close()
    gt_file.writelines(gtlines)
    gt_file.close()
    
def save2ContraviepathLabelfile(list, labelpath, gtPath):
    file = open(labelpath, "w")
    gt_file = open(gtPath, "w")
    labellines=[]
    gtlines =[]
    for i in range(len(list)):
        if len(list[i]) == 3:
            labelline = list[i][0] + "/" + list[i][1] \
                              + ' ' + list[i][0] + "/" + list[i][2] + "\n"
            labellines.append(labelline)
            gtlines.append('1' + ' ' + labelline)
        elif len(list[i]) == 4:
            labelline = list[i][0] + "/" + list[i][1] \
                              + ' ' + list[i][2] + "/" + list[i][3] + "\n"
            labellines.append(labelline)
            gtlines.append('0' + ' ' + labelline)
    file.writelines(labellines)
    file.close()
    gt_file.writelines(gtlines)
    gt_file.close()


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--pair_file', type=str, help='File of face pairs list')
    parser.add_argument('--result_file', type=str, help='Result of face recognition')
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_arguments(sys.argv[1:])
    list = get_all_images(args.pair_file)
    gtPath = "gt_" + args.result_file
    save2ContraviepathLabelfile(list, args.result_file, gtPath)
    print("Done!")
