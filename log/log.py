import datetime
import os

def get_log_info(args):
    logInfo = open("./data/output/logInfo_{}_{}.txt".format(args.model.replace("/", "_"), args.dataset), mode="w",encoding="utf-8")

    return logInfo

def get_knock_out_info(args):
    output_dir = './data/output'
    os.makedirs(output_dir, exist_ok=True)

    if args.knock_out:
        logInfo = open("./data/output/knock_out_logInfo_{}_{}_{}.txt".format(args.model.replace("/", "_"), args.dataset, datetime.datetime.now().strftime("%Y%m%d%H%M%S")),  mode="w",encoding="utf-8")
    else:
        logInfo = open("./data/output/logInfo_{}_{}.txt".format(args.model.replace("/", "_"), args.dataset), mode="w",encoding="utf-8")
    return logInfo