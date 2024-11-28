def get_log_info(args):
    logInfo = open("./data/output/logInfo_{}_{}.txt".format(args.model, args.dataset), mode="w",encoding="utf-8")