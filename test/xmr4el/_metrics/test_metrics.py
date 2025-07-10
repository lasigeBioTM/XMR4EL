from xmr4el._metrics.metrics import Metrics

def main():
    
    path = "_REEL/REEL/bc5cdr_Disease_medic/xlinker_preds.tsv"
    
    metric = Metrics.test(path)

main()