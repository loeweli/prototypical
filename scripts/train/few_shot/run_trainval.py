import argparse

from trainval import main

parser = argparse.ArgumentParser(description='Re-run prototypical networks training in trainval mode')

parser.add_argument('--model.model_path', type=str, default=r'D:\project\prototypical_networks_master\results\trace.txt', metavar='MODELPATH',
                    help="location of pretrained model to retrain in trainval mode")

args = vars(parser.parse_args())

main(args)
