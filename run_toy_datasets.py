from utils import *

seed = 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--number_points", default=2000, type=int, help="number of data points in the dataset")
    parser.add_argument("--var", default=15, type=float, help="variance of the train dataset")
    parser.add_argument("--epochs", default=500, type=int, help="number of epochs for the training")

    args = parser.parse_args()

    test_performance_ADAM_ASAM_SAM(args.number_points, args.var, args.epochs)

    
