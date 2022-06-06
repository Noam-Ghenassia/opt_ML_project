from cifar.py import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--minimizer", default='ASAM', type=str, help="ASAM, SAM or ADAM")
    parser.add_argument("--batch_size", default=128, type=int, help="Batch size")
    parser.add_argument("--epochs", default=50, type=int, help="Number of epochs.")
    parser.add_argument("--smoothing", default=0.1, type=float, help="Label smoothing.")
    args = parser.parse_args()
    assert args.minimizer in ['ASAM', 'SAM', 'ADAM'], \
            f"Invalid minimizer type. Please select ASAM or SAM"

    trainloader, testloader, classes = load_cifar(batch_size=args.batch_size)
    net = wrn28_10(num_classes=10).to(device)
    if(args.minimizer == 'ADAM') :
        train_ADAM(trainloader, testloader, model=net, epochs=1)
    else :
        train_minimizer(trainloader, testloader, model=net, minimizer=eval(args.minimizer), epochs=1, rho_=0.5, smoothing_=args.smoothing)
