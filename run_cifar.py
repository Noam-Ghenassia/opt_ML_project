from utils_cifar.py import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default='net', type=str, help="Name of model architecure : Net or wrn28_10")
    parser.add_argument("--minimizer", default='ASAM', type=str, help="ASAM, SAM or ADAM")
    parser.add_argument("--batch_size", default=128, type=int, help="Batch size")
    parser.add_argument("--epochs", default=200, type=int, help="Number of epochs.")
    parser.add_argument("--smoothing", default=0.1, type=float, help="Label smoothing.")
    args = parser.parse_args()
    assert args.minimizer in ['ASAM', 'SAM', 'ADAM'], f"Invalid minimizer type. Please select ASAM, SAM or ADAM"
    assert args.model in ['Net', 'wrn28_10'], f"Invalid model. Please select Net or wrn28_10"

    #Define data loaders
    trainloader, testloader, classes = load_cifar(batch_size=args.batch_size)
    #Define the model
    if(model=='wrn28_10') :
        net = wrn28_10(num_classes=10).to(device)
    else :
        net = Net().to(device)
    #Train with the chosen minimizer
    if(args.minimizer == 'ADAM') :
        train_ADAM(trainloader, testloader, model=args.model, epochs=args.epochs)
    else :
        train_minimizer(trainloader, testloader, model=args.model, minimizer=eval(args.minimizer), epochs=args.epochs, rho_=0.5, smoothing_=args.smoothing)
