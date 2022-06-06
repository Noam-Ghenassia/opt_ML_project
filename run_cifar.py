from utils_cifar import *

seed = 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default='Net', type=str, help="Name of model architecure : Net or wrn28_10")
    parser.add_argument("--minimizer", default='ASAM', type=str, help="ASAM, SAM or ADAM")
    parser.add_argument("--batch_size", default=128, type=int, help="Batch size")
    parser.add_argument("--epochs", default=200, type=int, help="Number of epochs.")
    parser.add_argument("--smoothing", default=0.1, type=float, help="Label smoothing.")
    parser.add_argument("--rho", default=0.5, type=float, help="Rho for ASAM and SAM")
    args = parser.parse_args()
    assert args.minimizer in ['ASAM', 'SAM', 'ADAM'], f"Invalid minimizer type. Please select ASAM, SAM or ADAM"
    assert args.model in ['Net', 'wrn28_10'], f"Invalid model. Please select Net or wrn28_10"

    #Define data loaders
    trainloader, testloader, classes = load_cifar(batch_size=args.batch_size)
    #Define the model
    if(args.model=='wrn28_10') :
        torch.manual_seed(seed)
        net = wrn28_10(num_classes=10).to(device)
    else :
        torch.manual_seed(seed)
        net = Net().to(device)
    #Train with the chosen minimizer
    if(args.minimizer == 'ADAM') :
        train_ADAM(trainloader, testloader, model=net, epochs=args.epochs)
    else :
        train_minimizer(trainloader, testloader, model=net, minimizer=eval(args.minimizer), epochs=args.epochs, rho_=args.rho, smoothing_=args.smoothing)
