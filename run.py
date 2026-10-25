import torch
from Utilities import get_args, get_loaders, ConvNet, Trainer, plot


def main():
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_loader, test_loader = get_loaders(args.batch_size)
    model = ConvNet().to(device)
    print(model)

    trainer = Trainer(model, train_loader, test_loader, args.learning_rate, device)
    loss_list, acc_train, acc_test = trainer.train(args.num_epochs)
    plot(loss_list, acc_train, acc_test)


if __name__ == '__main__':
    main()