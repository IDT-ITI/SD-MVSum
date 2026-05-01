import argparse
from model.configs import Config, str2bool
from model.data_loader import get_loader
from model.solver import Solver


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_num', type=str, default='exp1', help='Creates the dir to save results')
    parser.add_argument('--epochs', type = int, default = 50, help = 'the number of training epochs')
    parser.add_argument('--lr', type = float, default = 5e-5, help = 'the learning rate')
    parser.add_argument('--l2_reg', type = float, default = 1e-4, help = 'l2 regularizer')
    parser.add_argument('--dropout_ratio', type = float, default = 0.5, help = 'the dropout ratio')
    parser.add_argument('--batch_size', type = int, default = 64, help = 'the batch size')
    parser.add_argument('--tag', type = str, default = 'dev', help = 'A tag for experiments')
    parser.add_argument('--ckpt_path', type = str, default = 'None', help = 'checkpoint path for inference or weight initialization')
    parser.add_argument('--clip', type=float, default=5.0, help='Max norm of the gradients')
    parser.add_argument('--train', type=str2bool, default='True', help='True for training and evaluation | False for testing a pretrained model')
    parser.add_argument('--seed', type=int, default=12345, help='Chosen seed for generating random numbers')
    parser.add_argument('--init_type', type=str, default="xavier", help='Weight initialization method')
    parser.add_argument('--init_gain', type=float, default=None, help='Scaling factor for the initialization methods')

    parser.add_argument('--input_size', type=int, default=512, help='Feature size expected in the input')
    parser.add_argument('--text_size', type=int, default=512, help='Feature size expected in the input')
    parser.add_argument('--pos_enc', type=str2bool, default=True, help="If positional encoding will be used")
    parser.add_argument('--heads', type=int, default=8, help="Number of heads for the cross-attention module")
    parser.add_argument('--annotations', type=int, default=1, help="Number of annotations per video (in SM-MrHiSum = 1)")
    parser.add_argument('--visual_weights', type=str2bool, default=True, help="Enable cross-attention between visual features and text features.")
    parser.add_argument('--transcript_weights', type=str2bool, default=True, help="Enable cross-attention between transcript features and text features.")

    parser.add_argument('--dataset', type=str, default='SM_MrHiSum', help="The name of the dataset")

    opt = parser.parse_args()

    kwargs = vars(opt)
    config = Config(**kwargs)

    print(f"Running experiment: {config.exp_num}")

    train_loader = get_loader('train', dataset=config.dataset)
    val_loader = get_loader('val', dataset=config.dataset)
    test_loader = get_loader('test', dataset=config.dataset)

    solver = Solver(config, train_loader, val_loader, test_loader)

    solver.build()
    test_model_ckpt_path = None

    if config.train:
        best_f1_ckpt_path = solver.train()
        solver.test(best_f1_ckpt_path)
    else:
        test_model_ckpt_path = config.ckpt_path
        if test_model_ckpt_path is None:
            print("Trained model checkpoint required. Exit program")
            exit()
        else:
            solver.test(test_model_ckpt_path)
