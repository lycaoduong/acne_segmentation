import argparse
import nni
from utils.utils import Dict2Class
from train import train_model

def get_args():
    parser = argparse.ArgumentParser('Pytorch multi backbone segmentation source')
    parser.add_argument('-p', '--dataset', type=str, default='tri_acne', help='Select dataset')
    parser.add_argument('-m', '--model', type=str, default='u2net', help='Choosing model for training')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')

    parser.add_argument('-lt', '--loss_type', type=int, default=0, help='Choosing loss function')
    parser.add_argument('-size', '--input_size', type=int, default=512, help='Input size')
    parser.add_argument('-lr', '--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('-ep', '--epochs', type=int, default=300, help='Select epochs')
    parser.add_argument('-ot', '--optimizer', type=str, default='sgd', help='Select Optimizer')

    args = parser.parse_args()
    return args

def dic_opt():
    parse_dic = {
        'project': 'pytorch_segmentation',
        'dataset': 'tri_acne',
        'device': '',
        'epochs': 300
    }
    return parse_dic


if __name__ == '__main__':
    ###################################### Train from NNi search space
    nni_params = nni.get_next_parameter()
    parse_params = dic_opt()
    nni_params.update(parse_params)
    train_opt = Dict2Class(nni_params)

    ###################################### Train from Argparse
    # train_opt = get_args()

    # Start Train
    trainer = train_model(train_opt)
    trainer.start()