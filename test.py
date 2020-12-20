import argparse
import torch
import os
import numpy as np
import json
from myRes import vgg16dres
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter


def parse_args():
    parser = argparse.ArgumentParser(description='Test ')
    parser.add_argument('--device', default='0', help='assign device')
    parser.add_argument('--model-path', type=str, default='ckpts/model_qnrf.pth',
                        help='model path to test')
    parser.add_argument('--dataset', help='dataset name', default='shb')
    parser.add_argument('--pred-density-map', type=str, default='',
                        help='save predicted density maps when this is not empty.')
    args = parser.parse_args()

    with open('args/dataset_paths.json') as f:
        # TODO: Check args
        dataset_paths = json.load(f)[args.dataset]
    # load default dataset configurations from datasets/dataset_cfg.json
    args = {**vars(args), **dataset_paths}
    return args


if __name__ == '__main__':
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args['device'].strip()  # set vis gpu
    device = torch.device('cuda')

    model_path = args['model_path']
    crop_size = 32
    data_path = args['data_path']

    dataset_name = args['dataset'].lower()
    if dataset_name == 'qnrf':
        from datasets.crowd import Crowd_qnrf as Crowd
    elif dataset_name == 'nwpu':
        from datasets.crowd import Crowd_nwpu as Crowd
    elif dataset_name == 'sha':
        from datasets.crowd import Crowd_sh as Crowd
    elif dataset_name == 'shb':
        from datasets.crowd import Crowd_sh as Crowd
    else:
        raise NotImplementedError
    # TODO: solve deleted checkpoint file issue
    dataset = Crowd(os.path.join(args['data_path'], args["val_path"]),
                    crop_size=32,
                    downsample_ratio=8, method='val')
    dataloader = torch.utils.data.DataLoader(dataset, 1, shuffle=False,
                                             num_workers=1, pin_memory=True)
    time_str = datetime.strftime(datetime.now(), '%m%d-%H%M%S')
    log_dir = os.path.join('./runs', 'test_res', args['dataset'], time_str)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    logger = SummaryWriter(log_dir='runs')
    create_image = args['pred_density_map']

    model = vgg16dres(map_location=device)
    # model = vgg19()
    model.to(device)
    model.load_state_dict(torch.load(model_path, device))
    logger.add_graph(model, verbose=True)
    model.eval()
    image_errs = []
    logger.add_scalar('img_count', len(dataloader))

    for inputs, count, name in dataloader:
        inputs = inputs.to(device)
        assert inputs.size(0) == 1, 'the batch size should equal to 1'
        with torch.set_grad_enabled(False):
            outputs, _ = model(inputs)
        img_err = count[0].item() - torch.sum(outputs).item()

        image_errs.append(img_err)

        if create_image:
            # import cv2
            mse = np.sqrt(np.mean(np.square(image_errs[-1])))
            mae = np.mean(np.abs(image_errs[-1]))
            print(outputs.shape)
            vis_img = outputs[0]
            # normalize density map values from 0 to 1, then map it to 0-255.
            vis_img = (vis_img - vis_img.min()) / (vis_img.max() - vis_img.min() + 1e-5)
            vis_img = (vis_img * 255)
            # vis_img = cv2.applyColorMap(vis_img, cv2.COLORMAP_JET)
            print(vis_img.shape)
            logger.add_image(str(name[0]), vis_img)
            logger.add_image('img'+str(name[0]), inputs[0])
            logger.add_scalar('img_mae', mae)
            logger.add_scalar('img_mse', mse)
            # cv2.imwrite(os.path.join(args.pred_density_map_path, str(name[0]) + '.png'), vis_img)

    image_errs = np.array(image_errs)
    mse = np.sqrt(np.mean(np.square(image_errs)))
    mae = np.mean(np.abs(image_errs))
    logger.add_scalar('dataset_mae', mae)
    logger.add_scalar('dataset_mse', mse)
    
    print('{}: mae {}, mse {}\n'.format(model_path, mae, mse))
