import torch
import distributed_utils as utils
from torch import nn

from upload.DICE import DiceLoss
from upload.Evaluator import Evaluator


def create_lr_scheduler(optimizer,
                        num_step: int,
                        epochs: int,
                        warmup=True,
                        warmup_epochs=1,
                        warmup_factor=1e-3):
    assert num_step > 0 and epochs > 0
    if warmup is False:
        warmup_epochs = 0

    def f(x):
        """
        根据step数返回一个学习率倍率因子，
        注意在训练开始之前，pytorch会提前调用一次lr_scheduler.step()方法
        """
        if warmup is True and x <= (warmup_epochs * num_step):
            alpha = float(x) / (warmup_epochs * num_step)
            # warmup过程中lr倍率因子从warmup_factor -> 1
            return warmup_factor * (1 - alpha) + alpha
        else:
            # warmup后lr倍率因子从1 -> 0
            # 参考deeplab_v2: Learning rate policy
            return (1 - (x - warmup_epochs * num_step) / ((epochs - warmup_epochs) * num_step)) ** 0.9

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=f)
def train_one_epoch(model, optimizer, data_loader, device, epoch, lr_scheduler, print_freq=10, scaler=None):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    loss_fn = nn.CrossEntropyLoss()
    dice_loss = DiceLoss(2)
    for image, target in metric_logger.log_every(data_loader, print_freq, header):
        image, target = image.to(device), target.to(device)
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            output = model(image)

            loss1 = loss_fn(output, target)
            loss_dice = dice_loss(output, target, softmax=True)
            loss = 0.4 * loss1 + 0.6 * loss_dice

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        lr_scheduler.step()

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(loss=loss.item(), lr=lr)

    return metric_logger.meters["loss"].global_avg, lr
def evaluate(epoch,model, data_loader, device, num_classes):
    model.eval()
    best_moiu = 0
    best_mpa = 0
    confmat = utils.ConfusionMatrix(num_classes)
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'\

    with torch.no_grad():
        evaluator = Evaluator(2)
        for image, target in metric_logger.log_every(data_loader, 100, header):
            image, target = image.to(device), target.to(device)
            output = model(image)

            output= torch.argmax(output, dim=1)
            evaluator.add_batch(target, output)
            # confmat.update(target.flatten(), output.argmax(1).flatten())

        # confmat.reduce_from_all_processes()
            moiu = evaluator.Mean_Intersection_over_Union()
            mpa = evaluator.MPA()
            fwoiu = evaluator.Frequency_Weighted_Intersection_over_Union()
            dsc = evaluator.DSC()
            val_sen = evaluator.Sen()
            val_ppv = evaluator.PPV()
            val_acc = evaluator.Pixel_Accuracy()
            f1_socre=2*val_ppv*val_sen/(val_ppv+val_sen)
            if moiu>best_moiu:
                best_moiu=moiu
                torch.save(model.state_dict(), './moiu_small_224_{}.pth'.format(epoch))
            if mpa > best_mpa:
                best_mpa = mpa
                torch.save(model.state_dict(), './mpa_small_224_{}.pth'.format(epoch))
    return ('best_moiu: {:.1f}\n'
            'best_mpa: {}\n'
            'val_fwoiu: {}\n'
            'val_dsc: {:.1f}\n'
            'val_sen: {}\n'
            'val_ppv: {}\n'
            'val_acc: {}\n'
            'val_F1:{}\n' ).format(
                moiu.item() * 100,
                mpa*100,
                fwoiu*100,
                dsc * 100,
                val_sen * 100,
                val_ppv * 100,
                val_acc*100,
                f1_socre*100)