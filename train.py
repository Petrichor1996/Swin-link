import datetime
import os
import time

import torch
from torch import nn
from torch.utils.data import DataLoader

from DICE import DiceLoss

from upload.Evaluator import Evaluator
from upload.model import SwinTransformer
from upload.seg_dataset import mydata
from upload.train_and_eval import create_lr_scheduler, train_one_epoch, evaluate


def swin_tiny_patch4_window7_224(num_classes: int = 1000):
    # trained ImageNet-1K
    # https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth
    model = SwinTransformer(in_chans=3,
                            patch_size=4,
                            window_size=7,
                            embed_dim=96,
                            depths=(2, 2, 6, 2),
                            num_heads=(3, 6, 12, 24),
                            num_classes=num_classes,
                            )
    return model
def swin_tiny_patch4_window12_384(num_classes: int = 1000):
    # trained ImageNet-1K
    # https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth
    model = SwinTransformer(in_chans=3,
                            patch_size=4,
                            window_size=12,
                            embed_dim=96,
                            depths=(2, 2, 6, 2),
                            num_heads=(3, 6, 12, 24),
                            num_classes=num_classes,
                            )
    return model
def swin_small_patch4_window7_224(num_classes: int = 1000):
    # trained ImageNet-1K
    # https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_small_patch4_window7_224.pth
    model = SwinTransformer(in_chans=3,
                            patch_size=4,
                            window_size=7,
                            embed_dim=96,
                            depths=(2, 2, 18, 2),
                            num_heads=(3, 6, 12, 24),
                            num_classes=num_classes,
                           )
    return model
def swin_small_patch4_window12_384(num_classes: int = 1000):
    # trained ImageNet-1K
    # https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_small_patch4_window7_224.pth
    model = SwinTransformer(in_chans=3,
                            patch_size=4,
                            window_size=12,
                            embed_dim=96,
                            depths=(2, 2, 18, 2),
                            num_heads=(3, 6, 12, 24),
                            num_classes=num_classes,
                           )
    return model
def swin_base_patch4_window7_224(num_classes: int = 1000):
    # trained ImageNet-1K
    # https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window7_224.pth
    model = SwinTransformer(in_chans=3,
                            patch_size=4,
                            window_size=7,
                            embed_dim=128,
                            depths=(2, 2, 18, 2),
                            num_heads=(4, 8, 16, 32),
                            num_classes=num_classes)
    return model
def swin_base_patch4_window12_384(num_classes: int = 1000):
    # trained ImageNet-1K
    # https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window12_384.pth
    model = SwinTransformer(in_chans=3,
                            patch_size=4,
                            window_size=12,
                            embed_dim=128,
                            depths=(2, 2, 18, 2),
                            num_heads=(4, 8, 16, 32),
                            num_classes=num_classes)
    return model
def swin_large_patch4_window12_384(num_classes: int = 21841):
    # trained ImageNet-22K
    # https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window12_384_22k.pth
    model = SwinTransformer(in_chans=3,
                            patch_size=4,
                            window_size=12,
                            embed_dim=192,
                            depths=(2, 2, 18, 2),
                            num_heads=(6, 12, 24, 48),
                            num_classes=num_classes)
    return model
def swin_large_patch4_window7_224(num_classes: int = 21841):
    # trained ImageNet-22K
    # https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window7_224_22k.pth
    model = SwinTransformer(in_chans=3,
                            patch_size=4,
                            window_size=7,
                            embed_dim=192,
                            depths=(2, 2, 18, 2),
                            num_heads=(6, 12, 24, 48),
                            num_classes=num_classes)
    return model

def create_model(num_classes, pretrain=False):
    model = swin_small_patch4_window7_224(num_classes=2).to('cuda')

    return model



def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    batch_size = args.batch_size
    # segmentation nun_classes + background
    num_classes = args.num_classes + 1

    # 用来保存训练以及验证过程中信息
    results_file = "results8s_small_480_e4_32s{}.txt".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

    iss =224
    train_data = mydata('D:\\study\\pytorch_study\\seg_thryoid_picture\\datasets', iss, 'train')
    test_data = mydata('D:\\study\\pytorch_study\\seg_thryoid_picture\\datasets', iss, 'test')
    val_data = mydata('D:\\study\\pytorch_study\\seg_thryoid_picture\\datasets', iss, 'val')

    train_loader = DataLoader(train_data, batch_size=batch_size , shuffle=True,drop_last=True)
    test_loader = DataLoader(test_data, batch_size=batch_size,shuffle=False,drop_last=True)

    model = create_model( num_classes=num_classes)
    # flops, params = get_model_complexity_info(model, (3, 224, 224), as_strings=True, print_per_layer_stat=True)
    # print('FLOPs:{}'.format(flops))
    # print('Params:{}'.format(params))
    model.to(device)

    # params_to_optimize = [
    #     {"params": [p for p in model.backbone.parameters() if p.requires_grad]},
    #     {"params": [p for p in model.classifier.parameters() if p.requires_grad]}
    # ]

    # if args.aux:
    #     params = [p for p in model.aux_classifier.parameters() if p.requires_grad]
    #     params_to_optimize.append({"params": params, "lr": args.lr * 10})

    # optimizer = torch.optim.SGD(
    #     model.parameters(),
    #     lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay
    # )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=5e-2)
    scaler = torch.cuda.amp.GradScaler() if args.amp else None

    # 创建学习率更新策略，这里是每个step更新一次(不是每个epoch)
    lr_scheduler = create_lr_scheduler(optimizer, len(train_loader), args.epochs, warmup=True)

    # import matplotlib.pyplot as plt
    # lr_list = []
    # for _ in range(args.epochs):
    #     for _ in range(len(train_loader)):
    #         lr_scheduler.step()
    #         lr = optimizer.param_groups[0]["lr"]
    #         lr_list.append(lr)
    # plt.plot(range(len(lr_list)), lr_list)
    # plt.show()

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        args.start_epoch = checkpoint['epoch'] + 1
        if args.amp:
            scaler.load_state_dict(checkpoint["scaler"])

    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        mean_loss, lr = train_one_epoch(model, optimizer, train_loader, device, epoch,
                                        lr_scheduler=lr_scheduler, print_freq=args.print_freq, scaler=scaler)

        confmat = evaluate(epoch,model, test_loader, device=device, num_classes=num_classes)
        val_info = str(confmat)
        print(val_info)
        # write into txt
        with open(results_file, "a") as f:
            # 记录每个epoch对应的train_loss、lr以及验证集各指标
            train_info = f"[epoch: {epoch}]\n" \
                         f"train_loss: {mean_loss:.4f}\n" \
                         f"lr: {lr:.6f}\n"
            f.write(train_info + val_info + "\n\n")

        save_file = {"model": model.state_dict(),
                     "optimizer": optimizer.state_dict(),
                     "lr_scheduler": lr_scheduler.state_dict(),
                     "epoch": epoch,
                     "args": args}
        if args.amp:
            save_file["scaler"] = scaler.state_dict()
        torch.save(save_file, "save_weights/small480_32s_e4_model_{}.pth".format(epoch))

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("training time {}".format(total_time_str))


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="pytorch deeplabv3 training")

    parser.add_argument("--data-path", default="/data/", help="VOCdevkit root")
    parser.add_argument("--num-classes", default=1, type=int)
    parser.add_argument("--aux", default=False, type=bool, help="auxilier loss")#me
    parser.add_argument("--device", default="cuda", help="training device")
    parser.add_argument("-b", "--batch-size", default=6, type=int)#me
    parser.add_argument("--epochs", default=20, type=int, metavar="N",
                        help="number of total epochs to train")#me

    parser.add_argument('--lr', default=0.0001, type=float, help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')#me
    parser.add_argument('--print-freq', default=10, type=int, help='print frequency')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    # Mixed precision training parameters
    parser.add_argument("--amp", default=False, type=bool,
                        help="Use torch.cuda.amp for mixed precision training")

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()

    if not os.path.exists("./save_weights"):
        os.mkdir("./save_weights")

    main(args)

# model=swin_tiny_patch4_window7_224(num_classes=2).to('cuda')
#
# # model=swin_tiny_patch4_window12_384(num_classes=2).to('cuda')
# # model=swin_small_patch4_window7_224(num_classes=2).to('cuda')
# # model=swin_small_patch4_window12_384(num_classes=2).to('cuda')
# # model=swin_base_patch4_window7_224(num_classes=2).to('cuda')
# # model=swin_base_patch4_window12_384(num_classes=2).to('cuda')
# # model=swin_large_patch4_window7_224(num_classes=2).to('cuda')
# # model=swin_large_patch4_window12_384(num_classes=2).to('cuda')
#
# # model.load_state_dict(torch.load('D:\\study\pytorch_study\\seg_thryoid_picture\\swin_link\\small_224_model.pth'))
#
# loss_fn=nn.CrossEntropyLoss()
# dice_loss = DiceLoss(2)
#
# optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=5e-2)
#
#
#
# best_acc=0
# best_mou=0
# best_dsc=0
# best_mpa=0
# best_fwiou=0
# best_loss=90
# best_sen=0
# best_ppv=0
# i=0
#
# def fit(epoch,model,trainloader,testloader):
#     img_size=224
#     correct=0
#     total=0
#     running_loss=0
#
#     model.train()
#     evaluator = Evaluator(2)
#     for x,y in trainloader:
#         if torch.cuda.is_available():
#             x,y=x.to('cuda'),y.to('cuda')
#             y_pred= model(x)
#
#         loss1=loss_fn(y_pred,y)
#         loss_dice = dice_loss(y_pred, y, softmax=True)
#         loss = 0.4 * loss1 + 0.6 * loss_dice
#
#
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#
#         with torch.no_grad():
#             y_pred=torch.argmax(y_pred,dim=1)
#             total+=y.size(0)
#             running_loss+=loss.item()
#         global i
#         if i % 3000 == 0:
#             print('loss.item()', loss.item(), i * y.size(0))
#         # viz.line([moiu], [i * y.size(0)], win='miou', update='append')
#         # viz.line([dsc], [i * y.size(0)], win='dsc', update='append')
#         # viz.line([mpa], [i * y.size(0)], win='mpa', update='append')
#         # viz.line([fwoiu], [i * y.size(0)], win='fwoiu', update='append')
#         # viz.line([loss.item()],[i*y.size(0)], win='loss', update='append')
#         i+=1
#         correct += (y_pred == y).sum().item()
#
#     # exp_lr_scheduler.step()
#
#     epoch_loss=running_loss/len(trainloader.dataset)
#     epoch_acc=correct/(total*img_size*img_size)
#
#
#     #-----------------------------------test
#     test_correct = 0
#     test_total = 0
#     test_running_loss = 0
#
#     model.eval()
#     evaluator2 = Evaluator(2)
#     with torch.no_grad():
#
#         for x, y in testloader:
#             if torch.cuda.is_available():
#                 x, y = x.to('cuda'), y.to('cuda')
#             y_pred = model(x)
#
#             loss1 = loss_fn(y_pred, y)
#             loss_dice = dice_loss(y_pred, y, softmax=True)
#             loss = 0.4 * loss1 + 0.6 * loss_dice
#
#             y_pred = torch.argmax(y_pred, dim=1)
#
#
#             evaluator2.add_batch(y, y_pred)
#
#             val_mou = evaluator2.Mean_Intersection_over_Union()
#             val_mpa = evaluator2.MPA()
#             val_fwoiu = evaluator2.Frequency_Weighted_Intersection_over_Union()
#             val_dsc = evaluator2.DSC()
#             val_sen=evaluator2.Sen()
#             val_ppv=evaluator2.PPV()
#             val_acc=evaluator2.Pixel_Accuracy()
#
#
#             test_correct += (y_pred == y).sum().item()
#             test_total += y.size(0)
#             test_running_loss += loss.item()
#             global best_mpa,best_mou
#             if val_mpa > best_mpa:
#                 best_mpa = val_mpa
#                 torch.save(model.state_dict(), './a/val_mpa_small_224.pth')
#             if val_mou > best_mou:
#                 best_mou = val_mou
#                 torch.save(model.state_dict(), './a/val_mou_small_224.pth')
#     global best_acc,best_dsc,best_fwiou,best_sen,best_ppv,best_loss
#
#
#     epoch_test_loss = test_running_loss / len(testloader.dataset)
#     epoch_test_acc = test_correct / (test_total *img_size *img_size)
#
#
#     if epoch_test_loss < best_loss:
#         best_loss = epoch_test_loss
#         torch.save(model.state_dict(), './a/val_loss_small_224.pth')
#     if epoch_test_acc > best_acc:
#         best_acc = epoch_test_acc
#         torch.save(model.state_dict(), './a/val_acc_small_224.pth')
#
#     if val_dsc > best_dsc:
#         best_dsc = val_dsc
#         torch.save(model.state_dict(), './a/val_dsc_small_224.pth')
#
#     if val_fwoiu > best_fwiou:
#         best_fwiou = val_fwoiu
#         torch.save(model.state_dict(), './a/val_fwoiu_small_224.pth')
#     if val_sen > best_sen:
#         best_sen = val_sen
#         torch.save(model.state_dict(), './a/val_sen_small_224.pth')
#     if val_ppv > best_ppv:
#         best_ppv = val_ppv
#         torch.save(model.state_dict(), './a/val_ppv_small_224.pth')
#     # viz.line([val_miou], [epoch], win='val_miou', update='append')
#     # viz.line([val_dsc], [epoch], win='val_dsc', update='append')
#     # viz.line([val_mpa], [epoch], win='val_mpa', update='append')
#     # viz.line([val_fwoiu], [epoch], win='val_fwoiu', update='append')
#     # viz.line([epoch_test_loss], [epoch], win='val_loss', update='append')
#     # viz.line([epoch_test_acc], [epoch], win='val_acc', update='append')
#
#     print('epoch', epoch,
#           'best_iou',best_mou,
#           'val_miou: ', val_mou,
#           'val_dsc： ', val_dsc,
#           'val_mpa:', val_mpa,
#           'val_fwoiu：', val_fwoiu,
#           'val_sen',val_sen,
#           'val_ppv',val_ppv,
#           'val_acc:',val_acc)
#
#     return epoch_loss, epoch_acc, epoch_test_loss, epoch_test_acc
#
# epochs = 50
#
# # %%
#
# train_loss = []
# train_acc = []
# test_loss = []
# test_acc = []
#
# def main():
#     train_data = mydata('D:\\study\\pytorch_study\\seg_thryoid_picture\\datasets',224,'train')
#     test_data = mydata('D:\\study\\pytorch_study\\seg_thryoid_picture\\datasets',224, 'test')
#     val_data = mydata('D:\\study\\pytorch_study\\seg_thryoid_picture\\datasets', 224, 'val')
#
#
#
#     train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
#     test_loader = DataLoader(test_data, batch_size=16)
#     val_loader = DataLoader(val_data, batch_size=16)
#
#     for epoch in range(epochs):
#         epoch_loss, epoch_acc, epoch_test_loss, epoch_test_acc = fit(epoch,
#                                                                      model,
#                                                                      train_loader,
#                                                                      test_loader)
#         train_loss.append(epoch_loss)
#         train_acc.append(epoch_acc)
#         test_loss.append(epoch_test_loss)
#         test_acc.append(epoch_test_acc)
#
# if __name__ == '__main__':
#     main()