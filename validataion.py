import torch
from torch import nn
from torch.utils.data import DataLoader

from DICE import DiceLoss
from upload.Evaluator import Evaluator
from upload.model import SwinTransformer
from upload.seg_dataset import mydata


def swin_tiny_patch4_window7_224(num_classes: int = 2):
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


loss_fn = nn.CrossEntropyLoss()
dice_loss = DiceLoss(2)


def fit(epoch, model, trainloader):
    test_correct = 0
    test_total = 0
    test_running_loss = 0

    model.eval()
    evaluator = Evaluator(2)
    with torch.no_grad():
        for x, y in trainloader:

            if torch.cuda.is_available():
                x, y = x.to('cuda'), y.to('cuda')
            y_pred = model(x)
            loss1 = loss_fn(y_pred, y)
            loss_dice = dice_loss(y_pred, y, softmax=True)
            loss = 0.4 * loss1 + 0.6 * loss_dice
            y_pred = torch.argmax(y_pred, dim=1)



            evaluator.add_batch(y, y_pred)


            test_correct += (y_pred == y).sum().item()
            test_total += y.size(0)
            test_running_loss += loss.item()
        mou = evaluator.Mean_Intersection_over_Union()
        mpa = evaluator.MPA()
        fwoiu = evaluator.Frequency_Weighted_Intersection_over_Union()
        dsc = evaluator.DSC()
        sen = evaluator.Sen()
        ppv = evaluator.PPV()
        acc = evaluator.Pixel_Accuracy()
    epoch_test_loss = test_running_loss / len(trainloader.dataset)
    epoch_test_acc = test_correct / (test_total * 224 * 224)
    print('epoch', epoch,
          'best_iou', mou,
          'val_miou: ', mou,
          'val_dsc： ', dsc,
          'val_mpa:', mpa,
          'val_fwoiu：', fwoiu,
          'val_sen', sen,
          'val_ppv', ppv,
          'val_acc:', acc)



model=swin_tiny_patch4_window7_224(num_classes=2).to('cuda')

PATH = './a/val_mpa_small_224.pth'
model.load_state_dict(torch.load(PATH))


val_data = mydata('D:\\study\\pytorch_study\\seg_thryoid_picture\\datasets', 224, 'val')
val_loader = DataLoader(val_data, batch_size=16)


epochs = 50
for epoch in range(epochs):
    fit(epoch, model,val_loader)
image, mask = next(iter(val_loader))

#
# plt.figure(figsize=(10, 10))
# for i in range(num):
#     plt.subplot(num, 3, i*num+1)
#     plt.imshow(image[i+3].permute(1,2,0).cpu().numpy())
#     plt.subplot(num, 3, i*num+2)
#     plt.imshow(mask[i+3].cpu().numpy())
#     plt.subplot(num, 3, i*num+3)
#     plt.imshow(torch.argmax(pred_mask[i+3].permute(1,2,0), axis=-1).detach().numpy())
#     plt.show()