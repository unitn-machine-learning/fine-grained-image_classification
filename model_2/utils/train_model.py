import os
import glob
import torch
from tqdm import tqdm
from config import max_checkpoint_num, proposalN, eval_trainset, set, eval_testset
from utils.eval_model import eval
import wandb


def train(model,
          trainloader,
          testloader,
          criterion,
          optimizer,
          scheduler,
          save_path,
          start_epoch,
          end_epoch,
          save_interval):
    # Initialize wandb
    wandb.init(project='model_2_competition_dataset')
    for epoch in range(start_epoch + 1, end_epoch + 1):
        model.train()

        print('Training %d epoch' % epoch)

        lr = next(iter(optimizer.param_groups))['lr']
        
        
        
        for i, data in enumerate(tqdm(trainloader)):
            if set == 'CUB':
                images, labels, _, _ = data
            else:
                images, labels = data
            images, labels = images.cuda(), labels.cuda()

            optimizer.zero_grad()

            proposalN_windows_score, proposalN_windows_logits, indices, \
            window_scores, _, raw_logits, local_logits, _ = model(images, epoch, i, 'train')

            raw_loss = criterion(raw_logits, labels)
            local_loss = criterion(local_logits, labels)
            windowscls_loss = criterion(proposalN_windows_logits,
                               labels.unsqueeze(1).repeat(1, proposalN).view(-1))

            if epoch < 2:
                total_loss = raw_loss
            else:
                total_loss = raw_loss + local_loss + windowscls_loss

            total_loss.backward()

            optimizer.step()

        scheduler.step()
        
        
        # evaluation every epoch
        if eval_trainset:
            raw_loss_avg, windowscls_loss_avg, total_loss_avg, raw_accuracy, local_accuracy, local_loss_avg\
                = eval(model, trainloader, criterion, 'train', save_path, epoch)

            print(
                'Train set: raw accuracy: {:.2f}%, local accuracy: {:.2f}%'.format(
                    100. * raw_accuracy, 100. * local_accuracy))
            # wandb logging
            wandb.log({
                'Train/raw_accuracy': raw_accuracy,
                'Train/local_accuracy': local_accuracy,
                'Train/raw_loss_avg': raw_loss_avg,
                'Train/local_loss_avg': local_loss_avg,
                'Train/windowscls_loss_avg': windowscls_loss_avg,
                'Train/total_loss_avg': total_loss_avg,
            }, step=epoch)

        if eval_testset:
            # eval testset
            raw_loss_avg, windowscls_loss_avg, total_loss_avg, raw_accuracy, local_accuracy, \
            local_loss_avg\
                = eval(model, testloader, criterion, 'test', save_path, epoch)

            print(
                'Test set: raw accuracy: {:.2f}%, local accuracy: {:.2f}%'.format(
                    100. * raw_accuracy, 100. * local_accuracy))
        
            # wandb logging
            wandb.log({
                'Test/raw_accuracy': raw_accuracy,
                'Test/local_accuracy': local_accuracy,
                'Test/raw_loss_avg': raw_loss_avg,
                'Test/local_loss_avg': local_loss_avg,
                'Test/windowscls_loss_avg': windowscls_loss_avg,
                'Test/total_loss_avg': total_loss_avg,
            }, step=epoch)
        
        
        # save checkpoint
        if (epoch % save_interval == 0) or (epoch == end_epoch):
            print('Saving checkpoint')
            checkpoint_path = os.path.join(save_path, 'epoch' + str(epoch) + '.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'learning_rate': lr,
            }, checkpoint_path)
            wandb.save(checkpoint_path)

        # Limit the number of checkpoints to less than or equal to max_checkpoint_num,
        # and delete the redundant ones
        checkpoint_list = [os.path.basename(path) for path in glob.glob(os.path.join(save_path, '*.pth'))]
        if len(checkpoint_list) == max_checkpoint_num + 1:
            idx_list = [int(name.replace('epoch', '').replace('.pth', '')) for name in checkpoint_list]
            min_idx = min(idx_list)
            os.remove(os.path.join(save_path, 'epoch' + str(min_idx) + '.pth'))

