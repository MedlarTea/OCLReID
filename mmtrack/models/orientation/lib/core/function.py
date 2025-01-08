from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
 
import time
import logging
import os
from turtle import pd
import matplotlib.pyplot as plt
from PIL import Image
import gc
import numpy as np
import torch
import pickle
import cv2

from lib.core.evaluate import accuracy
from lib.core.evaluate import comp_deg_error, continous_comp_deg_error, ori_numpy


logger = logging.getLogger(__name__)

def save_obj(obj, name ):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

def print_msg(step, loader_len, batch_time, has_hkd, loss_hoe, losses, degree_error, acc_label, acc, speed=False, epoch = None, loss_hkd=None, loss_mask=None):
  
  if epoch != None:
    msg = 'Epoch: [{0}][{1}/{2}]\t' \
          'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
          'Speed {speed:.1f} samples/s\t'.format(epoch,step, loader_len, batch_time=batch_time, speed = speed)
  else:
    msg = 'Test: [{0}/{1}]\t' \
        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'.format(
            step, loader_len, batch_time=batch_time)
  if has_hkd:
    # 'Loss_vis {loss_vis.val:.3e} ({loss_vis.avg:.3e})\t' \

    msg += 'Loss_hkd {loss_hkd.val:.3e} ({loss_hkd.avg:.3e})\t' \
        'Loss_hoe {loss_hoe.val:.3e} ({loss_hoe.avg:.3e})\t' \
        'Loss {loss.val:.3e} ({loss.avg:.3e})\t'.format(loss_hkd=loss_hkd, loss_hoe=loss_hoe,loss_mask=loss_mask, loss=losses)
  else:
    msg += 'Loss {loss.val:.3e} ({loss.avg:.3e})\t'.format(loss=losses)
  
  msg += 'Degree_error {Degree_error.val:.3f} ({Degree_error.avg:.3f})\t' \
        '{acc_label} {acc.val:.1%} ({acc.avg:.1%})'.format(Degree_error = degree_error, acc_label=acc_label, acc=acc)
  logger.info(msg)

def my_train(config, train_loader, train_dataset, model, criterions, optimizer, epoch,
          output_dir, tb_log_dir, writer_dict):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_2d_log = AverageMeter()
    loss_hoe_log = AverageMeter()
    loss_mask_log = AverageMeter()
    losses = AverageMeter()
    degree_error = AverageMeter()
    acc = AverageMeter()

    # switch to train mode
    model.train()
    end = time.time()

    for i, (input, target, target_weight, degree, meta, mask_gt) in enumerate(train_loader):
        # print(mask_gt)
        data_time.update(time.time() - end)

        # compute output
        plane_output, hoe_output, mask = model(input)

        # change to cuda format
        target = target.cuda(non_blocking=True)
        target_weight = target_weight.cuda(non_blocking=True)
        degree = degree.cuda(non_blocking=True)
        mask_gt = mask_gt.cuda(non_blocking=True)
        # import pdb;pdb.set_trace()

        # compute loss
        if config.LOSS.USE_ONLY_HOE:
            loss_hoe = criterions['hoe_loss'](hoe_output, degree)
            loss_2d = loss_hoe
            loss = loss_hoe
        else:
            loss_2d = criterions['2d_pose_loss'](plane_output, target, target_weight)
            loss_hoe = criterions['hoe_loss'](hoe_output , degree)
            # import pdb;pdb.set_trace()
            loss_mask = criterions['mask_loss'](mask, mask_gt)
            loss = loss_2d + 0.1*loss_hoe + 0.001*loss_mask

        # compute gradient and do update step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure accuracy and record loss
        loss_2d_log.update(loss_2d.item(), input.size(0))
        loss_hoe_log.update(loss_hoe.item(), input.size(0))
        if type(loss_mask) == dict:
            loss_mask_log.update(loss_mask.item(),input.size(0))
        losses.update(loss.item(), input.size(0))

        if config.DATASET.DATASET == 'tud_dataset':
            avg_degree_error, _, mid, _ , _, _, _, _, cnt = continous_comp_deg_error(hoe_output.detach().cpu().numpy(),
                                               meta['val_dgree'].numpy())
            
            acc.update(mid/cnt, cnt)
            has_hkd=False
            acc_label = 'mid15'
        elif config.LOSS.USE_ONLY_HOE:
            avg_degree_error, _, mid, _ , _, _, _, _, cnt= comp_deg_error(hoe_output.detach().cpu().numpy(),
                                                   degree.detach().cpu().numpy())
            acc.update(mid/cnt, cnt)
            has_hkd=False 
            acc_label = 'mid15'
        else:
            avg_degree_error, _, _, _ , _, _, _, _, _= comp_deg_error(hoe_output.detach().cpu().numpy(),
                                                   degree.detach().cpu().numpy())
            _, avg_acc, cnt, pred = accuracy(plane_output.detach().cpu().numpy(),
                                             target.detach().cpu().numpy())
            acc.update(avg_acc, cnt)
            has_hkd=True
            acc_label = 'kpd_acc'
            

        degree_error.update(avg_degree_error, input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % config.PRINT_FREQ == 0:
            print_msg(epoch = epoch, step=i, speed=input.size(0) / batch_time.val, has_hkd= has_hkd, loader_len=len(train_loader), batch_time=batch_time, loss_hkd=loss_2d_log, loss_hoe=loss_hoe_log,loss_mask=loss_mask_log, losses=losses, degree_error=degree_error, acc_label=acc_label, acc=acc)

            writer = writer_dict['writer']
            global_steps = writer_dict['train_global_steps']
            writer.add_scalar('train_loss', losses.val, global_steps)
            writer.add_scalar('train_hkd_loss', loss_2d_log.val, global_steps)
            writer.add_scalar('train_hoe_loss', loss_hoe_log.val, global_steps)
            writer.add_scalar('train_mask_loss', loss_mask_log.val, global_steps)
            writer.add_scalar('train_acc', acc.val, global_steps)
            writer.add_scalar('degree_error', degree_error.val, global_steps)
            writer_dict['train_global_steps'] = global_steps + 1

def train(config, train_loader, train_dataset, model, criterions, optimizer, epoch,
          output_dir, tb_log_dir, writer_dict):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_2d_log = AverageMeter()
    loss_hoe_log = AverageMeter()
    losses = AverageMeter()
    degree_error = AverageMeter()
    acc = AverageMeter()

    # switch to train mode
    model.train()
    end = time.time()

    for i, (input, target, target_weight, degree, meta) in enumerate(train_loader):
        data_time.update(time.time() - end)

        # compute output
        plane_output, hoe_output = model(input)

        # change to cuda format
        target = target.cuda(non_blocking=True)
        target_weight = target_weight.cuda(non_blocking=True)
        degree = degree.cuda(non_blocking=True)

        # compute loss
        if config.LOSS.USE_ONLY_HOE:
            loss_hoe = criterions['hoe_loss'](hoe_output, degree)
            loss_2d = loss_hoe
            loss = loss_hoe
        else:
            loss_2d = criterions['2d_pose_loss'](plane_output, target, target_weight)
            loss_hoe = criterions['hoe_loss'](hoe_output , degree)

            loss = loss_2d + 0.1*loss_hoe

        # compute gradient and do update step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure accuracy and record loss
        loss_2d_log.update(loss_2d.item(), input.size(0))
        loss_hoe_log.update(loss_hoe.item(), input.size(0))
        losses.update(loss.item(), input.size(0))

        if config.DATASET.DATASET == 'tud_dataset':
            avg_degree_error, _, mid, _ , _, _, _, _, cnt = continous_comp_deg_error(hoe_output.detach().cpu().numpy(),
                                               meta['val_dgree'].numpy())
            
            acc.update(mid/cnt, cnt)
            has_hkd=False
            acc_label = 'mid15'
        elif config.LOSS.USE_ONLY_HOE:
            avg_degree_error, _, mid, _ , _, _, _, _, cnt= comp_deg_error(hoe_output.detach().cpu().numpy(),
                                                   degree.detach().cpu().numpy())
            acc.update(mid/cnt, cnt)
            has_hkd=False 
            acc_label = 'mid15'
        else:
            avg_degree_error, _, _, _ , _, _, _, _, _= comp_deg_error(hoe_output.detach().cpu().numpy(),
                                                   degree.detach().cpu().numpy())
            _, avg_acc, cnt, pred = accuracy(plane_output.detach().cpu().numpy(),
                                             target.detach().cpu().numpy())
            acc.update(avg_acc, cnt)
            has_hkd=True
            acc_label = 'kpd_acc'
            

        degree_error.update(avg_degree_error, input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % config.PRINT_FREQ == 0:
            print_msg(epoch = epoch, step=i, speed=input.size(0) / batch_time.val, has_hkd= has_hkd, loader_len=len(train_loader), batch_time=batch_time, loss_hkd=loss_2d_log, loss_hoe=loss_hoe_log, losses=losses, degree_error=degree_error, acc_label=acc_label, acc=acc)

            writer = writer_dict['writer']
            global_steps = writer_dict['train_global_steps']
            writer.add_scalar('train_loss', losses.val, global_steps)
            writer.add_scalar('train_hkd_loss', loss_2d_log.val, global_steps)
            writer.add_scalar('train_hoe_loss', loss_hoe_log.val, global_steps)
            writer.add_scalar('train_acc', acc.val, global_steps)
            writer.add_scalar('degree_error', degree_error.val, global_steps)
            writer_dict['train_global_steps'] = global_steps + 1

def draw_orientation(img_np, gt_ori, pred_ori, path, keypoints, gt_keypoints, predict_mask, alis=''):
    if not os.path.exists(path):
        os.makedirs(path)
    for idx in range(len(pred_ori)):
        img_tmp = img_np[idx]

        img_tmp = np.transpose(img_tmp, axes=[1, 2, 0])
        img_tmp *= [0.229, 0.224, 0.225]
        img_tmp += [0.485, 0.456, 0.406]
        img_tmp *= 255.0
        img_tmp = img_tmp.astype(np.uint8)
        for joint_index, position in keypoints[idx].items():
            # left red , right green
            # import pdb;pdb.set_trace()
            img_tmp = img_tmp.copy()
            # print("joint_index{} x{} y{}".format(joint_index, position[0]*4, position[1]*4))
            if int(joint_index) %2 ==0:
                #right
                cv2.circle(img_tmp, (int(position[1]*4), int(position[0]*4)), radius=2, color=(0,0,255), thickness=-1, lineType=cv2.LINE_AA)
            else:
                #left
                cv2.circle(img_tmp, (int(position[1]*4), int(position[0]*4)), radius=2, color=(0,255,0), thickness=-1, lineType=cv2.LINE_AA)

        for joint_index, position in gt_keypoints[idx].items():
            # left red , right green
            # import pdb;pdb.set_trace()
            img_tmp = img_tmp.copy()
            # print("joint_index{} x{} y{}".format(joint_index, position[0]*4, position[1]*4))
            if int(joint_index) %2 ==0:
                #right
                cv2.circle(img_tmp, (int(position[1]*4), int(position[0]*4)), radius=2, color=(255,0,0), thickness=-1, lineType=cv2.LINE_AA)
            else:
                #left
                cv2.circle(img_tmp, (int(position[1]*4), int(position[0]*4)), radius=2, color=(255,255,255), thickness=-1, lineType=cv2.LINE_AA)

        # then draw the image
        fig = plt.figure(figsize=(2, 2))
        ax = fig.add_subplot(1, 1, 1)

        theta_1 = gt_ori[idx]/180 * np.pi + np.pi/2
        plt.plot([0, np.cos(theta_1)], [0, np.sin(theta_1)], color="red", linewidth=3)
        theta_2 = pred_ori[idx]/180 * np.pi + np.pi/2
        plt.plot([0, np.cos(theta_2)], [0, np.sin(theta_2)], color="blue", linewidth=3)
        circle = plt.Circle((0, 0), 1, color='black', fill=False, linewidth=2)
        ax.add_patch(circle)
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)
        ax.axis('off')
        fig.savefig(os.path.join(path, str(idx)+'_'+alis+'.jpg'))
        ori_img = cv2.imread(os.path.join(path, str(idx)+'_'+alis+'.jpg'))

        # 右半
        project_img =  np.array(Image.fromarray(img_tmp).resize((48,64)))
        tmp_mask = np.array([predict_mask[idx][0], predict_mask[idx][0], predict_mask[idx][0]])
        tmp_mask = tmp_mask.transpose(1,2,0)
        project_img[tmp_mask > 1] = 255
        project_img = Image.fromarray(project_img)

        project_img.save(os.path.join(path, str(idx)+'_'+alis+'_raw_mask0.jpg'))

        # 左半
        project_img =  np.array(Image.fromarray(img_tmp).resize((48,64)))
        tmp_mask = np.array([predict_mask[idx][1], predict_mask[idx][1], predict_mask[idx][1]])
        tmp_mask = tmp_mask.transpose(1,2,0)
        project_img[tmp_mask > 1] = 255
        project_img = Image.fromarray(project_img)

        project_img.save(os.path.join(path, str(idx)+'_'+alis+'_raw_mask1.jpg'))

        # 上半
        project_img =  np.array(Image.fromarray(img_tmp).resize((48,64)))
        tmp_mask = np.array([predict_mask[idx][2], predict_mask[idx][2], predict_mask[idx][2]])
        tmp_mask = tmp_mask.transpose(1,2,0)
        project_img[tmp_mask > 1] = 255
        project_img = Image.fromarray(project_img)

        project_img.save(os.path.join(path, str(idx)+'_'+alis+'_raw_mask2.jpg'))

        # 下半
        project_img =  np.array(Image.fromarray(img_tmp).resize((48,64)))
        tmp_mask = np.array([predict_mask[idx][3], predict_mask[idx][3], predict_mask[idx][3]])
        tmp_mask = tmp_mask.transpose(1,2,0)
        project_img[tmp_mask > 1] = 255
        project_img = Image.fromarray(project_img)

        project_img.save(os.path.join(path, str(idx)+'_'+alis+'_raw_mask3.jpg'))

        # 全身
        project_img =  np.array(Image.fromarray(img_tmp).resize((48,64)))
        tmp_mask = np.array([predict_mask[idx][4], predict_mask[idx][4], predict_mask[idx][4]])
        tmp_mask = tmp_mask.transpose(1,2,0)
        project_img[tmp_mask > 1] = 255
        project_img = Image.fromarray(project_img)

        project_img.save(os.path.join(path, str(idx)+'_'+alis+'_raw_mask4.jpg'))
        
        width = img_tmp.shape[1]
        ori_img = cv2.resize(ori_img, (width, width), interpolation=cv2.INTER_CUBIC)
        img_all = np.concatenate([img_tmp, ori_img],axis=0)
        im = Image.fromarray(img_all)
        im.save(os.path.join(path, str(idx)+'_'+alis+'_raw.jpg'))
        plt.close("all")
        del ori_img,img_all, im,img_tmp
        gc.collect()

# this is validate part
def my_validate(config, val_loader, val_dataset, model, criterions,  output_dir,
             tb_log_dir, writer_dict=None, draw_pic=False, save_pickle=False):
    batch_time = AverageMeter()
    loss_hkd_log = AverageMeter()
    loss_hoe_log = AverageMeter()
    loss_mask_log = AverageMeter()
    losses = AverageMeter()
    degree_error = AverageMeter()
    acc = AverageMeter()
    Excellent = 0
    Mid_good = 0
    Poor_good = 0
    Poor_225 = 0
    Poor_45 = 0
    Total = 0

    # switch to evaluate mode
    model.eval()

    num_samples = len(val_dataset)
    idx = 0
    ori_list = []
    with torch.no_grad():
        end = time.time()
        for i, (input, target, target_weight, degree, meta, mask_gt) in enumerate(val_loader):
            # compute output
            plane_output, hoe_output, mask = model(input)

            predict_mask = mask_gt.detach().cpu().numpy()
            # change to cuda format
            target = target.cuda(non_blocking=True)
            target_weight = target_weight.cuda(non_blocking=True)
            degree = degree.cuda(non_blocking=True)
            mask_gt = mask_gt.cuda(non_blocking=True)
            print(mask_gt.detach().cpu().numpy().max())
            print(mask.detach().cpu().numpy().max())

            # compute loss
            if config.LOSS.USE_ONLY_HOE:
                loss_hoe = criterions['hoe_loss'](hoe_output, degree)
                loss_2d = loss_hoe
                loss = loss_hoe
            else:
                loss_2d = criterions['2d_pose_loss'](plane_output, target, target_weight)
                loss_hoe = criterions['hoe_loss'](hoe_output, degree)
                loss_mask = criterions['mask_loss'](mask , mask_gt)
                loss = loss_2d + loss_hoe + loss_mask

            num_images = input.size(0)
            # measure accuracy and record loss
            loss_hkd_log.update(loss_2d.item(), num_images)
            loss_hoe_log.update(loss_hoe.item(), num_images)
            if type(loss_mask) == dict:
                loss_mask_log.update(loss_mask.item(),num_images)

            losses.update(loss.item(), num_images)

            if 'tud' in config.DATASET.VAL_ROOT:
                avg_degree_error, excellent, mid, poor_225, poor, poor_45, gt_ori, pred_ori, cnt  = continous_comp_deg_error(hoe_output.detach().cpu().numpy(),
                                                                                   meta['val_dgree'].numpy())
                acc.update(mid/cnt, cnt)
                acc_label = 'mid15'
                has_hkd = False
            elif config.LOSS.USE_ONLY_HOE:
                avg_degree_error, excellent, mid, poor_225, poor, poor_45, gt_ori, pred_ori, cnt  = comp_deg_error(hoe_output.detach().cpu().numpy(),
                                                                                   degree.detach().cpu().numpy())
                acc.update(mid/cnt, cnt)
                acc_label = 'mid15'
                has_hkd = False
            else:
                avg_degree_error, excellent, mid, poor_225, poor, poor_45,gt_ori, pred_ori, _  = comp_deg_error(hoe_output.detach().cpu().numpy(),
                                                           degree.detach().cpu().numpy())
                _, avg_acc, cnt, pred = accuracy(plane_output.cpu().numpy(),
                                                 target.cpu().numpy())

                acc.update(avg_acc, cnt)
                acc_label = 'kpd_acc'
                has_hkd = True
            
            if draw_pic:
                ori_path = os.path.join(output_dir, 'orientation_img')
                batch_keypoints = []
                gt_batch_keypoints = []

                for index in range(len(plane_output)):
                    keypoints = dict()
                    gt_keypoints = dict()
                    for j in range(17):
                        # 关键点输出值大于0.4 认为其存在
                        if plane_output[index][j].cpu().numpy().max() > 0.4:
                            position = np.unravel_index(plane_output[index][j].cpu().numpy().argmax(), plane_output[index][j].cpu().numpy().shape)
                            gt_position = np.unravel_index(target[index][j].cpu().numpy().argmax(), plane_output[index][j].cpu().numpy().shape)
                            keypoint = {"{}".format(j):position}
                            gt_keypoint = {"{}".format(j):gt_position}
                            keypoints.update(keypoint)
                            gt_keypoints.update(gt_keypoint)
                    
                    batch_keypoints.append(keypoints)
                    gt_batch_keypoints.append(gt_keypoints)
                if not os.path.exists(ori_path):
                    os.makedirs(ori_path)
                img_np = input.numpy()
                draw_orientation(img_np, degree.detach().cpu().numpy().argmax(axis = 1)*5, hoe_output.detach().cpu().numpy().argmax(axis = 1)*5, ori_path, batch_keypoints, gt_batch_keypoints, predict_mask, alis=str(i))

            if save_pickle:
                tamp_list = ori_numpy(gt_ori, pred_ori)
                ori_list = ori_list + tamp_list

            degree_error.update(avg_degree_error, num_images)

            Total += num_images
            Excellent += excellent
            Mid_good += mid
            Poor_good += poor
            Poor_45 += poor_45
            Poor_225 += poor_225

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            idx += num_images

            if i % config.PRINT_FREQ == 0:
                print_msg(step=i, loader_len=len(val_loader), batch_time=batch_time, has_hkd= has_hkd, loss_hkd=loss_hkd_log, loss_hoe=loss_hoe_log,loss_mask=loss_mask_log, losses=losses, degree_error=degree_error, acc_label=acc_label, acc=acc)

        if save_pickle:
            save_obj(ori_list, 'ori_list')
        excel_rate = Excellent / Total
        mid_rate = Mid_good / Total
        poor_rate = Poor_good / Total
        poor_225_rate = Poor_225 / Total
        poor_45_rate = Poor_45 / Total
        name_values = {'Degree_error': degree_error.avg, '5_Excel_rate': excel_rate, '15_Mid_rate': mid_rate, '225_rate': poor_225_rate, '30_Poor_rate': poor_rate, '45_poor_rate': poor_45_rate}
        _print_name_value(name_values, config.MODEL.NAME)

        if writer_dict:
            writer = writer_dict['writer']
            global_steps = writer_dict['valid_global_steps']
            writer.add_scalar(
                'valid_loss',
                losses.avg,
                global_steps
            )
            writer.add_scalar(
                'valid_hkd_loss',
                loss_hkd_log.avg,
                global_steps
            )
            writer.add_scalar(
                'valid_hoe_loss',
                loss_hoe_log.avg,
                global_steps
            )
            writer.add_scalar(
                'valid_acc',
                acc.avg,
                global_steps
            )
            writer.add_scalar(
                'degree_error_val',
                degree_error.avg,
                global_steps
            )
            writer.add_scalar(
                'excel_rate',
                excel_rate,
                global_steps
            )
            writer.add_scalar(
                'mid_rate',
                mid_rate,
                global_steps
            )
            writer.add_scalar(
                'poor_rate',
                poor_rate,
                global_steps
            )

            writer_dict['valid_global_steps'] = global_steps + 1

        perf_indicator = degree_error.avg
    return perf_indicator

# this is validate part
def validate(config, val_loader, val_dataset, model, criterions,  output_dir,
             tb_log_dir, writer_dict=None, draw_pic=False, save_pickle=False):
    batch_time = AverageMeter()
    loss_hkd_log = AverageMeter()
    loss_hoe_log = AverageMeter()
    losses = AverageMeter()
    degree_error = AverageMeter()
    acc = AverageMeter()
    Excellent = 0
    Mid_good = 0
    Poor_good = 0
    Poor_225 = 0
    Poor_45 = 0
    Total = 0

    # switch to evaluate mode
    model.eval()

    num_samples = len(val_dataset)
    idx = 0
    ori_list = []
    with torch.no_grad():
        end = time.time()
        for i, (input, target, target_weight, degree,  meta) in enumerate(val_loader):
            # compute output
            plane_output, hoe_output = model(input)

            # change to cuda format
            target = target.cuda(non_blocking=True)
            target_weight = target_weight.cuda(non_blocking=True)
            degree = degree.cuda(non_blocking=True)

            # compute loss
            if config.LOSS.USE_ONLY_HOE:
                loss_hoe = criterions['hoe_loss'](hoe_output, degree)
                loss_2d = loss_hoe
                loss = loss_hoe
            else:
                loss_2d = criterions['2d_pose_loss'](plane_output, target, target_weight)
                loss_hoe = ['hoe_loss'](hoe_output, degree)

                loss = loss_2d + 0.1 * loss_hoe

            num_images = input.size(0)
            # measure accuracy and record loss
            loss_hkd_log.update(loss_2d.item(), num_images)
            loss_hoe_log.update(loss_hoe.item(), num_images)
            losses.update(loss.item(), num_images)

            if 'tud' in config.DATASET.VAL_ROOT:
                avg_degree_error, excellent, mid, poor_225, poor, poor_45, gt_ori, pred_ori, cnt  = continous_comp_deg_error(hoe_output.detach().cpu().numpy(),
                                                                                   meta['val_dgree'].numpy())
                acc.update(mid/cnt, cnt)
                acc_label = 'mid15'
                has_hkd = False
            elif config.LOSS.USE_ONLY_HOE:
                avg_degree_error, excellent, mid, poor_225, poor, poor_45, gt_ori, pred_ori, cnt  = comp_deg_error(hoe_output.detach().cpu().numpy(),
                                                                                   degree.detach().cpu().numpy())
                acc.update(mid/cnt, cnt)
                acc_label = 'mid15'
                has_hkd = False
            else:
                avg_degree_error, excellent, mid, poor_225, poor, poor_45,gt_ori, pred_ori, _  = comp_deg_error(hoe_output.detach().cpu().numpy(),
                                                           degree.detach().cpu().numpy())
                _, avg_acc, cnt, pred = accuracy(plane_output.cpu().numpy(),
                                                 target.cpu().numpy())

                acc.update(avg_acc, cnt)
                acc_label = 'kpd_acc'
                has_hkd = True
            
            if draw_pic:
                ori_path = os.path.join(output_dir, 'orientation_img')
                if not os.path.exists(ori_path):
                    os.makedirs(ori_path)
                img_np = input.numpy()
                draw_orientation(img_np, gt_ori, pred_ori , ori_path, alis=str(i))

            if save_pickle:
                tamp_list = ori_numpy(gt_ori, pred_ori)
                ori_list = ori_list + tamp_list

            degree_error.update(avg_degree_error, num_images)

            Total += num_images
            Excellent += excellent
            Mid_good += mid
            Poor_good += poor
            Poor_45 += poor_45
            Poor_225 += poor_225

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            idx += num_images

            if i % config.PRINT_FREQ == 0:
                print_msg(step=i, loader_len=len(val_loader), batch_time=batch_time, has_hkd= has_hkd, loss_hkd=loss_hkd_log, loss_hoe=loss_hoe_log, losses=losses, degree_error=degree_error, acc_label=acc_label, acc=acc)

        if save_pickle:
            save_obj(ori_list, 'ori_list')
        excel_rate = Excellent / Total
        mid_rate = Mid_good / Total
        poor_rate = Poor_good / Total
        poor_225_rate = Poor_225 / Total
        poor_45_rate = Poor_45 / Total
        name_values = {'Degree_error': degree_error.avg, '5_Excel_rate': excel_rate, '15_Mid_rate': mid_rate, '225_rate': poor_225_rate, '30_Poor_rate': poor_rate, '45_poor_rate': poor_45_rate}
        _print_name_value(name_values, config.MODEL.NAME)

        if writer_dict:
            writer = writer_dict['writer']
            global_steps = writer_dict['valid_global_steps']
            writer.add_scalar(
                'valid_loss',
                losses.avg,
                global_steps
            )
            writer.add_scalar(
                'valid_hkd_loss',
                loss_hkd_log.avg,
                global_steps
            )
            writer.add_scalar(
                'valid_hoe_loss',
                loss_hoe_log.avg,
                global_steps
            )
            writer.add_scalar(
                'valid_acc',
                acc.avg,
                global_steps
            )
            writer.add_scalar(
                'degree_error_val',
                degree_error.avg,
                global_steps
            )
            writer.add_scalar(
                'excel_rate',
                excel_rate,
                global_steps
            )
            writer.add_scalar(
                'mid_rate',
                mid_rate,
                global_steps
            )
            writer.add_scalar(
                'poor_rate',
                poor_rate,
                global_steps
            )

            writer_dict['valid_global_steps'] = global_steps + 1

        perf_indicator = degree_error.avg
    return perf_indicator


def simple_train(config, train_loader, train_dataset, model, criterions, optimizer, epoch,
          output_dir, tb_log_dir, writer_dict):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_hoe_log = AverageMeter()
    losses = AverageMeter()
    degree_error = AverageMeter()
    acc = AverageMeter()

    # switch to train mode
    model.train()
    end = time.time()

    for i, (input, img, degree, meta) in enumerate(train_loader):
        data_time.update(time.time() - end)

        # compute output
        hoe_output = model(input)

        # change to cuda format
        degree = degree.cuda(non_blocking=True)
        # compute loss
        loss_hoe = criterions['hoe_loss'](hoe_output, degree)
        loss_2d = loss_hoe
        loss = loss_hoe
        # compute gradient and do update step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure accuracy and record loss
        loss_hoe_log.update(loss_hoe.item(), input.size(0))
        losses.update(loss.item(), input.size(0))

        if config.DATASET.DATASET == 'tud_dataset':
            avg_degree_error, _, mid, _ , _, _, _, _, cnt = continous_comp_deg_error(hoe_output.detach().cpu().numpy(),
                                               meta['val_dgree'].numpy())
            
            acc.update(mid/cnt, cnt)
            has_hkd=False
            acc_label = 'mid15'
        else:
            avg_degree_error, _, mid, _ , _, _, _, _, cnt= comp_deg_error(hoe_output.detach().cpu().numpy(),
                                                   degree.detach().cpu().numpy())
            acc.update(mid/cnt, cnt)
            has_hkd=False 
            acc_label = 'mid15'

        degree_error.update(avg_degree_error, input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % config.PRINT_FREQ == 0:
            print_msg(epoch = epoch, step=i, speed=input.size(0) / batch_time.val, has_hkd= has_hkd, loader_len=len(train_loader), batch_time=batch_time, loss_hoe=loss_hoe_log, losses=losses, degree_error=degree_error, acc_label=acc_label, acc=acc)

            writer = writer_dict['writer']
            global_steps = writer_dict['train_global_steps']
            writer.add_scalar('train_loss', losses.val, global_steps)
            writer.add_scalar('train_hoe_loss', loss_hoe_log.val, global_steps)
            writer.add_scalar('train_acc', acc.val, global_steps)
            writer.add_scalar('degree_error', degree_error.val, global_steps)
            writer_dict['train_global_steps'] = global_steps + 1


# this is validate part
def simple_validate(config, val_loader, val_dataset, model, criterions,  output_dir,
             tb_log_dir, writer_dict=None, draw_pic=False, save_pickle=False):
    batch_time = AverageMeter()
    loss_hoe_log = AverageMeter()
    losses = AverageMeter()
    degree_error = AverageMeter()
    acc = AverageMeter()
    Excellent = 0
    Mid_good = 0
    Poor_good = 0
    Poor_225 = 0
    Poor_45 = 0
    Total = 0

    # switch to evaluate mode
    model.eval()

    num_samples = len(val_dataset)
    idx = 0
    ori_list = []
    with torch.no_grad():
        end = time.time()
        for i, (input, img, degree,  meta) in enumerate(val_loader):
            # compute output
            hoe_output = model(input)

            # change to cuda format
            degree = degree.cuda(non_blocking=True)

            # compute loss
            loss_hoe = criterions['hoe_loss'](hoe_output, degree)
            loss_2d = loss_hoe
            loss = loss_hoe

            num_images = input.size(0)
            # measure accuracy and record loss
            loss_hoe_log.update(loss_hoe.item(), num_images)
            losses.update(loss.item(), num_images)

            if 'tud' in config.DATASET.VAL_ROOT:
                avg_degree_error, excellent, mid, poor_225, poor, poor_45, gt_ori, pred_ori, cnt  = continous_comp_deg_error(hoe_output.detach().cpu().numpy(),
                                                                                   meta['val_dgree'].numpy())
                acc.update(mid/cnt, cnt)
                acc_label = 'mid15'
                has_hkd = False
            else:
                avg_degree_error, excellent, mid, poor_225, poor, poor_45, gt_ori, pred_ori, cnt  = comp_deg_error(hoe_output.detach().cpu().numpy(),
                                                                                   degree.detach().cpu().numpy())
                acc.update(mid/cnt, cnt)
                acc_label = 'mid15'
                has_hkd = False


            if draw_pic:
                ori_path = os.path.join(output_dir, 'orientation_img')
                if not os.path.exists(ori_path):
                    os.makedirs(ori_path)
                img_np = input.numpy()
                draw_orientation(img_np, gt_ori, pred_ori , ori_path, alis=str(i))

            if save_pickle:
                tamp_list = ori_numpy(gt_ori, pred_ori)
                ori_list = ori_list + tamp_list

            degree_error.update(avg_degree_error, num_images)

            Total += num_images
            Excellent += excellent
            Mid_good += mid
            Poor_good += poor
            Poor_45 += poor_45
            Poor_225 += poor_225

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            idx += num_images

            if i % config.PRINT_FREQ == 0:
                print_msg(step=i, loader_len=len(val_loader), batch_time=batch_time, has_hkd= has_hkd, loss_hoe=loss_hoe_log, losses=losses, degree_error=degree_error, acc_label=acc_label, acc=acc)

        if save_pickle:
            save_obj(ori_list, 'ori_list')
        excel_rate = Excellent / Total
        mid_rate = Mid_good / Total
        poor_rate = Poor_good / Total
        poor_225_rate = Poor_225 / Total
        poor_45_rate = Poor_45 / Total
        name_values = {'Degree_error': degree_error.avg, '5_Excel_rate': excel_rate, '15_Mid_rate': mid_rate, '225_rate': poor_225_rate, '30_Poor_rate': poor_rate, '45_poor_rate': poor_45_rate}
        _print_name_value(name_values, config.MODEL.NAME)

        if writer_dict:
            writer = writer_dict['writer']
            global_steps = writer_dict['valid_global_steps']
            writer.add_scalar(
                'valid_loss',
                losses.avg,
                global_steps
            )
            writer.add_scalar(
                'valid_hoe_loss',
                loss_hoe_log.avg,
                global_steps
            )
            writer.add_scalar(
                'valid_acc',
                acc.avg,
                global_steps
            )
            writer.add_scalar(
                'degree_error_val',
                degree_error.avg,
                global_steps
            )
            writer.add_scalar(
                'excel_rate',
                excel_rate,
                global_steps
            )
            writer.add_scalar(
                'mid_rate',
                mid_rate,
                global_steps
            )
            writer.add_scalar(
                'poor_rate',
                poor_rate,
                global_steps
            )

            writer_dict['valid_global_steps'] = global_steps + 1

        perf_indicator = degree_error.avg
    return perf_indicator

# markdown format output
def _print_name_value(name_value, full_arch_name):
    names = name_value.keys()
    values = name_value.values()
    num_values = len(name_value)
    logger.info(
        '| Arch ' +
        ' '.join(['| {}'.format(name) for name in names]) +
        ' |'
    )
    logger.info('|---' * (num_values+1) + '|')

    if len(full_arch_name) > 15:
        full_arch_name = full_arch_name[:8] + '...'
    logger.info(
        '| ' + full_arch_name + ' ' +
        ' '.join(['| {:.3f}'.format(value) for value in values]) +
         ' |'
    )


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0
