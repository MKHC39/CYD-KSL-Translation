import os
import pdb
import sys
import copy
import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
import torch.nn.functional as F
import matplotlib.pyplot as plt
from evaluation.slr_eval.wer_calculation import evaluate
from torch.cuda.amp import autocast as autocast
from torch.cuda.amp import GradScaler
import gc


def _tqdm(iterable, **kwargs):
    try:
        from tqdm.auto import tqdm
        return tqdm(iterable, **kwargs)
    except Exception:
        # fallback: no progress bar if tqdm isn't installed
        return iterable

def _extract_clip_labels_from_ctc(label_1d: torch.Tensor, label_lgt: torch.Tensor) -> torch.Tensor:
    """
    CorrNet collate gives:
      label_1d: (sum_i L_i,)
      label_lgt: (B,)
    For ISLR, L_i should be 1. If not, we take the first token per sample.
    Returns: (B,) long
    """
    if label_lgt is None:
        return label_1d.view(-1)

    # Ensure 1D
    label_1d = label_1d.view(-1)
    label_lgt = label_lgt.view(-1)

    # Fast path: all length-1
    if torch.all(label_lgt == 1).item():
        return label_1d.view(-1)

    # General path: take first token of each sequence
    out = []
    start = 0
    for L in label_lgt.tolist():
        if L <= 0:
            out.append(torch.tensor(0, device=label_1d.device, dtype=label_1d.dtype))
        else:
            out.append(label_1d[start])
        start += int(L)
    return torch.stack(out, dim=0)


def seq_train(loader, model, optimizer, device, epoch_idx, recoder, cfg=None):
    # Route by task (mirrors your seq_eval pattern)
    if cfg is not None and cfg.dataset_info.get("task", "").lower() == "islr":
        return islr_train(cfg, loader, model, optimizer, device, epoch_idx, recoder)
    model.train()
    loss_value = []
    clr = [group['lr'] for group in optimizer.optimizer.param_groups]
    scaler = GradScaler()
    for batch_idx, data in enumerate(tqdm(loader)):
        vid = device.data_to_device(data[0])
        vid_lgt = device.data_to_device(data[1])
        label = device.data_to_device(data[2])
        label_lgt = device.data_to_device(data[3])
        optimizer.zero_grad()
        with autocast():
            ret_dict = model(vid, vid_lgt, label=label, label_lgt=label_lgt)
            loss, _ = model.criterion_calculation(ret_dict, label, label_lgt)
        if np.isinf(loss.item()) or np.isnan(loss.item()):
            print('loss is nan')
            #print(data[-1])
            print(str(data[1])+'  frames')
            print(str(data[3])+'  glosses')
            del ret_dict
            del loss
            continue
        scaler.scale(loss).backward()
        scaler.step(optimizer.optimizer)
        scaler.update()
        # nn.utils.clip_grad_norm_(model.rnn.parameters(), 5)
        loss_value.append(loss.item())
        if batch_idx % recoder.log_interval == 0:
            recoder.print_log(
                '\tEpoch: {}, Batch({}/{}) done. Loss: {:.8f}  lr:{:.6f}'
                    .format(epoch_idx, batch_idx, len(loader), loss.item(), clr[0]))
        del ret_dict
        del loss
    optimizer.scheduler.step()
    recoder.print_log('\tMean training loss: {:.10f}.'.format(np.mean(loss_value)))
    del loss_value
    del clr
    gc.collect()
    torch.cuda.empty_cache()
    return 


def islr_train(cfg, loader, model, optimizer, device, epoch_idx, recoder):
    """
    ISLR training with CrossEntropy on ret_dict["clip_logits"] (B, C).
    Keeps CorrNet loader/feeder format (label + label_lgt) so you don't have to
    rewrite collate_fn immediately.
    """
    model.train()
    ce = nn.CrossEntropyLoss()
    loss_value = []
    clr = [group['lr'] for group in optimizer.optimizer.param_groups]

    pbar = _tqdm(enumerate(loader), total=len(loader), desc=f"[train][ISLR] ep{epoch_idx}", leave=False)
    for batch_idx, data in pbar:
        vid = device.data_to_device(data[0])
        vid_lgt = device.data_to_device(data[1])
        label = device.data_to_device(data[2])
        label_lgt = device.data_to_device(data[3]) if len(data) > 3 else None

        # Convert CorrNet-style (flat label sequence) -> clip label id
        y = _extract_clip_labels_from_ctc(label, label_lgt).long()

        optimizer.zero_grad()

        ret_dict = model(vid, vid_lgt)  # don't pass label/label_lgt for ISLR
        if "clip_logits" not in ret_dict:
            raise RuntimeError('SLRModel forward() must return "clip_logits" for ISLR.')

        logits = ret_dict["clip_logits"]  # (B,C)

        # If you keep CorrNet's "+1 blank" class (id 0 unused), this still works
        # because your labels appear to already be >=1. (No remap needed.)
        loss = ce(logits, y)

        if not torch.isfinite(loss).item():
            recoder.print_log(f"[ISLR] loss nan/inf at batch {batch_idx}")
            del ret_dict, loss
            continue

        loss.backward()
        optimizer.optimizer.step()

        loss_value.append(loss.item())
        if batch_idx % recoder.log_interval == 0:
            recoder.print_log(
                '\t[ISLR] Epoch: {}, Batch({}/{}) done. Loss: {:.8f}  lr:{:.6f}'
                .format(epoch_idx, batch_idx, len(loader), loss.item(), clr[0])
            )

        if hasattr(pbar, "set_postfix"):
            pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{clr[0]:.2e}")

        del ret_dict, loss

    optimizer.scheduler.step()
    recoder.print_log('\t[ISLR] Mean training loss: {:.10f}.'.format(np.mean(loss_value) if loss_value else float("nan")))
    del loss_value, clr
    gc.collect()
    torch.cuda.empty_cache()


def islr_eval(cfg, loader, model, device, mode, epoch, work_dir, recoder):
    """
    ISLR eval: Top-1 accuracy on clip_logits.
    Returns a single float like WER% so main.py can stay unchanged:
      return 100 * (1 - top1_acc)
    """
    model.eval()
    total = 0
    correct = 0

    with torch.no_grad():
        pbar = _tqdm(enumerate(loader), total=len(loader), desc=f"[eval][ISLR] {mode} ep{epoch}", leave=False)
        for batch_idx, data in pbar:
            vid = device.data_to_device(data[0])
            vid_lgt = device.data_to_device(data[1])
            label = device.data_to_device(data[2])
            label_lgt = device.data_to_device(data[3]) if len(data) > 3 else None

            y = _extract_clip_labels_from_ctc(label, label_lgt).long()  # (B,)

            ret_dict = model(vid, vid_lgt)
            if "clip_logits" not in ret_dict:
                raise RuntimeError('SLRModel forward() must return "clip_logits" for ISLR.')

            logits = ret_dict["clip_logits"]  # (B,C)

            # Optional: ignore class 0 (blank) if your labels start at 1
            # This avoids ever predicting 0 by accident.
            # If you later remap labels to 0..C-1, remove this block.
            if logits.size(1) > 1 and torch.min(y).item() >= 1:
                pred = torch.argmax(logits[:, 1:], dim=1) + 1
            else:
                pred = torch.argmax(logits, dim=1)

            correct += int((pred == y).sum().item())
            total += int(y.numel())

            if hasattr(pbar, "set_postfix"):
                running_acc = (correct / total) if total > 0 else 0.0
                pbar.set_postfix(acc=f"{running_acc * 100:.2f}%")

    acc = (correct / total) if total > 0 else 0.0
    err_pct = 100.0 * (1.0 - acc)

    recoder.print_log(f"[ISLR] Epoch {epoch}, {mode} acc={acc*100:.2f}%, err={err_pct:.2f}%", f"{work_dir}/{mode}.txt")
    return err_pct


def seq_eval(cfg, loader, model, device, mode, epoch, work_dir, recoder,
             evaluate_tool="python"):
    if cfg.dataset_info.get("task", "").lower() == "islr":
        return islr_eval(cfg, loader, model, device, mode, epoch, work_dir, recoder)
    model.eval()
    total_sent = []
    total_info = []
    total_conv_sent = []
    stat = {i: [0, 0] for i in range(len(loader.dataset.dict))}
    for batch_idx, data in enumerate(tqdm(loader)):
        recoder.record_timer("device")
        vid = device.data_to_device(data[0])
        vid_lgt = device.data_to_device(data[1])
        label = device.data_to_device(data[2])
        label_lgt = device.data_to_device(data[3])
        with torch.no_grad():
            ret_dict = model(vid, vid_lgt, label=label, label_lgt=label_lgt)

        total_info += [file_name.split("|")[0] for file_name in data[-1]]
        total_sent += ret_dict['recognized_sents']
        total_conv_sent += ret_dict['conv_sents']
    try:
        python_eval = True if evaluate_tool == "python" else False
        write2file(work_dir + "output-hypothesis-{}.ctm".format(mode), total_info, total_sent)
        write2file(work_dir + "output-hypothesis-{}-conv.ctm".format(mode), total_info,
                   total_conv_sent)
        """
        conv_ret = evaluate(
            prefix=work_dir, mode=mode, output_file="output-hypothesis-{}-conv.ctm".format(mode),
            evaluate_dir=cfg.dataset_info['evaluation_dir'],
            evaluate_prefix=cfg.dataset_info['evaluation_prefix'],
            output_dir="epoch_{}_result/".format(epoch),
            python_evaluate=python_eval,
        )
        lstm_ret = evaluate(
            prefix=work_dir, mode=mode, output_file="output-hypothesis-{}.ctm".format(mode),
            evaluate_dir=cfg.dataset_info['evaluation_dir'],
            evaluate_prefix=cfg.dataset_info['evaluation_prefix'],
            output_dir="epoch_{}_result/".format(epoch),
            python_evaluate=python_eval,
            triplet=True,
        )
        """
    except:
        print("Unexpected error:", sys.exc_info()[0])
        lstm_ret = 100.0
    finally:
        pass
    # del conv_ret
    del total_sent
    del total_info
    del total_conv_sent
    del vid
    del vid_lgt
    del label
    del label_lgt
    gc.collect()
    recoder.print_log(f"Epoch {epoch}, {mode} {lstm_ret: 2.2f}%", f"{work_dir}/{mode}.txt")
    return lstm_ret


def seq_feature_generation(loader, model, device, mode, work_dir, recoder):
    model.eval()

    src_path = os.path.abspath(f"{work_dir}{mode}")
    tgt_path = os.path.abspath(f"./features/{mode}")
    if not os.path.exists("./features/"):
        os.makedirs("./features/")

    if os.path.islink(tgt_path):
        curr_path = os.readlink(tgt_path)
        if work_dir[1:] in curr_path and os.path.isabs(curr_path):
            return
        else:
            os.unlink(tgt_path)
    else:
        if os.path.exists(src_path) and len(loader.dataset) == len(os.listdir(src_path)):
            os.symlink(src_path, tgt_path)
            return

    for batch_idx, data in tqdm(enumerate(loader)):
        recoder.record_timer("device")
        vid = device.data_to_device(data[0])
        vid_lgt = device.data_to_device(data[1])
        with torch.no_grad():
            ret_dict = model(vid, vid_lgt)
        if not os.path.exists(src_path):
            os.makedirs(src_path)
        start = 0
        for sample_idx in range(len(vid)):
            end = start + data[3][sample_idx]
            filename = f"{src_path}/{data[-1][sample_idx].split('|')[0]}_features.npy"
            save_file = {
                "label": data[2][start:end],
                "features": ret_dict['framewise_features'][sample_idx][:, :vid_lgt[sample_idx]].T.cpu().detach(),
            }
            np.save(filename, save_file)
            start = end
        assert end == len(data[2])
    os.symlink(src_path, tgt_path)


def write2file(path, info, output):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        for sample_idx, sample in enumerate(output):
            for word_idx, word in enumerate(sample):
                f.write(
                    "{} 1 {:.2f} {:.2f} {}\n".format(
                        info[sample_idx],
                        word_idx * 1.0 / 100,
                        (word_idx + 1) * 1.0 / 100,
                        word[0],
                    )
                )
