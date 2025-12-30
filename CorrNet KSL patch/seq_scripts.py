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

def seq_train(loader, model, optimizer, device, epoch_idx, recoder):
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
    except:
        print("Unexpected error:", sys.exc_info()[0])
        lstm_ret = 100.0
    finally:
        pass
    del conv_ret
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

def islr_eval(cfg, loader, model, device, mode, epoch, work_dir, recoder):
    """
    ISLR evaluation for single-gloss clips.
    - Uses sequence_logits directly (no decode / no CTM / no WER).
    - Masks blank class (id=0) so Top-1 isn't dominated by CTC blank.
    - Optionally prints a confusion summary.

    Returns: (100 - Top1Acc) to preserve the "lower is better" convention
             used by WER-based callers.
    """
    import os
    import torch
    from tqdm import tqdm

    model.eval()

    num_classes = getattr(model, "num_classes", None)
    if num_classes is None:
        raise RuntimeError("Model has no attribute `num_classes`; cannot size confusion matrix safely.")

    # Confusion matrix: rows=GT, cols=Pred
    conf = torch.zeros((num_classes, num_classes), dtype=torch.int64)

    correct = 0
    total = 0

    for batch_idx, data in enumerate(tqdm(loader)):
        vid = device.data_to_device(data[0])       # (B,T,C,H,W) per your model forward expectation
        vid_lgt = device.data_to_device(data[1])   # (B,)
        label = device.data_to_device(data[2])     # concatenated targets
        label_lgt = device.data_to_device(data[3]) # (B,)
        # data[-1] is info; unused here

        with torch.no_grad():
            ret_dict = model(vid, vid_lgt)

        # ---- Extract per-sample GT id (assumes single-gloss: label_lgt[i] == 1) ----
        gt_ids = []
        ptr = 0
        B = int(vid_lgt.shape[0])
        for i in range(B):
            L = int(label_lgt[i])
            if L <= 0:
                # no label? treat as unknown/blank
                gt_ids.append(0)
            else:
                gt_ids.append(int(label[ptr].item()))
            ptr += L

        # ---- Predict from sequence_logits (T,B,C) ----
        logits = ret_dict["sequence_logits"]       # (T,B,C)
        logits = logits.permute(1, 0, 2)           # (B,T,C)

        for b in range(B):
            T = int(vid_lgt[b])
            if T <= 0:
                pred_id = 0
            else:
                frame_logits = logits[b, :T]       # (T,C)

                # Time-aggregate (mean) then mask blank (class 0)
                mean_logits = frame_logits.mean(dim=0)  # (C,)
                mean_logits = mean_logits.clone()
                mean_logits[0] = -1e9                  # ignore blank id 0
                pred_id = int(mean_logits.argmax().item())

            gt_id = int(gt_ids[b])

            # update stats
            total += 1
            if pred_id == gt_id and gt_id != 0:
                correct += 1

            # update confusion (guard indices)
            if 0 <= gt_id < num_classes and 0 <= pred_id < num_classes:
                conf[gt_id, pred_id] += 1

        # cleanup
        del ret_dict

    top1 = (100.0 * correct / total) if total > 0 else 0.0

    # ---- Optional: write confusion matrix + quick confusion summary ----
    out_dir = os.path.join(work_dir, f"epoch_{epoch}_islr")
    os.makedirs(out_dir, exist_ok=True)

    # Save raw confusion matrix
    torch.save(conf.cpu(), os.path.join(out_dir, f"confusion_{mode}.pt"))

    # Print top off-diagonal confusions among non-blank classes (1..)
    conf_nb = conf[1:, 1:].cpu()
    if conf_nb.numel() > 0:
        flat = conf_nb.flatten()
        k = min(15, flat.numel())
        topv, topi = torch.topk(flat, k=k)
        C = conf_nb.size(1)
        lines = []
        for v, idx in zip(topv.tolist(), topi.tolist()):
            if v == 0:
                break
            gt = idx // C
            pr = idx % C
            if gt != pr:
                # +1 to map back to original ids
                lines.append(f"GT {gt+1} -> Pred {pr+1}: {v}")
        if lines:
            recoder.print_log(
                f"Top confusions ({mode}): " + " | ".join(lines[:10]),
                os.path.join(out_dir, f"confusions_{mode}.txt"),
            )

    recoder.print_log(
        f"Epoch {epoch}, {mode} Top1Acc {top1:.2f}%",
        f"{work_dir}/{mode}.txt"
    )

    # Return WER-like metric so caller's "best_dev" logic remains consistent
    return 100.0 - top1




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
