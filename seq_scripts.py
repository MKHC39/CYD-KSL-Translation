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

"""
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
"""

def seq_train(loader, model, optimizer, device, epoch_idx, recoder):
    import gc
    import numpy as np
    import torch

    model.train()
    loss_value = []
    clr = [group['lr'] for group in optimizer.optimizer.param_groups]

    for batch_idx, data in enumerate(loader):
        vid = device.data_to_device(data[0])
        vid_lgt = device.data_to_device(data[1])
        label = device.data_to_device(data[2])
        label_lgt = device.data_to_device(data[3])

        optimizer.zero_grad()

        # ---- NO AMP ----
        ret_dict = model(vid, vid_lgt, label=label, label_lgt=label_lgt)
        loss, _ = model.criterion_calculation(ret_dict, label, label_lgt)

        if (not torch.isfinite(loss).item()):
            print('loss is nan/inf')
            print(str(data[1]) + '  frames')
            print(str(data[3]) + '  glosses')

            # extra: tell us if logits are already bad
            x = ret_dict["sequence_logits"]
            if torch.isnan(x).any() or torch.isinf(x).any():
                print("sequence_logits already NaN/Inf before backward!")

            del ret_dict
            del loss
            continue

        loss.backward()
        optimizer.optimizer.step()

        loss_value.append(loss.item())
        if batch_idx % recoder.log_interval == 0:
            recoder.print_log(
                '\tEpoch: {}, Batch({}/{}) done. Loss: {:.8f}  lr:{:.6f}'
                .format(epoch_idx, batch_idx, len(loader), loss.item(), clr[0])
            )

        del ret_dict
        del loss

    optimizer.scheduler.step()
    recoder.print_log('\tMean training loss: {:.10f}.'.format(np.mean(loss_value) if loss_value else float("nan")))
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

def islr_eval(self, data_loader, device, config, epoch, global_step, generate_cfg, **kwargs):
    """
    ISLR eval for CorrNet-style SLRModel output.

    Expects model forward() to return:
      - ret_dict["sequence_logits"]: (T,B,C) (CorrNet default) OR (B,T,C)
      - ret_dict["feat_len"]: (B,) effective temporal length after temporal conv stack

    Computes Top-K accuracy over non-blank classes (blank id = 0).
    """
    self.eval()

    total = 0
    top1_correct = 0
    topk_correct = {k: 0 for k in getattr(config, "topk", [1, 5, 10]) if k >= 1}

    # Optional: log a few debugging stats
    debug_max_batches = getattr(config, "eval_debug_max_batches", 0)
    debug_batches_done = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            # CorrNet feeders typically return (vid, vid_lgt, label, label_lgt, ...) but
            # you said feeder is correct, so keep this minimal and robust.
            if isinstance(batch, (list, tuple)):
                vid = batch[0]
                vid_lgt = batch[1] if len(batch) > 1 else None
                label = batch[2] if len(batch) > 2 else None
            elif isinstance(batch, dict):
                vid = batch.get("video", batch.get("vid", None))
                vid_lgt = batch.get("video_len", batch.get("vid_lgt", None))
                label = batch.get("label_id", batch.get("label", None))
            else:
                raise TypeError(f"Unsupported batch type: {type(batch)}")

            if vid is None or label is None:
                raise RuntimeError("Eval batch missing video or label.")

            vid = vid.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True).view(-1)  # (B,)

            # Forward
            ret_dict = self(vid, vid_lgt=vid_lgt, label=label, epoch=epoch, **kwargs)

            logits = ret_dict["sequence_logits"]  # (T,B,C) or (B,T,C)

            # Infer/logits -> (B,T,C)
            if logits.dim() != 3:
                raise RuntimeError(f"sequence_logits should be 3D, got {logits.shape}")

            B = label.shape[0]
            if logits.shape[0] == B:
                logits_btc = logits  # (B,T,C)
            elif logits.shape[1] == B:
                logits_btc = logits.permute(1, 0, 2).contiguous()  # (B,T,C)
            else:
                raise RuntimeError(f"Can't infer batch dim: logits={logits.shape}, label={label.shape}")

            # Use model's effective length after temporal conv stack
            if "feat_len" not in ret_dict:
                raise RuntimeError("ret_dict missing feat_len; needed for ISLR eval slicing.")
            feat_len = ret_dict["feat_len"].to("cpu")  # (B,)

            C = logits_btc.shape[-1]
            ks = sorted([k for k in topk_correct.keys() if k <= (C - 1)])  # exclude blank

            for b in range(B):
                T = int(feat_len[b])
                if T <= 0:
                    continue

                # Slice valid timesteps
                frame_logits = logits_btc[b, :T, :]  # (T,C)

                # CTC-consistent aggregation:
                # evidence that class appears somewhere in time
                frame_logp = frame_logits.log_softmax(dim=-1)      # (T,C)
                agg = frame_logp.logsumexp(dim=0)                  # (C,)

                # Exclude blank
                agg = agg.clone()
                agg[0] = -1e9

                # Ground truth id (label_len == 1, label_id >= 1)
                gt = int(label[b].item())

                # Top-1
                pred1 = int(torch.argmax(agg).item())
                top1_correct += int(pred1 == gt)

                # Top-K
                if ks:
                    topk_ids = torch.topk(agg, k=max(ks), dim=0).indices.tolist()
                    for k in ks:
                        topk_correct[k] += int(gt in topk_ids[:k])

                total += 1

            # Optional debugging: blank ratio, etc.
            if debug_batches_done < debug_max_batches:
                # Compute average blank-ratio on argmax frames (just as a diagnostic)
                # Note: this uses raw argmax of logits, not the agg decision.
                pred_frames = logits_btc.argmax(dim=-1)  # (B,T)
                blank_ratio = (pred_frames == 0).float().mean().item()
                print(f"[ISLR eval debug] batch={batch_idx} logits={tuple(logits.shape)} "
                      f"btc={tuple(logits_btc.shape)} blank_ratio={blank_ratio:.3f}")
                debug_batches_done += 1

    # Package results
    top1 = (top1_correct / total) if total > 0 else 0.0
    results = {
        "top1_acc": top1,  # 0..1
        "top1_err": 1.0 - top1,  # 0..1
        "top1_wer": 100.0 * (1.0 - top1),  # 0..100  (WER%-like)
    }
    for k, v in topk_correct.items():
        results[f"top{k}_acc"] = (v / total) if total > 0 else 0.0

    return results



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
