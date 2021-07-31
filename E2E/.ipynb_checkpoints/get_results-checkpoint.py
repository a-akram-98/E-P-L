import io as sysio
from det3d.datasets.utils.eval import calculate_iou_partly
from det3d.datasets.utils.eval import prepare_data
from det3d.datasets.utils.eval import compute_statistics_jit
import numba
import numpy as np


def print_str(value, *arg, sstream=None):
    if sstream is None:
        sstream = sysio.StringIO()
    sstream.truncate(0)
    sstream.seek(0)
    print(value, *arg, file=sstream)
    return sstream.getvalue()


def get_mAP(prec):
    sums = 0
    for i in range(0, prec.shape[-1], 4):
        sums = sums + prec[..., i]
    return sums / 11 * 100


@numba.jit
def get_thresholds(scores: np.ndarray, num_gt, num_sample_pts=41):
    scores.sort()
    scores = scores[::-1]
    current_recall = 0
    thresholds = []
    for i, score in enumerate(scores):
        l_recall = (i + 1) / num_gt
        if i < (len(scores) - 1):
            r_recall = (i + 2) / num_gt
        else:
            r_recall = l_recall
        if ((r_recall - current_recall) < (current_recall - l_recall)) and (
                i < (len(scores) - 1)
        ):
            continue
        # recall = l_recall
        thresholds.append(score)
        current_recall += 1 / (num_sample_pts - 1.0)
    # print(len(thresholds), len(scores), num_gt)
    return thresholds


def clean_data(gt_anno, dt_anno, current_class, difficulty):
    CLASS_NAMES = [
        "car",
        "pedestrian",
        "bicycle",
        "truck",
        "bus",
        "trailer",
        "construction_vehicle",
        "motorcycle",
        "barrier",
        "traffic_cone",
        "cyclist",
    ]
    MIN_HEIGHT = [40, 25, 25]
    MAX_OCCLUSION = [0, 1, 2]
    MAX_TRUNCATION = [0.15, 0.3, 0.5]
    dc_bboxes, ignored_gt, ignored_dt = [], [], []
    current_cls_name = CLASS_NAMES[current_class].lower()
    num_gt = len(gt_anno["name"])
    num_dt = len(dt_anno["name"])
    num_valid_gt = 0
    for i in range(num_gt):
        bbox = gt_anno["bbox"][i]
        gt_name = gt_anno["name"][i].lower()
        height = bbox[3] - bbox[1]
        valid_class = -1
        if gt_name == current_cls_name:
            valid_class = 1
        elif (
                current_cls_name == "Pedestrian".lower()
                and "Person_sitting".lower() == gt_name
        ):
            valid_class = 0
        elif current_cls_name == "Car".lower() and "Van".lower() == gt_name:
            valid_class = 0
        else:
            valid_class = -1
        ignore = False
        if (
                (gt_anno["occluded"][i] > MAX_OCCLUSION[difficulty])
                or (gt_anno["truncated"][i] > MAX_TRUNCATION[difficulty])
                or (height <= MIN_HEIGHT[difficulty])
        ):
            ignore = True
        if valid_class == 1 and not ignore:
            ignored_gt.append(0)
            num_valid_gt += 1
        elif valid_class == 0 or (ignore and (valid_class == 1)):
            ignored_gt.append(1)
        else:
            ignored_gt.append(-1)
        # for i in range(num_gt):
        if (gt_anno["name"][i] == "DontCare") or (gt_anno["name"][i] == "ignore"):
            dc_bboxes.append(gt_anno["bbox"][i])
    for i in range(num_dt):
        if dt_anno["name"][i].lower() == current_cls_name:
            valid_class = 1
        else:
            valid_class = -1
        height = abs(dt_anno["bbox"][i, 3] - dt_anno["bbox"][i, 1])
        if height < MIN_HEIGHT[difficulty]:
            ignored_dt.append(1)
        elif valid_class == 1:
            ignored_dt.append(0)
        else:
            ignored_dt.append(-1)

    return num_valid_gt, ignored_gt, ignored_dt, dc_bboxes


def get_split_parts(num, num_part):
    same_part = num // num_part
    remain_num = num % num_part
    if remain_num == 0:
        return [same_part] * num_part
    else:
        return [same_part] * num_part + [remain_num]


def fused_compute_statistics(
        overlaps,
        pr,
        gt_nums,
        dt_nums,
        dc_nums,
        gt_datas,
        dt_datas,
        dontcares,
        ignored_gts,
        ignored_dets,
        metric,
        min_overlap,
        thresholds,
        compute_aos=False,
):
    gt_num = 0
    dt_num = 0
    dc_num = 0
    for i in range(gt_nums.shape[0]):
        for t, thresh in enumerate(thresholds):
            overlap = overlaps[
                      dt_num: dt_num + dt_nums[i], gt_num: gt_num + gt_nums[i]
                      ]

            gt_data = gt_datas[gt_num: gt_num + gt_nums[i]]
            dt_data = dt_datas[dt_num: dt_num + dt_nums[i]]
            ignored_gt = ignored_gts[gt_num: gt_num + gt_nums[i]]
            ignored_det = ignored_dets[dt_num: dt_num + dt_nums[i]]
            dontcare = dontcares[dc_num: dc_num + dc_nums[i]]
            tp, fp, fn, similarity, _ = compute_statistics_jit(
                overlap,
                gt_data,
                dt_data,
                ignored_gt,
                ignored_det,
                dontcare,
                metric,
                min_overlap=min_overlap,
                thresh=thresh,
                compute_fp=True,
                compute_aos=compute_aos,
            )
            pr[t, 0] += tp
            pr[t, 1] += fp
            pr[t, 2] += fn
            if similarity != -1:
                pr[t, 3] += similarity
        gt_num += gt_nums[i]
        dt_num += dt_nums[i]
        dc_num += dc_nums[i]


def eval_class_v3(
        gt_annos,
        dt_annos,
        current_classes,
        difficultys,
        metric,
        min_overlaps,
        compute_aos=False,
        z_axis=1,
        z_center=1.0,
        num_parts=50,
):
    """Kitti eval. support 2d/bev/3d/aos eval. support 0.5:0.05:0.95 coco AP.
    Args:
        gt_annos: dict, must from get_label_annos() in kitti_common.py
        dt_annos: dict, must from get_label_annos() in kitti_common.py
        current_class: int, 0: car, 1: pedestrian, 2: cyclist
        difficulty: int. eval difficulty, 0: easy, 1: normal, 2: hard
        metric: eval type. 0: bbox, 1: bev, 2: 3d
        min_overlap: float, min overlap. official:
            [[0.7, 0.5, 0.5], [0.7, 0.5, 0.5], [0.7, 0.5, 0.5]]
            format: [metric, class]. choose one from matrix above.
        num_parts: int. a parameter for fast calculate algorithm

    Returns:
        dict of recall, precision and aos
    """
    print(len(gt_annos))
    print(len(dt_annos))
    assert len(gt_annos) == len(dt_annos)
    num_examples = len(gt_annos)
    split_parts = get_split_parts(num_examples, num_parts)
    split_parts = [i for i in split_parts if i != 0]

    rets = calculate_iou_partly(
        dt_annos, gt_annos, metric, num_parts, z_axis=z_axis, z_center=z_center
    )
    overlaps, parted_overlaps, total_dt_num, total_gt_num = rets
    N_SAMPLE_PTS = 41
    num_minoverlap = len(min_overlaps)
    num_class = len(current_classes)
    num_difficulty = len(difficultys)
    precision = np.zeros([num_class, num_difficulty, num_minoverlap, N_SAMPLE_PTS])
    recall = np.zeros([num_class, num_difficulty, num_minoverlap, N_SAMPLE_PTS])
    aos = np.zeros([num_class, num_difficulty, num_minoverlap, N_SAMPLE_PTS])
    all_thresholds = np.zeros([num_class, num_difficulty, num_minoverlap, N_SAMPLE_PTS])
    for m, current_class in enumerate(current_classes):
        for l, difficulty in enumerate(difficultys):
            rets = prepare_data(
                gt_annos,
                dt_annos,
                current_class,
                difficulty=difficulty,
                clean_data=clean_data,
            )
            (
                gt_datas_list,
                dt_datas_list,
                ignored_gts,
                ignored_dets,
                dontcares,
                total_dc_num,
                total_num_valid_gt,
            ) = rets
            for k, min_overlap in enumerate(min_overlaps[:, metric, m]):
                thresholdss = []
                for i in range(len(gt_annos)):
                    rets = compute_statistics_jit(
                        overlaps[i],
                        gt_datas_list[i],
                        dt_datas_list[i],
                        ignored_gts[i],
                        ignored_dets[i],
                        dontcares[i],
                        metric,
                        min_overlap=min_overlap,
                        thresh=0.0,
                        compute_fp=False,
                    )
                    tp, fp, fn, similarity, thresholds = rets
                    thresholdss += thresholds.tolist()
                thresholdss = np.array(thresholdss)
                thresholds = get_thresholds(thresholdss, total_num_valid_gt)
                thresholds = np.array(thresholds)
                # print(thresholds)
                all_thresholds[m, l, k, : len(thresholds)] = thresholds
                pr = np.zeros([len(thresholds), 4])
                idx = 0
                for j, num_part in enumerate(split_parts):
                    gt_datas_part = np.concatenate(
                        gt_datas_list[idx: idx + num_part], 0
                    )
                    dt_datas_part = np.concatenate(
                        dt_datas_list[idx: idx + num_part], 0
                    )
                    dc_datas_part = np.concatenate(dontcares[idx: idx + num_part], 0)
                    ignored_dets_part = np.concatenate(
                        ignored_dets[idx: idx + num_part], 0
                    )
                    ignored_gts_part = np.concatenate(
                        ignored_gts[idx: idx + num_part], 0
                    )
                    fused_compute_statistics(
                        parted_overlaps[j],
                        pr,
                        total_gt_num[idx: idx + num_part],
                        total_dt_num[idx: idx + num_part],
                        total_dc_num[idx: idx + num_part],
                        gt_datas_part,
                        dt_datas_part,
                        dc_datas_part,
                        ignored_gts_part,
                        ignored_dets_part,
                        metric,
                        min_overlap=min_overlap,
                        thresholds=thresholds,
                        compute_aos=compute_aos,
                    )
                    idx += num_part
                for i in range(len(thresholds)):
                    # recall[m, l, k, i] = pr[i, 0] / (pr[i, 0] + pr[i, 2])
                    precision[m, l, k, i] = pr[i, 0] / (pr[i, 0] + pr[i, 1])
                    if compute_aos:
                        aos[m, l, k, i] = pr[i, 3] / (pr[i, 0] + pr[i, 1])
                for i in range(len(thresholds)):
                    precision[m, l, k, i] = np.max(precision[m, l, k, i:], axis=-1)
                    if compute_aos:
                        aos[m, l, k, i] = np.max(aos[m, l, k, i:], axis=-1)
                # use interp to calculate recall
                """
                current_recalls = np.linspace(0, 1, 41)
                prec_unique, inds = np.unique(precision[m, l, k], return_index=True)
                current_recalls = current_recalls[inds]
                f = interp1d(prec_unique, current_recalls)
                precs_for_recall = np.linspace(0, 1, 41)
                max_prec = np.max(precision[m, l, k])
                valid_prec = precs_for_recall < max_prec
                num_valid_prec = valid_prec.sum()
                recall[m, l, k, :num_valid_prec] = f(precs_for_recall[valid_prec])
                """
    ret_dict = {
        "recall": recall,  # [num_class, num_difficulty, num_minoverlap, N_SAMPLE_PTS]
        "precision": precision,
        "orientation": aos,
        "thresholds": all_thresholds,
        "min_overlaps": min_overlaps,
    }
    return ret_dict


def do_eval_v3(
        gt_annos,
        dt_annos,
        current_classes,
        min_overlaps,
        compute_aos=False,
        difficultys=(0, 1, 2),
        z_axis=1,
        z_center=1.0,
):
    # min_overlaps: [num_minoverlap, metric, num_class]
    types = ["bbox", "bev", "3d"]
    metrics = {}
    for i in range(3):
        ret = eval_class_v3(
            gt_annos,
            dt_annos,
            current_classes,
            difficultys,
            i,
            min_overlaps,
            compute_aos,
            z_axis=z_axis,
            z_center=z_center,
        )
        metrics[types[i]] = ret
    return metrics


def get_official_eval_result(
        gt_annos, dt_annos, current_classes, difficultys=[0, 1, 2], z_axis=1, z_center=1.0
):
    """
        gt_annos and dt_annos must contains following keys:
        [bbox, location, dimensions, rotation, score]
    """
    overlap_mod = np.array(
        [
            [0.7, 0.5, 0.5, 0.7, 0.7, 0.7, 0.7, 0.5, 0.5, 0.5, 0.5],
            [0.7, 0.5, 0.5, 0.7, 0.7, 0.7, 0.7, 0.5, 0.5, 0.5, 0.5],
            [0.7, 0.5, 0.5, 0.7, 0.7, 0.7, 0.7, 0.5, 0.5, 0.5, 0.5],
        ]
    )
    overlap_easy = np.array(
        [
            [0.7, 0.5, 0.5, 0.7, 0.7, 0.7, 0.7, 0.5, 0.25, 0.25, 0.5],
            [0.5, 0.25, 0.25, 0.5, 0.5, 0.5, 0.5, 0.25, 0.25, 0.25, 0.25],
            [0.5, 0.25, 0.25, 0.5, 0.5, 0.5, 0.5, 0.25, 0.25, 0.25, 0.25],
        ]
    )
    min_overlaps = np.stack([overlap_mod, overlap_easy], axis=0)  # [2, 3, 5]
    class_to_name = {
        0: "car",
        1: "pedestrian",
        2: "bicycle",
        3: "truck",
        4: "bus",
        5: "trailer",
        6: "construction_vehicle",
        7: "motorcycle",
        8: "barrier",
        9: "traffic_cone",
        10: "cyclist",
    }
    name_to_class = {v: n for n, v in class_to_name.items()}
    if not isinstance(current_classes, (list, tuple)):
        current_classes = [current_classes]
    current_classes_int = []
    for curcls in current_classes:
        if isinstance(curcls, str):
            current_classes_int.append(name_to_class[curcls.lower()])
        else:
            current_classes_int.append(curcls)
    current_classes = current_classes_int
    min_overlaps = min_overlaps[:, :, current_classes]
    result = ""
    # check whether alpha is valid
    compute_aos = False
    for anno in dt_annos:
        if anno["alpha"].shape[0] != 0:
            if anno["alpha"][0] != -10:
                compute_aos = True
            break
    # TODO dt2gt
    metrics = do_eval_v3(
        gt_annos,
        dt_annos,
        current_classes,
        min_overlaps,
        compute_aos,
        difficultys,
        z_axis=z_axis,
        z_center=z_center,
    )
    print(len(current_classes))
    detail = {}
    for j, curcls in enumerate(current_classes):
        # mAP threshold array: [num_minoverlap, metric, class]
        # mAP result: [num_class, num_diff, num_minoverlap]
        class_name = class_to_name[curcls]
        detail[class_name] = {}
        for i in range(min_overlaps.shape[0]):
            mAPbbox = get_mAP(metrics["bbox"]["precision"][j, :, i])
            mAPbev = get_mAP(metrics["bev"]["precision"][j, :, i])
            mAP3d = get_mAP(metrics["3d"]["precision"][j, :, i])
            detail[class_name][f"bbox@{min_overlaps[i, 0, j]:.2f}"] = mAPbbox.tolist()
            detail[class_name][f"bev@{min_overlaps[i, 1, j]:.2f}"] = mAPbev.tolist()
            detail[class_name][f"3d@{min_overlaps[i, 2, j]:.2f}"] = mAP3d.tolist()

            result += print_str(
                (
                    f"{class_to_name[curcls]} "
                    "AP(Average Precision)@{:.2f}, {:.2f}, {:.2f}:".format(
                        *min_overlaps[i, :, j]
                    )
                )
            )
            mAPbbox = ", ".join(f"{v:.2f}" for v in mAPbbox)
            mAPbev = ", ".join(f"{v:.2f}" for v in mAPbev)
            mAP3d = ", ".join(f"{v:.2f}" for v in mAP3d)
            result += print_str(f"bbox AP:{mAPbbox}")
            result += print_str(f"bev  AP:{mAPbev}")
            result += print_str(f"3d   AP:{mAP3d}")
            if compute_aos:
                mAPaos = get_mAP(metrics["bbox"]["orientation"][j, :, i])
                detail[class_name][f"aos"] = mAPaos.tolist()
                mAPaos = ", ".join(f"{v:.2f}" for v in mAPaos)
                result += print_str(f"aos  AP:{mAPaos}")
    return {
        "result": result,
        "detail": detail,
    }
