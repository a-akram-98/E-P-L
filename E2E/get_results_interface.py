from det3d.datasets.kitti import kitti
#from det3d.datasets.kitti.get_results import get_official_eval_result
#from get_results import get_official_eval_result
from det3d.datasets.kitti.eval import get_official_eval_result
def _read_imageset_file(path):
    with open(path, "r") as f:
        lines = f.readlines()
    return [int(line) for line in lines]


def evaluate(
    label_path,
    result_path,
    label_split_file,
    current_class=0,
    coco=False,
    score_thresh=-1,
):
    dt_annos = kitti.get_label_annos(result_path)
    if score_thresh > 0:
        dt_annos = kitti.filter_annos_low_score(dt_annos, score_thresh)
    val_image_ids = _read_imageset_file(label_split_file)
    gt_annos = kitti.get_label_annos(label_path, val_image_ids)
    ##print(len(gt_annos))
    #print(len(dt_annos))
    return get_official_eval_result(gt_annos, dt_annos, current_class)


# calling on test example

################neeeded imports while testing ##################################

#from det3d.datasets.kitti.get_results_interface import (evaluate as kitti_evaluate,)
###########################################################

# res_dir = os.path.join(cfg.work_dir, "predictions")
#note : choose the appropriate cf.work_dir for your case in paperspace
#         os.makedirs(res_dir, exist_ok=True)

# note  detections is one of the outputs of the test function in test document 
# as this example beloow here :
# result_dict, detections = test(data_loader, model, save_dir=None, distributed=distributed)

#         for dt in detections:
#             with open(os.path.join(res_dir, "%06d.txt" % int(dt["metadata"]["token"])), "w") as fout:
#                 lines = kitti.annos_to_kitti_label(dt)
#                 for line in lines:
#                     fout.write(line + "\n")
# data_root = "/content/KITTI_DATASET"
# gt_labels_dir = data_root + "/training/label_2"
# label_split_file = "/content/CIA_SSD/cia/det3d/datasets/ImageSets/val.txt"
# ap_result_str, ap_dict = kitti_evaluate(gt_labels_dir, res_dir, label_split_file=label_split_file, current_class=0,)
# print(ap_result_str)