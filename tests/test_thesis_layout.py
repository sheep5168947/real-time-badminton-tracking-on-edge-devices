from pathlib import Path
import importlib
import torch

def test_thesis_model_accepts_all_export_resolutions():
    from thesis.models.miniroi_unet import MiniROIUNet
    model = MiniROIUNet().eval()
    with torch.inference_mode():
        for height, width in [(32,32),(64,64),(128,128),(288,512)]:
            result = model(torch.zeros(1,9,height,width))
            assert result.shape == (1,1,height,width)

def test_moved_entrypoints_import_without_running_jobs():
    for name in [
        'thesis.data.cache_teacher_heatmaps', 'thesis.training.train_student_128',
        'thesis.training.train_student_64', 'thesis.training.train_student_32',
        'thesis.platforms.pc.student_realtime_events', 'thesis.platforms.pc.teacher_realtime',
        'thesis.analysis.find_mask_cases', 'thesis.evaluation.compare_motion_mask',
    ]:
        assert importlib.import_module(name)

def test_evaluation_manifests_resolve_under_project():
    import csv
    root = Path(__file__).resolve().parents[1]
    for manifest in (root/'configs/thesis').glob('*.csv'):
        with manifest.open(encoding='utf-8',newline='') as stream:
            for row in csv.DictReader(stream):
                for field in ['video_path','labels_path']:
                    path = root/row[field]
                    assert not Path(row[field]).is_absolute()
                    assert path.resolve().is_relative_to(root.resolve())

def test_mask_metric_boundaries():
    from thesis.evaluation.compare_motion_mask import self_test
    self_test()
