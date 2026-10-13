"""Pinned native STRHub recognition, unchanged complete task-path timer."""
import argparse, importlib.metadata, json, re, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'joint_deploy'))
from worker_protocol import run_worker

p = argparse.ArgumentParser()
p.add_argument('--model-id', required=True)
p.add_argument('--cpu-preflight', action='store_true')
a = p.parse_args()
out = ROOT / 'results/module14_high_quality_heavy'
protocol = json.loads((out / 'ocr_gapfill_shortlist.json').read_text())
entry = next(e for e in protocol['entries'] if e['model_id'] == a.model_id)
source = ROOT / 'models/module14b' / ('parseq-' + entry['implementation_commit'])
sys.path.insert(0, str(source))
import torch
from PIL import Image
from strhub.models.utils import create_model
from strhub.data.module import SceneTextDataModule

torch.set_num_threads(1)
torch.manual_seed(0)
device = 'cpu' if a.cpu_preflight else 'cuda:0'
model = create_model(entry['experiment'], pretrained=False)
weights = torch.load(entry['weight_path'], map_location='cpu', weights_only=True)
target = model.model if entry['experiment'].startswith('parseq') else model
loading = target.load_state_dict(weights, strict=True)
assert not loading.missing_keys and not loading.unexpected_keys
model.eval().float().to(device)
transform = SceneTextDataModule.get_transform(model.hparams.img_size, augment=False, rotation=0)
metadata = {
    'model_id': a.model_id, 'task': 'ocr', 'revision': entry['revision'],
    'implementation_commit': entry['implementation_commit'], 'checkpoint': entry['weight_path'],
    'dtype': str(next(model.parameters()).dtype), 'device': device, 'load_success': True,
    'recognition_only': True, 'native_image_transform': str(transform),
    'native_hyperparameters': dict(model.hparams),
    'parameter_count_metadata': sum(p.numel() for p in model.parameters()),
    'native_decoding': entry['decoding'], 'framework_versions': {},
    'TF32_matmul': torch.backends.cuda.matmul.allow_tf32,
    'TF32_cudnn': torch.backends.cudnn.allow_tf32,
    'cuda_initialized': torch.cuda.is_initialized(),
    'strict_checkpoint_loading': True, 'no_compile_quantization_autocast': True,
}
for package in ['torch', 'torchvision', 'timm', 'pytorch-lightning', 'torchmetrics', 'numpy', 'pillow']:
    metadata['framework_versions'][package] = importlib.metadata.version(package)


def normalize(text):
    return re.sub(r'[^a-z0-9]', '', text.lower())


def handler(request):
    assert request['task_type'] == 'ocr'
    sample = request['sample']
    with torch.inference_mode():
        with Image.open(sample['input']) as image:
            inputs = transform(image.convert('RGB')).unsqueeze(0).to(device)
        logits = model(inputs)
        text, confidence = model.tokenizer.decode(logits.softmax(-1))
        prediction = normalize(text[0])
        reference = normalize(sample.get('normalized_label', sample['label']))
        torch.cuda.synchronize()
    return {'model': a.model_id, 'prediction': prediction, 'reference': reference,
            'correct': prediction == reference, 'raw_response': text[0], 'recognition_only': True}


if a.cpu_preflight:
    with torch.inference_mode():
        dummy = transform(Image.new('RGB', (128, 32))).unsqueeze(0)
        logits = model(dummy)
        labels, _ = model.tokenizer.decode(logits.softmax(-1))
    assert not torch.cuda.is_initialized()
    metadata.update(cpu_native_forward_success=True, cpu_logits_shape=list(logits.shape),
                    cpu_decode_success=len(labels) == 1)
    print('CPU_PREFLIGHT|' + json.dumps(metadata), flush=True)
else:
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    run_worker(a.model_id, handler, metadata)
    torch.cuda.synchronize()
    print('MEMORY_END|' + json.dumps({
        'allocated_current_mib': torch.cuda.memory_allocated() / 1048576,
        'reserved_current_mib': torch.cuda.memory_reserved() / 1048576,
        'allocated_peak_mib': torch.cuda.max_memory_allocated() / 1048576,
        'reserved_peak_mib': torch.cuda.max_memory_reserved() / 1048576,
    }), flush=True)
