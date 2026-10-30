import copy
from pathlib import Path
import sys
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'scripts'))
import gpu_mutation_experiment as gpu

DEVICE = dict(device_name='test GPU', vendor_id=1, device_id=2, device_type='discrete_gpu',
              api_version='1.3.0', driver_version='1', queue_family=0, software_device=False, identity='1:2:1:uuid')


class DevicePolicy(unittest.TestCase):
    def test_hardware(self):
        self.assertEqual(gpu.device_policy(DEVICE), DEVICE)

    def test_missing_device(self):
        with self.assertRaisesRegex(ValueError, 'Missing device'):
            gpu.device_policy({})

    def test_software_is_explicit_diagnostic(self):
        device = dict(DEVICE, software_device=True, device_type='cpu')
        with self.assertRaisesRegex(ValueError, 'inconclusive'):
            gpu.device_policy(device)
        self.assertEqual(gpu.device_policy(device, True), device)

    def test_queue_required(self):
        with self.assertRaisesRegex(ValueError, 'Invalid device'):
            gpu.device_policy(dict(DEVICE, queue_family=-1))

    def test_repeat_fingerprint(self):
        m = dict(state='measured_pending_review', revision='a', versions={}, device=DEVICE,
                 fingerprint={}, environment={}, hardware_measurement=True, mutants=[dict(
                     id=f'cxx_lt_to_le:{gpu.SOURCE}:1:2:1:3', status='killed', file=gpu.SOURCE,
                     mutator='cxx_lt_to_le', line=1, column=2)])
        runs = [copy.deepcopy(m) for _ in range(3)]
        self.assertTrue(gpu.compare_runs(runs)['stable'])
        runs[1]['mutants'][0]['status'] = 'timed_out'
        self.assertEqual(gpu.compare_runs(runs)['unstable_ids'], [m['mutants'][0]['id']])
        runs[1]['device']['driver_version'] = 'changed'
        with self.assertRaisesRegex(ValueError, 'Incomparable runs: device'):
            gpu.compare_runs(runs)
