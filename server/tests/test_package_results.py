"""Packaging must work while the launcher lock is held and preserve prior ZIPs."""
import ast
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from parallel_benchmark import FileLock


def load_package(root):
    source=Path(__file__).resolve().parents[1]/'Astar_JPS_SA_JPS_config_profiles_ready.py'
    node=next(n for n in ast.parse(source.read_text(encoding='utf-8')).body
              if isinstance(n,ast.FunctionDef) and n.name=='package_results')
    namespace={'Path':Path,'RESULTS_DIR':root,'dataset_output_path':lambda path:path}
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(source),'exec'),namespace)
    return namespace['package_results']


class PackagingTest(unittest.TestCase):
    def test_held_lock_skipped_results_preserved_and_no_self_inclusion(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            (root/'result.csv').write_text('runtime,turns\n1,2\n')
            cache=root/'cache';cache.mkdir()
            (cache/'checkpoint.sqlite3').write_bytes(b'checkpoint payload')
            locks=cache/'checkpoint.sqlite3.locks';locks.mkdir()
            (locks/'01.lock').write_bytes(b'lock')
            (root/'.experiment.lock').write_bytes(b'locked')
            lock=FileLock(root/'.experiment.lock')
            self.assertTrue(lock.acquire())
            try:
                package=load_package(root)
                archive=package(root,str(root/'results'))
                archive=package(root,str(root/'results'))
                with zipfile.ZipFile(archive) as z:
                    self.assertEqual(set(z.namelist()),{'result.csv','cache/checkpoint.sqlite3'})
                    self.assertEqual(z.read('cache/checkpoint.sqlite3'),b'checkpoint payload')
                    self.assertIsNone(z.testzip())
            finally:
                lock.release()

    def test_real_result_error_preserves_previous_archive(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);(root/'result.csv').write_text('saved data')
            archive=root/'results.zip';archive.write_bytes(b'previous archive')
            with patch.object(zipfile.ZipFile,'write',side_effect=PermissionError('locked result')):
                with self.assertRaisesRegex(PermissionError,'result.csv'):
                    load_package(root)(root,str(root/'results'))
            self.assertEqual(archive.read_bytes(),b'previous archive')
            self.assertEqual((root/'result.csv').read_text(),'saved data')
            self.assertFalse(list(root.glob('*.tmp')))


if __name__=='__main__':unittest.main()
