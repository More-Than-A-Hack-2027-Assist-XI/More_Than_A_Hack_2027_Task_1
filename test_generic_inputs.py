"""Regression tests for arbitrary EPTS filenames and input folders."""
import tempfile
import unittest
from pathlib import Path
from src.enhanced_classifier import EnhancedPossessionClassifier
from main import BallPossessionClassifier

XML = '<EPTS><Team id="h"/><Team id="a"/><Player id="100" teamId="h"/><Player id="200" teamId="a"/></EPTS>'
RAW = ('1:0.1,0.1,NaN;0.9,0.9,0:0.1,0.1,NaN\n'
       '2:0.1,0.1,0;0.9,0.9,0:0.9,0.9,0\n'
       '3:0.1,0.1,0;0.9,0.9,0:NaN,NaN,NaN\n')

class TestGenericInputs(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.folder = Path(self.tmp.name) / 'renamed input folder'
        self.folder.mkdir()

    def pair(self, name):
        raw = self.folder / (name + '_atd.txt')
        raw.write_text(RAW)
        raw.with_suffix('.xml').write_text(XML)
        return raw

    def verify(self, name):
        self.pair(name)
        c = EnhancedPossessionClassifier()
        frames, ball, players = c.load_tactical_data(self.folder)
        result = c.generate_submission(c.classify_frames(frames, ball, players))
        self.assertEqual(result.frame.tolist(), [0, 1, 2])
        self.assertEqual(result.label.tolist(), ['H', 'A', 'D'])
        self.assertEqual(list(result), ['frame', 'label'])

    def test_broadcast_filename(self):
        self.verify('BAR-ATH_broadcast')

    def test_deneme_filename(self):
        self.verify('BAR-ATH_denem')

    def test_arbitrary_prefix(self):
        self.verify('VAR-ATH_anything')

    def test_missing_xml(self):
        (self.folder / 'custom_atd.txt').write_text(RAW)
        with self.assertRaisesRegex(FileNotFoundError, 'custom_atd.xml'):
            EnhancedPossessionClassifier().load_tactical_data(self.folder)

    def test_no_candidates(self):
        with self.assertRaisesRegex(FileNotFoundError, 'atd'):
            EnhancedPossessionClassifier().load_tactical_data(self.folder)

    def test_multiple_pairs(self):
        self.pair('one')
        self.pair('two')
        with self.assertRaisesRegex(ValueError, 'Ambiguous'):
            EnhancedPossessionClassifier().load_tactical_data(self.folder)

    def test_valid_pair_plus_orphan(self):
        self.pair('one')
        (self.folder / 'orphan_atd.txt').write_text(RAW)
        with self.assertRaisesRegex(ValueError, 'Ambiguous'):
            EnhancedPossessionClassifier().load_tactical_data(self.folder)

    def test_full_pipeline_uses_resolved_metadata(self):
        self.pair('BAR-ATH_broadcast')
        output = self.folder / 'output.csv'
        BallPossessionClassifier(device='cpu').run_pipeline(self.folder, output)
        self.assertEqual(output.read_text().splitlines(), ['frame,label', '0,H', '1,A', '2,D'])

if __name__ == '__main__':
    unittest.main()
