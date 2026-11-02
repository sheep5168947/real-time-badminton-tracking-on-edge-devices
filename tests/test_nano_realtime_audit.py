import ast
import importlib.util
import os
from pathlib import Path
import unittest

HERE = Path(__file__).resolve().parent
AUDITOR = HERE/'audit_nano.py'
if not AUDITOR.exists():
    AUDITOR = HERE.parent/'thesis/platforms/jetson_nano/evaluation/audit_realtime.py'
spec = importlib.util.spec_from_file_location('auditor_under_test', str(AUDITOR))
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


class RemoveObservations(ast.NodeTransformer):
    """Prove that removing added observations restores the original program AST."""
    def visit_Assign(self, node):
        for target in node.targets:
            names = [n.id for n in ast.walk(target) if isinstance(n, ast.Name)]
            if any(n.startswith('audit_') for n in names):
                return None
        return self.generic_visit(node)

    def visit_Expr(self, node):
        if any(isinstance(n, ast.Name) and n.id == 'audit_trace' for n in ast.walk(node)):
            return None
        return self.generic_visit(node)

    def visit_Dict(self, node):
        pairs = [(k,v) for k,v in zip(node.keys,node.values)
                 if not (isinstance(k,ast.Str) and k.s == '_audit')]
        node.keys = [k for k,v in pairs]
        node.values = [v for k,v in pairs]
        return self.generic_visit(node)


class AuditTests(unittest.TestCase):
    def source(self):
        default = HERE.parent/'thesis/platforms/jetson_nano/evaluation/benchmark_nano_all146_matches.py'
        p = Path(os.environ.get('THESIS_AUDIT_SOURCE', str(default)))
        return p.read_text(encoding='utf-8-sig')

    def test_instrumentation_preserves_original_ast(self):
        source = self.source()
        original = ast.parse(source)
        instrumented = ast.parse(audit.instrument(source))
        restored = RemoveObservations().visit(instrumented)
        self.assertEqual(ast.dump(original), ast.dump(restored))

    def test_gt_not_used_for_tracking_decisions(self):
        tree = ast.parse(self.source())
        run = next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='run_realtime_clip')
        before_scoring = []
        for n in run.body:
            if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='total_time' for t in n.targets):
                break
            before_scoring.append(n)
        reads = [n for stmt in before_scoring for n in ast.walk(stmt)
                 if isinstance(n,ast.Name) and n.id=='gt_dict' and isinstance(n.ctx,ast.Load)]
        self.assertEqual(reads, [])

    def test_detection_categories_and_wrong_location_recall(self):
        gt = {i:dict(vis=1 if i<3 else 0,x=0,y=0) for i in range(5)}
        c,_ = audit.metric_counts(gt, {0:(15,0),1:(16,0),3:(0,0)})
        self.assertEqual(c,dict(tp=1,tn=1,wrong_location=1,false_positive_absent=1,
                               missed_visible=1,unknown_visibility=0))
        m = audit.metrics(c)
        self.assertEqual(m['recall'],1/3)
        self.assertEqual(m['legacy_recall'],1/2)
        self.assertEqual(m['precision'],1/3)

    def test_no_detections_and_no_visible_targets(self):
        c,_ = audit.metric_counts({0:dict(vis=0,x=0,y=0)}, {})
        self.assertEqual(c['tn'],1)
        self.assertIsNone(audit.metrics(c)['recall'])
        self.assertIsNone(audit.metrics(c)['precision'])

    def test_unseen_visibility_is_explicit(self):
        c,_ = audit.metric_counts({0:dict(vis=2,x=0,y=0)}, {})
        self.assertEqual(c['unknown_visibility'],1)

    def test_changed_source_refuses_silent_instrumentation(self):
        with self.assertRaises(RuntimeError):
            audit.instrument(self.source().replace('    predictions = {}\n','    predictions = dict()\n'))


if __name__=='__main__':
    unittest.main()
