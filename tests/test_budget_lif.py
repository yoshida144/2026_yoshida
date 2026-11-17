import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import torch
import torch.nn.functional as F
from snn import BudgetLIFLayer, event_linear, select_event_mask
from model import ML_CSC_Reconstruction, ML_CSC_Image_Classification
from utils import set_execution


class BudgetTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(43)
        torch.set_num_threads(1)

    def layer(self, budget=1000, stride=1):
        layer = BudgetLIFLayer(1, 3, 3, stride, 4).double()
        layer.configure_lif(dict(theta=0.15, beta=0.6, delta=0.1, event_budget=budget))
        with torch.no_grad():
            layer.Wz.normal_(0, 0.03)
        return layer

    def test_selection_per_image_signs_and_ties(self):
        candidate = torch.tensor([[[[1., -1., 1., 0.]]], [[[1., 1., 0., 0.]]]])
        mask = select_event_mask(candidate, candidate.abs(), torch.tensor([2, 0]))
        self.assertTrue(torch.equal(candidate*mask, torch.tensor([[[[1., -1., 0., 0.]]], [[[0., 0., 0., 0.]]]])))

    def test_event_convolution_boundaries(self):
        for stride in (1, 2):
            x = torch.randn(2, 3, 5, 6, dtype=torch.double)
            x[x.abs() < 1] = 0
            w = torch.randn(4, 3, 3, 3, dtype=torch.double)
            expected = F.conv2d(x, w, stride=stride, padding=1)
            got, count = event_linear(x, w, expected.shape[-2:], stride=stride, padding=1)
            torch.testing.assert_close(got, expected)
            self.assertGreater(count, 0)

    def test_dense_event_budgets_shapes_and_reset(self):
        for budget in (0, 2, 1000):
            for stride in (1, 2):
                layer = self.layer(budget, stride).eval()
                x = torch.randn(2, 1, 7, 8, dtype=torch.double)
                x[0] = 0
                with torch.no_grad():
                    dense = layer(x)
                    stats = layer.last_stats.copy()
                    layer.execution_mode = 'event'
                    event = layer(x)
                    again = layer(x)
                torch.testing.assert_close(event[0], dense[0], rtol=0, atol=0)
                torch.testing.assert_close(event[1], dense[1])
                torch.testing.assert_close(event[0], again[0])
                self.assertEqual(event[1].shape, x.shape)
                self.assertTrue(all(n <= budget for n in layer.last_stats['spikes_per_image']))
                self.assertEqual(layer.last_stats['spikes_per_image'][0], 0)
                self.assertEqual(layer.last_stats['spikes'], stats['spikes'])

    def test_bptt_and_optimizers(self):
        specs = [dict(dim=3, kernel=3, stride=1, T=4, model='lif_budget', prox='no_cons',
                      lif=dict(theta=0.15, event_budget=1000))]
        for classification in (False, True):
            model = (ML_CSC_Image_Classification(1, 2, specs, 'none') if classification
                     else ML_CSC_Reconstruction(1, specs, 'none')).double()
            optimizer = (torch.optim.Adam if classification else torch.optim.SGD)(model.parameters(), lr=0.01)
            x = torch.randn(2, 1, 6, 6, dtype=torch.double)
            before = {n: p.detach().clone() for n, p in model.named_parameters()}
            for _ in range(2):
                optimizer.zero_grad()
                out, codes, _ = model(x)
                loss = F.mse_loss(model.decode(0, x, codes), x)
                if classification:
                    loss = loss + F.cross_entropy(out, torch.tensor([0, 1]))
                loss.backward()
                for name, p in model.named_parameters():
                    self.assertIsNotNone(p.grad, name)
                    self.assertTrue(torch.isfinite(p.grad).all(), name)
                optimizer.step()
            for name, p in model.named_parameters():
                self.assertFalse(torch.equal(p, before[name]), name)
            model.eval()
            with torch.no_grad():
                expected = model(x)[0]
                set_execution(model, 'event')
                actual = model(x)[0]
            torch.testing.assert_close(actual, expected)

    def test_subthreshold_surrogate_gradient(self):
        layer = self.layer()
        x = torch.full((1, 1, 4, 4), 0.01, dtype=torch.double, requires_grad=True)
        z, _ = layer(x)
        self.assertEqual(int(z.count_nonzero()), 0)
        z.sum().backward()
        self.assertGreater(float(x.grad.abs().sum()), 0)

    def test_legacy_lista_and_old_lif(self):
        for mode in ('lista', 'lif'):
            specs = [dict(dim=2, kernel=3, stride=1, T=3, model=mode, prox='no_cons')]
            model = ML_CSC_Reconstruction(1, specs, 'none').eval()
            clone = ML_CSC_Reconstruction(1, specs, 'none').eval()
            clone.load_state_dict(model.state_dict(), strict=True)
            x = torch.randn(1, 1, 6, 6)
            with torch.no_grad():
                torch.testing.assert_close(clone(x)[0], model(x)[0])

    def test_event_requires_inference(self):
        layer = self.layer()
        layer.execution_mode = 'event'
        with self.assertRaises(RuntimeError):
            layer(torch.ones(1, 1, 4, 4, dtype=torch.double))


if __name__ == '__main__':
    unittest.main()
