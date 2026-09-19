'''Tests for the small modules of lpu_nn.modeling

lpu_nn.modeling の小さなモジュールのテスト。

これらは形状を変換する部品であり、形状の不変量 (系列長が保たれる、
語彙軸が置き換わる等) が崩れると、その場ではなく遠くの層で例外になる。
ここではその不変量と、デバイス / dtype の扱いを固定する。
'''

import math

import pytest
import torch
from torch import nn

from lpu_nn import modeling
from lpu_nn.modeling import activation
from lpu_nn.modeling.convolution import SequenceConvolution1d
from lpu_nn.modeling.linear import Linear

ACTIVATION_NAMES = ['gelu', 'mish', 'sigmoid', 'swish', 'swish1', 'relu', 'tanh']


class TestActivationFunctions:
    def test_gelu_matches_the_tanh_approximation_of_torch(self):
        x = torch.linspace(-5, 5, 101)
        expected = nn.functional.gelu(x, approximate='tanh')
        assert torch.allclose(activation.gelu(x), expected, atol=1e-5)

    def test_swish_with_beta_one_is_silu(self):
        x = torch.linspace(-5, 5, 101)
        assert torch.allclose(activation.swish(x, 1.0), nn.functional.silu(x), atol=1e-6)

    def test_mish_matches_its_definition(self):
        x = torch.linspace(-5, 5, 101)
        expected = x * torch.tanh(nn.functional.softplus(x))
        assert torch.allclose(activation.mish(x), expected, atol=1e-5)

    def test_mish_survives_an_input_that_overflows_the_exponential(self):
        # exp(x) が inf になっても、tanh(inf) = 1 なので x 自身に収束すること
        x = torch.tensor([100.0, 1000.0])
        assert torch.allclose(activation.mish(x), x)

    @pytest.mark.parametrize('func', [activation.gelu, activation.mish])
    def test_inplace_returns_the_same_tensor(self, func):
        x = torch.randn(4)
        assert func(x, inplace=True) is x


class TestActivationModules:
    @pytest.mark.parametrize('name', ACTIVATION_NAMES)
    def test_every_name_builds_a_working_module(self, name):
        # swish1 は super() に別のクラスを渡しており、必ず TypeError で
        # 落ちていた
        module = activation.get_activator(name, 4)
        out = module(torch.randn(2, 3, 4))
        assert out.shape == (2, 3, 4)

    @pytest.mark.parametrize('name', ACTIVATION_NAMES)
    def test_every_name_has_a_gain(self, name):
        assert activation.get_gain(name) > 0

    def test_gain_of_none_is_one(self):
        assert activation.get_gain('none') == 1.0

    def test_relu_and_tanh_gains_match_the_usual_values(self):
        assert activation.get_gain('relu') == pytest.approx(math.sqrt(2))
        assert activation.get_gain('tanh') == pytest.approx(5.0 / 3)

    def test_an_unknown_activator_is_rejected(self):
        with pytest.raises(ValueError):
            activation.get_activator('unknown', 4)

    def test_an_unknown_gain_is_rejected(self):
        with pytest.raises(ValueError):
            activation.get_gain('unknown')

    def test_swish_learns_one_beta_per_feature(self):
        module = activation.Swish(4)
        assert module.beta.shape == (4,)
        assert module.beta.requires_grad

    def test_swish_accepts_a_shape(self):
        assert activation.Swish([2, 4]).beta.shape == (2, 4)


class TestLinear:
    def test_replaces_the_last_axis(self):
        out = Linear(8, 4)(torch.randn(2, 5, 8))
        assert out.shape == (2, 5, 4)

    def test_applies_the_activation(self):
        # relu を通せば負の値は残らないこと
        layer = Linear(8, 4, activation='relu')
        assert (layer(torch.randn(2, 5, 8)) >= 0).all()

    def test_no_activation_leaves_the_output_alone(self):
        torch.manual_seed(0)
        layer = Linear(8, 4, activation='none')
        x = torch.randn(2, 8)
        assert torch.allclose(layer(x), nn.functional.linear(x, layer.weight, layer.bias))

    @pytest.mark.parametrize('initializer', ['orthogonal', 'he-normal', 'pytorch', None])
    def test_every_initializer_runs_and_zeroes_the_bias(self, initializer):
        layer = Linear(8, 4, activation='relu', initializer=initializer)
        layer.init_weights()
        assert layer.weight.shape == (4, 8)
        if initializer in ('orthogonal', 'he-normal'):
            assert torch.equal(layer.bias, torch.zeros(4))

    def test_to_records_the_device(self):
        layer = Linear(8, 4).to('cpu')
        assert layer.device == torch.device('cpu')


class TestSequenceConvolution:
    @pytest.mark.parametrize('ngram_order', [1, 2, 3, 4, 5])
    def test_the_sequence_length_is_preserved(self, ngram_order):
        # 畳み込みの前後で系列長が変わると、後段のマスクと形が合わなくなる
        conv = SequenceConvolution1d(8, 4, ngram_order)
        out = conv(torch.randn(2, 7, 8))
        assert out.shape == (2, 7, 4)

    @pytest.mark.parametrize('ngram_order', [2, 3])
    def test_it_works_before_to_has_been_called(self, ngram_order):
        # パディングを self.device から作っていたため、構築直後のモジュール
        # では AttributeError になっていた
        conv = SequenceConvolution1d(8, 4, ngram_order)
        assert conv(torch.randn(2, 5, 8)).shape == (2, 5, 4)

    def test_the_padding_follows_the_input_dtype(self):
        # float32 固定のパディングは float16 の系列と連結できなかった
        conv = SequenceConvolution1d(8, 4, 3).half()
        out = conv(torch.randn(2, 5, 8).half())
        assert out.dtype == torch.float16

    @pytest.mark.parametrize('initializer', ['orthogonal', 'he-normal', None])
    def test_every_initializer_runs(self, initializer):
        conv = SequenceConvolution1d(8, 4, 3, activation='relu', initializer=initializer)
        conv.init_weights()
        assert conv.weight.shape == (4, 8, 3)

    def test_applies_the_activation(self):
        conv = SequenceConvolution1d(8, 4, 3, activation='relu')
        assert (conv(torch.randn(2, 5, 8)) >= 0).all()


class TestModule:
    def test_to_records_the_device_and_dtype(self):
        module = modeling.Module()
        module.to(torch.device('cpu'), torch.float64)
        assert module.device == torch.device('cpu')
        assert module.dtype == torch.float64

    def test_to_returns_the_module_itself(self):
        module = modeling.Module()
        assert module.to('cpu') is module

    def test_to_reaches_the_nested_modules(self):
        # nn.Module.to() は属性まで伝えないため、入れ子にも適用すること
        outer = modeling.Module()
        outer.inner = modeling.Module()
        outer.to(torch.device('cpu'), torch.float64)
        assert outer.inner.dtype == torch.float64

    def test_dropout_defaults_to_the_module_ratio(self):
        module = modeling.Module()
        module.dropout_ratio = 0.0
        x = torch.randn(100)
        # 比率 0 なら値は変わらないこと (訓練モードでも)
        module.train()
        assert torch.equal(module.dropout(x), x)

    def test_dropout_is_a_no_op_in_evaluation_mode(self):
        module = modeling.Module()
        module.eval()
        x = torch.randn(100)
        assert torch.equal(module.dropout(x, 0.9), x)


class TestModuleArray:
    def test_holds_independent_copies(self):
        array = modeling.ModuleArray(Linear(4, 4), 3)
        assert len(array) == 3
        # deepcopy されているため、重みは共有されないこと
        assert array[0].weight is not array[1].weight

    def test_repr_collapses_the_repeated_children(self):
        text = repr(modeling.ModuleArray(Linear(4, 4), 3))
        assert '3 x [' in text

    def test_to_returns_the_array_itself(self):
        array = modeling.ModuleArray(Linear(4, 4), 2)
        assert array.to('cpu') is array


class TestModuleList:
    def test_repr_collapses_identical_children(self):
        # 同じ内容の子が続く場合は '(...)' にまとめること
        text = repr(modeling.ModuleList([nn.Identity(), nn.Identity()]))
        assert '(...)' in text

    def test_to_returns_the_list_itself(self):
        module_list = modeling.ModuleList([Linear(4, 4)])
        assert module_list.to('cpu') is module_list
