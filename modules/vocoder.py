import os

import onnxruntime
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.nn.utils.parametrizations import weight_norm

from modules.common import DotDict, complex_mul_in_real_3d

from modules.convnext_v2_like import ConvNeXtV2GLULikeEncoder, ConvNeXtV2LikeEncoder, LayerNorm1d


def load_model(
        model_path,
        device='cpu'):
    config_file = os.path.join(os.path.split(model_path)[0], 'config.yaml')
    print('Loading config file from: ' + config_file)
    with open(config_file, "r") as config:
        args = yaml.safe_load(config)
    args = DotDict(args)
    
    # load model
    model = None
    if args.model.type == 'FirNeXtV2':
        model = FirNeXtV2(
            sampling_rate=args.data.sampling_rate,
            block_size=args.data.block_size,
            win_length=args.model.win_length,
            n_unit=args.data.encoder_out_channels,
            n_hidden_channels=args.model.units_hidden_channels,
            n_spk=args.model.n_spk,
            n_layers=args.model.n_layers,
            n_dims=args.model.n_dims,
            n_kernel_sizes=args.model.n_kernel_sizes,
            use_speaker_embed=args.model.use_speaker_embed,
            use_embed_conv=not args.model.no_use_embed_conv,
            spk_embed_channels=args.data.spk_embed_channels,
            f0_input_variance=args.model.f0_input_variance,
            f0_offset_size_downsamples=args.model.f0_offset_size_downsamples,
            noise_env_size_downsamples=args.model.noise_env_size_downsamples,
            harmonic_env_size_downsamples=args.model.harmonic_env_size_downsamples,
            use_harmonic_env=args.model.use_harmonic_env,
            use_noise_env=args.model.use_noise_env,
            noise_to_harmonic_phase=args.model.noise_to_harmonic_phase,
            add_noise=args.model.add_noise,
            use_f0_offset=args.model.use_f0_offset,
            use_short_filter=args.model.use_short_filter,
            use_noise_short_filter=args.model.use_noise_short_filter,
            use_pitch_aug=args.model.use_pitch_aug,
            nsf_hifigan_in=args.model.nsf_hifigan.num_mels,
            nsf_hifigan_h=args.model.nsf_hifigan,
            noise_seed=args.model.noise_seed,
            )
            
    else:
        raise ValueError(f" [x] Unknown Model: {args.model.type}")
    
    print(' [Loading] ' + model_path)
    ckpt = torch.load(model_path, map_location=torch.device(device), weights_only=True)
    model.to(device)
    if 'model' in ckpt:
        ckpt = ckpt['model']
    else:
        ckpt = ckpt
    model.load_state_dict(ckpt, strict=False)
    model.eval()
    
    if args.model.use_speaker_embed:
        spk_info_path = os.path.join(os.path.split(model_path)[0], 'spk_info.npz')
        if os.path.isfile(spk_info_path):
            spk_info = np.load(spk_info_path, allow_pickle=True)
        else:
            print(' [Warning] spk_info.npz not found but model seems to setup with speaker embed')
            spk_info = None
    else:
        spk_info = None
    
    return model, args, spk_info


def load_onnx_model(
            model_path,
            providers=['CPUExecutionProvider'],
            device='cpu'):
    config_file = os.path.join(os.path.split(model_path)[0], 'config.yaml')
    with open(config_file, "r") as config:
        args = yaml.safe_load(config)
    args = DotDict(args)
    
    sess = onnxruntime.InferenceSession(
        model_path,
        providers=providers)
    
    # load model
    model = sess
    
    if args.model.use_speaker_embed:
        spk_info_path = os.path.join(os.path.split(model_path)[0], 'spk_info.npz')
        if os.path.isfile(spk_info_path):
            spk_info = np.load(spk_info_path, allow_pickle=True)
        else:
            print(' [Warning] spk_info.npz not found but model seems to setup with speaker embed')
            spk_info = None
    else:
        spk_info = None
    
    return model, args, spk_info


class CombSubMinimumNoisedPhaseStackOnly(torch.nn.Module):
    def __init__(self, 
            n_unit=256,
            n_hidden_channels=256):
        super().__init__()

        print(' [DDSP Model] Minimum-Phase harmonic Source Combtooth Subtractive Synthesiser (u2c stack only)')
        
        from .unit2control import Unit2ControlStackOnly
        self.unit2ctrl = Unit2ControlStackOnly(n_unit,
                                                n_hidden_channels=n_hidden_channels,
                                                conv_stack_middle_size=32)
        
    def forward(self, units_frames, **kwargs):
        '''
            units_frames: B x n_frames x n_unit
            f0_frames: B x n_frames x 1
            volume_frames: B x n_frames x 1 
            spk_id: B x 1
        '''
        
        return self.unit2ctrl(units_frames)
    
    
class FirNeXtV2(torch.nn.Module):
    def __init__(self, 
            sampling_rate,
            block_size,
            win_length,
            n_unit=256,
            n_hidden_channels=256,
            n_spk=1,
            n_layers=[3],
            n_dims=[256],
            n_kernel_sizes=[7],
            use_speaker_embed=True,
            use_embed_conv=True,
            spk_embed_channels=256,
            f0_input_variance=0.1,
            f0_offset_size_downsamples=1,
            noise_env_size_downsamples=4,
            harmonic_env_size_downsamples=4,
            use_harmonic_env=False,
            use_noise_env=True,
            use_add_noise_env=True,
            noise_to_harmonic_phase=False,
            add_noise=False,
            use_phase_offset=False,
            use_f0_offset=False,
            use_short_filter=False,
            use_noise_short_filter=False,
            use_pitch_aug=False,
            nsf_hifigan_in=128,
            nsf_hifigan_h=None,
            noise_seed=289,
            onnx_unit2ctrl=None,
            export_onnx=False,
            device=None):
        super().__init__()

        print(' [Model] ConvNeXtV2GLU Block FIR Filter Synthesiser')
        
        # device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # params
        self.register_buffer("sampling_rate", torch.tensor(sampling_rate))
        self.block_size = block_size
        self.win_length = win_length
        self.register_buffer("window", torch.hann_window(win_length))
        self.noise_env_size_downsamples = noise_env_size_downsamples
        self.register_buffer("harmonic_env_size_downsamples", torch.tensor(harmonic_env_size_downsamples))
        self.register_buffer("use_harmonic_env", torch.tensor(use_harmonic_env))
        self.register_buffer("use_noise_env", torch.tensor(use_noise_env))
        self.register_buffer("use_speaker_embed", torch.tensor(use_speaker_embed))
        self.register_buffer("use_embed_conv", torch.tensor(use_embed_conv))
        self.register_buffer("noise_seed", torch.tensor(noise_seed))
        
        self.pred_filter_size = win_length // 2 + 1
        
        self.gen_upsample_rates = nsf_hifigan_h.upsample_rates
        
        
        #Unit2Control
        split_map = {
            'harmonic_filt_kernel': win_length,
            'harmonic_filt_kernel_short': win_length//4,    #TODO: parametrize?
            'noise_filt_kernel': win_length//4,
        }
        
        if use_noise_env:
            split_map['noise_envelope_magnitude'] = block_size//noise_env_size_downsamples
        if use_harmonic_env:
            split_map['harmonic_envelope_magnitude'] = block_size//harmonic_env_size_downsamples
            
        self.use_short_filter = use_short_filter
        self.use_noise_short_filter = use_noise_short_filter
            
        
        if onnx_unit2ctrl is not None:
            from .unit2control import Unit2ControlGE2E_onnx
            self.unit2ctrl = Unit2ControlGE2E_onnx(onnx_unit2ctrl, split_map)
        elif export_onnx:
            from .unit2control import Unit2ControlGE2E_export
            self.unit2ctrl = Unit2ControlGE2E_export(n_unit, spk_embed_channels, split_map,
                                            n_hidden_channels=n_hidden_channels,
                                            n_spk=n_spk,
                                            use_pitch_aug=use_pitch_aug,
                                            use_spk_embed=use_speaker_embed,
                                            use_embed_conv=use_embed_conv,
                                            embed_conv_channels=64, conv_stack_middle_size=32)
        else:
            from .unit2control import Unit2ControlGE2ESignal8
            
            self.unit2ctrl = Unit2ControlGE2ESignal8(n_unit, spk_embed_channels, split_map,
                                            block_size=block_size,
                                            n_hidden_channels=n_hidden_channels,
                                            n_layers=n_layers,
                                            n_dims=n_dims,
                                            n_kernel_sizes=n_kernel_sizes,
                                            n_spk=n_spk,
                                            use_pitch_aug=use_pitch_aug,
                                            use_spk_embed=use_speaker_embed,
                                            use_embed_conv=use_embed_conv,
                                            embed_conv_channels=64, conv_stack_middle_size=32)
        
        # generate static noise
        self.gen = torch.Generator()
        self.gen.manual_seed(noise_seed)
        static_noise_t = (torch.rand([
            win_length*127  # about 5.9sec when sampling_rate=44100 and win_length=2048
        ], generator=self.gen)*2.0-1.0)
        
        self.register_buffer('static_noise_t', static_noise_t)

        if use_noise_env:
            self.register_buffer('noise_env_arange', torch.arange(self.noise_env_size_downsamples))
            
        # self.generator.apply(self.init_weights)
        
        
    @staticmethod
    def init_weights(m, mean=0.0, std=0.02):
        classname = m.__class__.__name__
        if classname.find("Conv") != -1:
            m.weight.data.kaiming_normal_(a=0, mode='fan_out', nonlinearity='relu')
        elif classname.find("Linear") != -1:
            m.weight.data.normal_(mean, std)
            
        if hasattr(m, "bias") and m.bias is not None:
            m.bias.data.zero_()
        

    def forward(self, units_frames, f0_frames, volume_frames,
                spk_id=None, spk_mix=None, aug_shift=None, initial_phase=None, infer=True, **kwargs):
        '''
            units_frames: B x n_frames x n_unit
            f0_frames: B x n_frames x 1
            volume_frames: B x n_frames x 1 
            spk_id: B x 1
        '''
        
        # f0 = upsample(f0_frames, self.block_size)
        # we just enough simply repeat to that because block size is short enough, it is sufficient I think.
        f0 = f0_frames.repeat(1,1,self.block_size).flatten(1).unsqueeze(-1)
        
        if infer:
            # NOTE: this should cast to double for precision for when long frames generation
            # TODO: cumsum is not numerically stable for long sequences (makes big number), may need to switch to calc from f0 and sampling rate directly
            x = torch.cumsum(f0.double() / self.sampling_rate, axis=1)
        else:
            x = torch.cumsum(f0 / self.sampling_rate, axis=1)
        
        x = torch.frac(x) # constrain x between 0 and 1
        x = x.to(f0)
        
        phase_frames = 2. * 3.141592653589793 * x.reshape(x.shape[0], -1, self.block_size)[:, :, 0:1]
        
        # sine source
        # sinewave = torch.sin(2. * 3.141592653589793 * x).reshape(x.shape[0], -1, self.block_size)
        
        # band limited harmonic source
        pix = (torch.pi*(x.flatten(1)))
        
        # n_harmonics = 7.
        # a = torch.round(torch.min(self.sampling_rate/f0*0.5, torch.full_like(f0, n_harmonics))).flatten(1)*2. + 1.
        a = torch.round(self.sampling_rate/torch.clamp(f0, min=20.)*0.5).flatten(1)*2. + 1.
        sinpix = torch.sin(pix)
        harmonic_source = torch.where(pix < 1e-8, 1., torch.sin(a*pix) / (a*sinpix)).reshape(x.shape[0], -1, self.block_size)
        
        noise = (self.static_noise_t*.3162).unsqueeze(0).repeat(harmonic_source.shape[0], (harmonic_source.shape[1]*harmonic_source.shape[2])//self.static_noise_t.shape[0] + 1)[:, :harmonic_source.shape[1]*harmonic_source.shape[2]]
        noise = noise.view_as(harmonic_source)
        
        # parameter prediction
        ctrls, hidden = self.unit2ctrl(units_frames, f0_frames, phase_frames, volume_frames,
                                       spk_id=spk_id, spk_mix=spk_mix, aug_shift=aug_shift)
        
        
        harmonic_source_short_t = F.pad(harmonic_source, (self.win_length//4, self.win_length//4-1)).unsqueeze(1)
        
        bi, bj, c, d = harmonic_source_short_t.shape
        harmonic_source_short_t = harmonic_source_short_t.transpose(1, 0).view(bj, bi * c, d)
        harmonic_filt_kernel_short_v = ctrls['harmonic_filt_kernel_short'].reshape(bi, -1, ctrls['harmonic_filt_kernel_short'].shape[2])
        harmonic_filt_kernel_short = harmonic_filt_kernel_short_v.view(bi*harmonic_filt_kernel_short_v.shape[1], 1, -1)
        harmonic_source_short = F.conv1d(harmonic_source_short_t, harmonic_filt_kernel_short, groups=bi*harmonic_filt_kernel_short_v.shape[1]).view(bj, bi, harmonic_filt_kernel_short_v.shape[1], -1).squeeze(0)
        
        output_size = f0.shape[1] + self.win_length//4
        
        
        noise_source_t = noise.view(noise.shape[0], -1, self.block_size)
        noise_source_t = F.pad(noise_source_t, (self.win_length//4, self.win_length//4-1)).unsqueeze(1)
        bi, bj, c, d = noise_source_t.shape
        noise_source_t = noise_source_t.transpose(1, 0).view(bj, bi * c, d)
        noise_filt_kernel_v = ctrls['noise_filt_kernel'].reshape(bi, -1, self.win_length//4)
        noise_filt_kernel = noise_filt_kernel_v.view(bi*noise_filt_kernel_v.shape[1], 1, -1)
        noise = F.conv1d(noise_source_t, noise_filt_kernel, groups=bi*noise_filt_kernel_v.shape[1]).view(bj, bi, noise_filt_kernel_v.shape[1], -1).squeeze(0)
        output_size = f0.shape[1] + self.win_length//4
        noise = F.fold(
            (noise + harmonic_source_short).transpose(2, 1),
            output_size=(1, output_size),
            kernel_size=(1, self.block_size + self.win_length//4),
            stride=(1, self.block_size)).squeeze(1).squeeze(1)[:, :f0.shape[1]]
        
        harmonic_source_t = (noise).view(harmonic_source_short.shape[0], -1, self.block_size)
        harmonic_source_t = F.pad(harmonic_source_t, (self.win_length, self.win_length-1)).unsqueeze(1)
        bi, bj, c, d = harmonic_source_t.shape
        harmonic_source_t = harmonic_source_t.transpose(1, 0).view(bj, bi * c, d)
        harmonic_filt_kernel_v = ctrls['harmonic_filt_kernel'].reshape(bi, -1, ctrls['harmonic_filt_kernel'].shape[2])
        harmonic_filt_kernel = harmonic_filt_kernel_v.view(bi*harmonic_filt_kernel_v.shape[1], 1, -1)
        harmonic_source = F.conv1d(harmonic_source_t, harmonic_filt_kernel, groups=bi*harmonic_filt_kernel_v.shape[1]).view(bj, bi, harmonic_filt_kernel_v.shape[1], -1).squeeze(0)
        
        # print(f'harmonics shape: {harmonic_source.shape}, harmonic_filt_kernel: {harmonic_filt_kernel_v.shape}')
        # overlap and add
        output_size = f0.shape[1] + self.win_length
        harmonic_source = F.fold(
            harmonic_source.transpose(2, 1),
            output_size=(1, output_size),
            kernel_size=(1, self.block_size + ctrls['harmonic_filt_kernel'].shape[2]),
            stride=(1, self.block_size)).squeeze(1).squeeze(1)
        
        #TODO: return last samples for seamless concatenation in inference
        return torch.clip(harmonic_source[:, :f0.shape[1]], min=-1., max=1.)
