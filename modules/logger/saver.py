'''
author: wayn391@mastertones

modified by: TylorShine
'''

import os
import glob
import json
import shutil
import time
import yaml
import datetime
import torch
import matplotlib.pyplot as plt
from ..common import to_json
from torch.utils.tensorboard import SummaryWriter

from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration

try:
    from huggingface_hub import HfApi
except:
    pass


class Saver(object):
    def __init__(
            self, 
            args,
            initial_global_step=-1,
            initial_global_epoch=-1):

        self.expdir = args.env.expdir
        self.pretrained = args.env.pretrained
        self.pretrained_d = args.env.pretrained_d
        self.sample_rate = args.data.sampling_rate
        self.args = args
        
        self.hf = args.hf
        
        # cold start
        self.global_step = initial_global_step
        self.global_epoch = initial_global_epoch
        self.init_time = time.time()
        self.last_time = time.time()

        # makedirs
        os.makedirs(self.expdir, exist_ok=True)

        # accelerator
        if args.train.get('accelerator_project_config'):
            project_config = ProjectConfiguration(
                project_dir=args.env.expdir,
                logging_dir=os.path.join(args.env.expdir, 'logs'),
                **args.train.get('accelerator_project_config')
            )
        else:
            project_config = ProjectConfiguration(
                project_dir=args.env.expdir,
                logging_dir=os.path.join(args.env.expdir, 'logs')
            )
        self.accelerator = Accelerator(
            project_config=project_config,
            **args.train.accelerator,
        )
        
        # save config
        path_config = os.path.join(self.expdir, 'config.yaml')
        with open(path_config, "w") as out_config:
            yaml.dump(dict(args), out_config)
            
        # create repo
        if self.hf is not None and self.hf.get("push_to_hub") == True and self.hf.get("repo"):
            self.hf_api = HfApi()
            self.hf_api.create_repo(
                repo_type="model",
                exist_ok=True,
                **self.hf.get("repo"),
            )
            self.hf_future = None
        else:
            self.hf_future = None
            self.hf_api = None


    def log_info(self, msg, end="\n"):
        '''log method'''
        if isinstance(msg, dict):
            msg_list = []
            for k, v in msg.items():
                if isinstance(v, dict):
                    msg_list.append(f'{k}:')
                    for kk, vv in v.items():
                        if isinstance(v, int):
                            msg_list.append(f' {kk}: {vv:,}')
                        else:
                            msg_list.append(f' {kk}: {vv}')
                else:
                    if isinstance(v, int):
                        msg_list.append(f'{k}: {v:,}')
                    else:
                        msg_list.append(f'{k}: {v}')
            msg_str = '\n'.join(msg_list)
        else:
            msg_str = msg
        
        # display
        self.accelerator.print(msg_str, end=end)
        

    def log_value(self, dict):
        self.accelerator.log(
            dict,
            step=self.global_step,
        )
    
    def log_spec(self, name, spec, spec_out, vmin=-14, vmax=3.5):  
        spec_cat = torch.cat([(spec_out - spec).abs() + vmin, spec, spec_out], -1)
        spec = spec_cat[0]
        if isinstance(spec, torch.Tensor):
            spec = spec.cpu().numpy()
        fig = plt.figure(figsize=(12, 9))
        plt.pcolor(spec.T, vmin=vmin, vmax=vmax)
        plt.tight_layout()
        self.writer.add_figure(name, fig, self.global_step)
    
    def log_audio(self, dict):
        for k, v in dict.items():
            # self.writer.add_audio(k, v, global_step=self.global_step, sample_rate=self.sample_rate)
            if "tensorboard" in [t.name for t in self.accelerator.trackers]:
                tb_tracker = self.accelerator.get_tracker("tensorboard", unwrap=True)
                if tb_tracker:
                    for k, v in dict.items():
                        tb_tracker.add_audio(k, v, global_step=self.global_step, sample_rate=self.sample_rate)
    
    def get_interval_time(self, update=True):
        cur_time = time.time()
        time_interval = cur_time - self.last_time
        if update:
            self.last_time = cur_time
        return time_interval

    def get_total_time(self, to_str=True):
        total_time = time.time() - self.init_time
        if to_str:
            total_time = str(datetime.timedelta(
                seconds=total_time))[:-5]
        return total_time
    
    def set_global_step(self, step):
        self.global_step = step
        
    def load_state(self):
        # path
        path_states = glob.glob(os.path.join(
            self.expdir, 'states', 'cp*'))
        if len(path_states) > 0:
            steps = [os.path.basename(s).lstrip("cp") for s in path_states]
            maxstep = max([int(s) if s.isdigit() else -1 for s in steps])
            
            if maxstep >= 0:
                path_state = os.path.join(
                    self.expdir , 'states', f'cp{maxstep}')

                # load states
                self.accelerator.load_state(path_state, strict=False)
                self.accelerator.step = maxstep
                self.set_global_step(maxstep)

                # log
                self.accelerator.print(f' [*] State loaded from {path_state}')
                
                return True
        
        return False
            
    def load_from_pretrained(self,
                             model,
                             optimizer,
                             model_d,
                             optimizer_d):
        ret = False
        if os.path.isfile(self.pretrained):
            # # load states
            # self.accelerator.load_state(path_pt)                
            ckpt = torch.load(self.pretrained, map_location=self.accelerator.device, weights_only=True)
            
            unwrapped_model = self.accelerator.unwrap_model(model)
            # load model
            unwrapped_model.load_state_dict(ckpt if not "model" in ckpt else ckpt["model"], strict=False)
            
            if ckpt.get('optimizer') != None:
                unwrapped_optimizer = self.accelerator.unwrap_model(optimizer)
                # load optimizer
                unwrapped_optimizer.load_state_dict(ckpt['optimizer'])

            # log
            self.accelerator.print(f' [*] Model loaded from {self.pretrained}')
            ret = True
            
        if os.path.isfile(self.pretrained_d):
            # load states
            ckpt_d = torch.load(self.pretrained_d, map_location=self.accelerator.device, weights_only=True)

            unwrapped_model_d = self.accelerator.unwrap_model(model_d)
            # load model
            unwrapped_model_d.load_state_dict(ckpt_d if not "model" in ckpt_d else ckpt_d["model"], strict=False)

            if ckpt_d.get('optimizer') != None:
                unwrapped_optimizer_d = self.accelerator.unwrap_model(optimizer_d)
                # load optimizer
                unwrapped_optimizer_d.load_state_dict(ckpt_d['optimizer'])

            # log
            self.accelerator.print(f' [*] Discriminator model loaded from {self.pretrained_d}')
            ret = True
            
        return ret
        
    
    def save_state(self):
        # path
        path_state = os.path.join(
            self.expdir , 'states', f'cp{self.global_step}')

        # save states
        # note: temporarly disable safe serialization for saving state
        # maybe related: https://github.com/huggingface/safetensors/issues/450
        self.accelerator.save_state(path_state, safe_serialization=False)
        
        if self.hf is not None and self.hf.get("push_to_hub") == True and self.hf_api is not None:
            if self.hf_future is not None:
                try:
                    if self.hf_future.done():
                        self.hf_future.result()
                    else:
                        self.accelerator.print(f"Model upload still in progress, skipping this save.")
                        return
                except Exception as e:
                    self.accelerator.print(f"Failed to upload model to hub: {e}\nskipping upload.")
                    pass
            self.hf_future = self.hf_api.upload_folder(
                **self.hf.get("upload"),
                repo_id=self.hf.repo.get("repo_id"),
                folder_path=self.expdir,
                run_as_future=True,
            )
        
        # log
        self.accelerator.print(f' [*] State saved at {path_state}')

    def save_model(
            self,
            model, 
            # optimizer,
            name='model',
            postfix='_',
            as_json=False,
            states=None):
        # path
        if postfix != '_':
            postfix = postfix
        
        path_pt = os.path.join(
            self.expdir , name+postfix)
        
        # # save states
        # self.accelerator.save_state(path_pt)
        
        # save model
        self.accelerator.save_model(model, path_pt, safe_serialization=False)
        
        # save config
        path_config = os.path.join(self.expdir, name+postfix, 'config.yaml')
        with open(path_config, "w") as out_config:
            yaml.dump(dict(self.args), out_config)
            
        # copy spk_info
        path_spk = os.path.join(self.expdir, 'spk_info.npz')
        shutil.copy2(path_spk, os.path.join(self.expdir, name+postfix, 'spk_info.npz'))
        
        # # check
        # print(' [*] model checkpoint saved: {}'.format(path_pt))
        self.accelerator.print(f' [*] Model saved at {path_pt}')
            
        # to json
        if as_json:
            path_json = os.path.join(
                self.expdir , name+'.json')
            to_json(path_pt, path_json)
    
    def delete_model(self, name='model', postfix='_'):
        # path
        if postfix != '_':
            postfix = postfix
        path_pt = os.path.join(
            self.expdir , name+postfix+'.pt')
       
        # delete
        if os.path.exists(path_pt):
            os.remove(path_pt)
            print(' [*] model checkpoint deleted: {}'.format(path_pt))
        
    def global_step_increment(self):
        self.global_step += 1
        
    def global_epoch_increment(self):
        self.global_epoch += 1
        
    def sync(self):
        if self.hf_future is not None and not self.hf_future.done():
            self.hf_future.result()


