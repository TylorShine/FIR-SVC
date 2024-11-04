# FIR-SVC [WIP]

Open source framework for Singing Voice Conversion


# Quic start
```sh
# Download dependent files
> dl-models.bat
# Open train/infer shell
> launch.bat
#...edit `configs/fir-san.yaml` config files as needed
# Sortup train/test data
> python sortup.py -c configs/fir-san.yaml
# Preprocessing files into train/test splits based on configs
> python preprocess.py -c configs/fir-san.yaml
# Train model
> python train.py -c configs/fir-san.yaml
```


# License
[MIT License](LICENSE)
