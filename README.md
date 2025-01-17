# MultiMAE Implementation for Face Anti-Spoofing

## How to use
- `train.py`: Define classification loss and LBP-guided loss here.
- `test.py`: Perform inference per **model_save_epoch** or **model_save_step**.
- `balanceloader.py`: Dataloader for training.
- `intradataloader.py`: Dataloader for testing.
- `multimae/multimae.py`: MultiMAE and MultiViT are defined here. The random masking strategy is defined in the forward method.
- `multimae/multimae_utils.py`: ViT block is defined here as class `Block`. The forward method is modified to freeze all parameters except MDA adapter and output adapter.
- `multimae/modality_disentangle_adapters.py`: Implement the MDA adapter here.
- `multimae/input_adapters.py`: Perform Image->Tokens here. Use `PatchedInputAdapter` for RGB, Depth, and IR.
- `multimae/output_adapters.py`: Use `LinearOutputAdapter` to utilize class token for classification.

One can run the following command to train the model:
```console
python train.py --train_dataset [dataset] --total_epoch [epoch] 
```

To test the model and log the results in APCER, BPCER, ACER, AUC:
```console
python test.py --train_dataset [dataset] --test_dataset [dataset] --missing [dataset/none]
```
## TODO
1. Download the DeiT-base pretrained weight:
    ```console
    wget https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth
    ```

2. Run `tools/vit2multimae_converter.py` to convert the above weight into MultiMAE format named `deit_b2multimae.pth`.
   
3. Modify the forward method in `multimae.py` to implement batch-level and sample-level random masking strategies.
   
4. Modify `modality_disentangle_adapters.py` to implement the MDA adapter.
   
5. Modify `train.py` to implement LBP-guided loss.
