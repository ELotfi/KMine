Code for the [paper](https://aclanthology.org/2021.nlp4convai-1.24/): **Teach Me What to Say and I Will Learn What to Pick: Unsupervised Knowledge 
Selection Through Response Generation with Pretrained Generative Models**

<img src="https://github.com/ELotfi/KMine/blob/main/figures/KM.png" width="700">

## Dataset
We use the pre-processed Wizard of Wikipedia (WoW) dataset provided by Zheng and Zhou (2019) which can be downloaded from [here](https://drive.google.com/drive/folders/1zS0xRy-UgQTafNhxGBGS4in6zmAMKlVM?usp=sharing).
Create a `data` folder in the root and save the files (6 files) there. On the first run, secondary files will be created and saved for faster future runs. 

## Running the code
* Before the main run, you can do a rapid sanity check to make sure everything works fine, by running: \
`torchrun  --nproc_per_node=4  main.py --do_sanity_check=True`
* Results can be recreated by: \
`torchrun  --nproc_per_node=4  main.py --num_candidates=60 --train_batch_size=1 --accumulate_grad=16`
* For a faster run (slightly lower R@1): \
`torchrun  --nproc_per_node=4  main.py --num_candidates=50 --train_batch_size=2 --accumulate_grad=8`
* To train on 1 GPU: \
`torchrun  --nproc_per_node=1  main.py --num_candidates=50 --train_batch_size=2 --accumulate_grad=32`

## Running arguments 
By default the code does unsupervised training for 8 epochs, does loss and R@1 evaluation on the dev set and saves the checkpoint after each epoch, and 
does prediction (with F1 log) on epochs 1,5,8 and saves the results (selected knowledge + generated response). The behavior can be changed using the 
following arguments:
* `--epochs_predict (default= [1,5,8])`: Epochs at which inference is done
* `--with_kn_loss (default=False)`: Whether to add supervision (on knowledge selection)
* `--ratn  (default= [1,5])`: Calculate Recall at these values
* `--single_valid (default=True)`: Whether to do validation on Dev-Seen and Dev-Unseen or treat them as a single set 
* `--num_candidates (default= 50)`: Number of knowledge candidates in the training and validation set (does not affect the test set)
* `--data_mode (default= 'all')`: Whether to include turns which do not use knowledge ('all') or remove them ('only_with_kn') in creating datasets. The
latter corresponds to the `W/Kn` version.
* `--max_function (default= 'softmax')`: Function to use to normalize candidate probabilities. Options are `softmax` or `entmax` for which you need to
install via `pip install entmax`
* `generation parameters`: Responses are generated with greedy decoding. This can be changed using `--temperature`, `--top_k`, `--top_p` and `--do_sample`.

## Citation
If used, please cite as:
```
@inproceedings{lotfi-etal-2021-teach,
    title = "Teach Me What to Say and {I} Will Learn What to Pick: Unsupervised Knowledge Selection Through Response Generation with Pretrained Generative Models",
    author = "Lotfi, Ehsan  and
      De Bruyn, Maxime  and
      Buhmann, Jeska  and
      Daelemans, Walter",
    booktitle = "Proceedings of the 3rd Workshop on Natural Language Processing for Conversational AI",
    month = nov,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.nlp4convai-1.24",
    doi = "10.18653/v1/2021.nlp4convai-1.24",
    pages = "254--262",
}
```
