# Museformer with Noise
CS470 Team1 Code
Based on references below : 
1. Paper: [Museformer: Transformer with Fine- and Coarse-Grained Attention for Music Generation](https://arxiv.org/abs/2210.10349), by Botao Yu, Peiling Lu, Rui Wang, Wei Hu, Xu Tan, Wei Ye, Shikun Zhang, Tao Qin, Tie-Yan Liu, NeurIPS 2022
2. Github Repository: https://github.com/microsoft/muzic/tree/main 

The following content describes the steps to run Museformer with noise using Lakh MIDI dataset. 
All the commands are run at the root directory unless specified.


## 1. Dataset

### 1.1 Our code: MIDI preprocessor
We designed a MIDI preprocessor as Museformer didn’t contain codes for MIDI preprocessing. You can find the codes in the pre_clean directory. The code downloads the lmd-full dataset and preprocesses MIDI files following the description in the Museformer paper. The instruments are normalized to 6 basic ones: square synthesizer (80), piano (0), guitar (25), string (48), bass (43), and drum, wherein the parentheses are MIDI program IDs if applicable. After the preprocess, we could get 21,911 MIDI files.

The following guide shows how to use pre_clean.
1. Install the requirements.txt with the command below.
   ```bash
   pip install -r requirements.txt
   ```
2. Run data_preprocess.py with desired configuration variables COPY_MIDI and NUM_MIDI.
   ```bash
   python3 data_preprocess.py
   ```
Note that if COPY_MIDI is set to true, the pre_clean directory must be located at museformer/pre_clean. After preprocessing, put all the MIDI files in `data/midi` and meta data in 'data/meta'.


### 1.2. Turn into tokens
Install [MidiProcessor](https://github.com/btyu/MidiProcessor). Then, encode the MIDI files into tokens:

```bash
midi_dir=data/midi
token_dir=data/token
mp-batch-encoding $midi_dir $token_dir --encoding-method REMIGEN --normalize-pitch-value --remove-empty-bars --sort-insts id
```

where the arguments are explained as follows:
- `encoding-method`: the representation method. We use the one called REMIGEN.
- `normalize-pitch-value`: normalize pitches so as to make the song *C major* or *A minor*.
- `remove-empty-bars`: remove empty bars at the beginning or the end of a song.
- `sort-insts`: designate a method that sorts the instruments.

After encoding, you should see the token representation of each MIDI file in `output_dir`. 

Then, run the following command to gather the tokens for each split.

```bash
token_dir=data/token
split_dir=data/split
for split in train valid test : 
	do python tools/generate_token_data_by_file_list.py data/meta/${split}.txt $token_dir $split_dir ;
done
```

### 1.3. Make binary data
Next, use `fairseq-preprocess` to make binary data:

```bash
split_dir=data/split
data_bin_dir=data-bin/lmd6remi

mkdir -p data-bin

fairseq-preprocess \
  --only-source \
  --trainpref $split_dir/train.data \
  --validpref $split_dir/valid.data \
  --testpref $split_dir/test.data \
  --destdir $data_bin_dir \
  --srcdict data/meta/general_use_dict.txt
```
 Now, you should see the binary data in `data-bin/lmd6remi`.
 

##  2. Environment

We used hardware and software environment below for training and other all processes of Museformer.

- Hardware environment: NVIDIA TITAN RTX (24GB)

- Software environment: 

    OS : Linux
    ```
    CUDA:          11.6
    Python:        3.8.10
    fairseq:       0.10.2
    tensorboardX:  2.2
    triton: 2.0.0
    ```
    
    And install other python packages listed in 'requirements.txt' using command below.
    ```bash
    pip install -r requirements.txt
    ```

## 3. Train

Run the following command to train Museformer:

```bash
bash ttrain/mf-lmd6remi-1.sh
```

In our experiment, we run it on 4 GPUs, and the batch size is set to 1, so the real batch size is 4. Current implementation only supports batch size = 1. You may change `UPDATE_FREQ` to modify the real batch size.

By modifying `con2con` and `con2sum`, you can control the bars for the fine-grained attention and the coarse-grained attention, respectively.

In your first run, it may take some time to build up auxiliary information and compile CUDA kernels, so you may take a cup of coffee at this moment.

You can download a checkpoint [here](https://1drv.ms/u/s!Aq3YEPZCcV5ibz9ySjjNsEB74CQ) given by the authors of the Museformer paper, and put it in `checkpoints/mf-lmd6remi-1` for evaluation and inference.

## 4. Evaluation 1 - Perplexity (From the paper)

You can obtain perplexity on the test set with the following command:

```bash
bash tval/val__mf-lmd6remi-x.sh 1 checkpoint_best.pt 10240
```

The number `10240` indicates the maximum sequence length for calculation.

## 5. Evaluation 2 - Self-BLEU (Our code)
We designed codes for computing the Self-BLEU metric. You can find the codes in the eval_clean directory. The code computes the Self-BLEU of the given MIDI dataset. 

The following guide shows how to use eval_clean.
1. Install the requirements.txt with the command below.
   ```bash
   pip install -r requirements.txt
   ```
2. Fill in the midi directory with midi files to be evaluated. Set configuration variables in eval.py as desired. Then run eval.py to obtain self-bleu.
   ```bash
   python3 eval.py
   ```

## 6. Inference: MIDI generation

Use the following command to generate 5 music pieces, with the random seed set to 1:

```bash
mkdir -p output_log
seed=1
printf '\n\n\n\n\n' | bash tgen/generation__mf-lmd6remi-x.sh 1 checkpoint_best.pt ${seed} | tee output_log/generation.log
```

The number of `\n` controls the number of generated music pieces. The generation would take a while. Once done, the generation log will be saved at `output_log/generation.log`.

Then, use the following command to extract the generated token sequences from the generation log:

```bash
python tools/batch_extract_log.py output_log/generation.log output/generation --start_idx 1
```

You should see token representation of each generated music piece in `output/generation`.

Finally, run the following command to convert the token sequences into MIDI files:

```bash
python tools/batch_generate_midis.py --encoding-method REMIGEN2 --input-dir output/generation --output-dir output/generation
```

You should see the MIDI files in `output/generation`.


