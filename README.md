# OpenAsp Dataset

`OpenAsp` is an Open Aspect-based Multi-Document Summarization dataset derived from `DUC` and `MultiNews` summarization datasets.

![Antarctica](antarctica.png)

## Dataset Access

To generate `OpenAsp`, you require access to the `DUC` dataset which OpenAsp is derived from.

### Steps:

1. Grant access to DUC dataset by following NIST instructions [here](https://duc.nist.gov/data.html).
   * you should receive two user-password pairs (for DUC01-02 and DUC06-07)
   * you should receive a file named `fwdrequestingducdata.zip` (see `5.` if you received individual files).
2. Clone this repository by running the following command: `git clone https://github.com/liatschiff/OpenAsp.git`
3. Optionally create a `conda` or `virtualenv` environment:

```bash
conda create -n openasp 'python>3.10,<3.11'
conda activate openasp
```

4. Install python requirements, currently requires python3.8-3.10 (later python versions have issues with `spacy`)

```bash
pip install -r requirements.txt
```

5. copy `fwdrequestingducdata.zip` into the `OpenAsp` repo directory

    Note: in case the NIST's DUC files are in separate `.tgz` files, you'll need to create a single zip file named `fwdrequestingducdata.zip` with the following 4 files:
    * DUC2001_Summarization_Documents.tgz
    * DUC2002_Summarization_Documents.tgz
    * DUC2006_Summarization_Documents.tgz
    * DUC2007_Summarization_Documents.tgz

6. run the prepare script command:

```bash
python prepare_openasp_dataset.py --nist-duc2001-user '<2001-user>' --nist-duc2001-password '<2001-pwd>' --nist-duc2006-user '<2006-user>' --nist-duc2006-password '<2006-pwd>'
```

<details>
   <summary>Example script output log of successful run:</summary>

   ```bash
   2024-04-05 11:40:32 [INFO] main:754 - logging started
   2024-04-05 11:40:32 [INFO] run:655 - start creating dataset
   2024-04-05 11:40:32 [INFO] make_workdir:199 - using work dir workdir
   2024-04-05 11:40:32 [INFO] make_workdir:200 - creating work dir
   2024-04-05 11:40:32 [INFO] extract_duc_zip:211 - extracting zip file
   2024-04-05 11:40:32 [INFO] extract_duc_zip:222 - extracting DUC2001_Summarization_Documents.tgz into workdir/DUC2001_Summarization_Documents.tgz
   2024-04-05 11:40:32 [INFO] extract_duc_zip:222 - extracting DUC2002_Summarization_Documents.tgz into workdir/DUC2002_Summarization_Documents.tgz
   2024-04-05 11:40:32 [INFO] extract_duc_zip:222 - extracting DUC2006_Summarization_Documents.tgz into workdir/DUC2006_Summarization_Documents.tgz
   2024-04-05 11:40:32 [INFO] extract_duc_zip:222 - extracting DUC2007_Summarization_Documents.tgz into workdir/DUC2007_Summarization_Documents.tgz
   2024-04-05 11:40:32 [INFO] extract_duc_zip:224 - successfully extracted zip files
   2024-04-05 11:40:32 [INFO] download_nist_urls:182 - downloading 2 files using 2002 credentials
   2024-04-05 11:40:32 [INFO] download_nist_urls:189 - downloading https://duc.nist.gov/past_duc/duc2002/data/test/summaries/duc2002extractsandabstracts.tar.gz into workdir/duc.nist.gov/duc2002extractsandabstracts.tar.gz
   2024-04-05 11:40:34 [INFO] download_nist_urls:189 - downloading https://duc.nist.gov/past_duc/duc2002/results/abstracts/phase1/SEEpeers/manualpeers.tar.gz into workdir/duc.nist.gov/manualpeers.tar.gz
   2024-04-05 11:40:36 [INFO] download_nist_urls:195 - done downloading urls
   2024-04-05 11:40:36 [INFO] download_nist_urls:182 - downloading 7 files using 2006 credentials
   2024-04-05 11:40:36 [INFO] download_nist_urls:189 - downloading https://duc.nist.gov/past_duc_aquaint/duc2006/results/NIST/NISTeval.tar.gz into workdir/duc.nist.gov/NISTeval.tar.gz
   2024-04-05 11:40:38 [INFO] download_nist_urls:189 - downloading https://duc.nist.gov/past_duc_aquaint/duc2006/results/NIST-secondary-automatic/NISTeval2.tar.gz into workdir/duc.nist.gov/NISTeval2.tar.gz
   2024-04-05 11:40:41 [INFO] download_nist_urls:189 - downloading https://duc.nist.gov/past_duc_aquaint/duc2006/testdata/duc2006_topics.sgml into workdir/duc.nist.gov/duc2006_topics.sgml
   2024-04-05 11:40:41 [INFO] download_nist_urls:189 - downloading https://duc.nist.gov/past_duc_aquaint/duc2007/results/mainEval.tar.gz into workdir/duc.nist.gov/mainEval.tar.gz
   2024-04-05 11:40:44 [INFO] download_nist_urls:189 - downloading https://duc.nist.gov/past_duc_aquaint/duc2007/results/updateEval.tar.gz into workdir/duc.nist.gov/updateEval.tar.gz
   2024-04-05 11:40:46 [INFO] download_nist_urls:189 - downloading https://duc.nist.gov/past_duc_aquaint/duc2007/testdata/duc2007_topics.sgml into workdir/duc.nist.gov/duc2007_topics.sgml
   2024-04-05 11:40:47 [INFO] download_nist_urls:189 - downloading https://duc.nist.gov/past_duc_aquaint/duc2007/testdata/duc2007_UPDATEtopics.sgml into workdir/duc.nist.gov/duc2007_UPDATEtopics.sgml
   2024-04-05 11:40:47 [INFO] download_nist_urls:195 - done downloading urls
   2024-04-05 11:40:47 [INFO] extract_duc:239 - extracting duc2001 files
   2024-04-05 11:40:48 [INFO] extract_duc:245 - extracting duc2002 files
   2024-04-05 11:40:52 [INFO] extract_duc:256 - extracting duc2006 files
   2024-04-05 11:40:53 [INFO] extract_duc:268 - extracting duc2007 files
   2024-04-05 11:40:54 [INFO] download_multinews:364 - downloading multinews files
   2024-04-05 11:40:54 [INFO] download_multinews:373 - getting multinews train file link from google drive
   2024-04-05 11:40:54 [INFO] _download_multinews_train_link_option1:311 - getting multinews train file link from google drive
   2024-04-05 11:40:55 [WARNING] download_multinews:380 - failed to download multinews train file with option #1, trying option #2
   Traceback (most recent call last):
     File "./OpenAsp/prepare_openasp_dataset.py", line 377, in download_multinews
       _download_multinews_train_link_option1(train_src_fname)
     File "./OpenAsp/prepare_openasp_dataset.py", line 319, in _download_multinews_train_link_option1
       .group(1)
   AttributeError: 'NoneType' object has no attribute 'group'
   2024-04-05 11:40:55 [INFO] _download_multinews_train_link_option2:333 - getting multinews train file link from https://drive.usercontent.google.com/download?id=1wHAWDOwOoQWSj7HYpyJ3Aeud8WhhaJ7P&export=download
   2024-04-05 11:40:55 [INFO] _download_multinews_train_link_option2:341 - downloading multinews train file from https://drive.usercontent.google.com/download into workdir/mnews/raw/train.src.cleaned
   2024-04-05 11:41:05 [INFO] download_multinews:387 - multi-news train downloaded with option #2 successfully
   Skipping downloading file 1wHAWDOwOoQWSj7HYpyJ3Aeud8WhhaJ7P
   Downloading file 1p_u9_jpz3Zbj0EL05QFX6wvJAahmOn6h to workdir/mnews/raw/val.src.cleaned
   Downloading 1p_u9_jpz3Zbj0EL05QFX6wvJAahmOn6h into workdir/mnews/raw/val.src.cleaned... Done.
   Downloading file 1-n_6fj-1nM7sWtBSNkQCSfl5Rb3zPVfr to workdir/mnews/raw/test.src.cleaned
   Downloading 1-n_6fj-1nM7sWtBSNkQCSfl5Rb3zPVfr into workdir/mnews/raw/test.src.cleaned... Done.
   Downloading file 1QVgswwhVTkd3VLCzajK6eVkcrSWEK6kq to workdir/mnews/raw/train.tgt
   Downloading 1QVgswwhVTkd3VLCzajK6eVkcrSWEK6kq into workdir/mnews/raw/train.tgt... Done.
   Downloading file 1Y1lBbBU5Q0aJMqLhYEOdEtTqQ85XnRRM to workdir/mnews/raw/val.tgt
   Downloading 1Y1lBbBU5Q0aJMqLhYEOdEtTqQ85XnRRM into workdir/mnews/raw/val.tgt... Done.
   Downloading file 1CX_YcgQ3WwNC1fXBpMfwMXFPCqsd9Lbp to workdir/mnews/raw/test.tgt
   Downloading 1CX_YcgQ3WwNC1fXBpMfwMXFPCqsd9Lbp into workdir/mnews/raw/test.tgt... Done.
   Downloading https://raw.githubusercontent.com/Alex-Fabbri/Multi-News/master/data/ids/train.id to workdir/mnews/raw/train.id
   Downloading https://raw.githubusercontent.com/Alex-Fabbri/Multi-News/master/data/ids/val.id to workdir/mnews/raw/val.id
   Downloading https://raw.githubusercontent.com/Alex-Fabbri/Multi-News/master/data/ids/test.id to workdir/mnews/raw/test.id
   Number of documents with no text
   Train 11
   Valid 1
   Test 2
   
   Number of source document counts
   Train Counter({2: 23741, 3: 12577, 4: 4921, 5: 1846, 6: 706, 1: 498, 7: 371, 8: 194, 9: 81, 10: 29, 0: 8}) 44972
   Valid Counter({2: 3066, 3: 1555, 4: 610, 5: 195, 6: 79, 1: 58, 7: 38, 8: 13, 9: 7, 0: 1}) 5622
   Test Counter({2: 3022, 3: 1540, 4: 609, 5: 219, 6: 96, 1: 71, 7: 40, 8: 15, 9: 8, 10: 1, 0: 1}) 5622
   
   Number of source document counts (after removing instances with 0 source documents)
   Train Counter({2: 23741, 3: 12577, 4: 4921, 5: 1846, 6: 706, 1: 498, 7: 371, 8: 194, 9: 81, 10: 29}) 44964
   Valid Counter({2: 3066, 3: 1555, 4: 610, 5: 195, 6: 79, 1: 58, 7: 38, 8: 13, 9: 7}) 5621
   Test Counter({2: 3022, 3: 1540, 4: 609, 5: 219, 6: 96, 1: 71, 7: 40, 8: 15, 9: 8, 10: 1}) 5621
   
   Number of source document counts (after removing instances with 1 source document)
   Train Counter({2: 23741, 3: 12577, 4: 4921, 5: 1846, 6: 706, 7: 371, 8: 194, 9: 81, 10: 29}) 44466
   Valid Counter({2: 3066, 3: 1555, 4: 610, 5: 195, 6: 79, 7: 38, 8: 13, 9: 7}) 5563
   Test Counter({2: 3022, 3: 1540, 4: 609, 5: 219, 6: 96, 7: 40, 8: 15, 9: 8, 10: 1}) 5550
   
   2024-04-05 11:42:50 [INFO] load_openasp_dataset_metadata:557 - loading aspect dataset from openasp_v1_dataset_metadata.json
   2024-04-05 11:42:50 [INFO] load_openasp_dataset_metadata:560 - found 1593 aspect summaries
   2024-04-05 11:42:50 [INFO] load_final_duc_data:423 - loading duc generic summaries
   2024-04-05 11:42:50 [INFO] load_final_duc_data:444 - loading DUC-2001 data from workdir/duc/2001/task2.train.200.jsonl
   2024-04-05 11:42:50 [INFO] load_final_duc_data:444 - loading DUC-2001 data from workdir/duc/2001/task2.test.200.jsonl
   2024-04-05 11:42:50 [INFO] load_final_duc_data:444 - loading DUC-2002 data from workdir/duc/2002/task2.200.jsonl
   2024-04-05 11:42:50 [INFO] load_final_duc_data:444 - loading DUC-2006 data from workdir/duc/2006/task1.jsonl
   2024-04-05 11:42:50 [INFO] load_final_duc_data:444 - loading DUC-2007 data from workdir/duc/2007/task1.jsonl
   2024-04-05 11:42:50 [INFO] load_final_duc_data:456 - loading DUC-2001 data from workdir/duc/2001/task2.train.200.jsonl
   2024-04-05 11:42:50 [INFO] load_final_duc_data:456 - loading DUC-2001 data from workdir/duc/2001/task2.test.200.jsonl
   2024-04-05 11:42:50 [INFO] load_final_duc_data:456 - loading DUC-2002 data from workdir/duc/2002/task2.200.summaries.jsonl
   2024-04-05 11:42:50 [INFO] load_final_duc_data:456 - loading DUC-2006 data from workdir/duc/2006/task1.summaries.jsonl
   2024-04-05 11:42:50 [INFO] load_final_duc_data:456 - loading DUC-2007 data from workdir/duc/2007/task1.summaries.jsonl
   2024-04-05 11:46:13 [INFO] save_aspect_dataset:588 - creating output dir openasp-v1
   2024-04-05 11:46:13 [INFO] save_aspect_dataset:597 - writing to openasp-v1/test.jsonl.gz
   2024-04-05 11:46:13 [INFO] save_aspect_dataset:597 - writing to openasp-v1/train.jsonl.gz
   2024-04-05 11:46:13 [INFO] save_aspect_dataset:597 - writing to openasp-v1/valid.jsonl.gz
   2024-04-05 11:46:13 [INFO] save_aspect_dataset:597 - writing to openasp-v1/filtered.jsonl.gz
   2024-04-05 11:46:13 [INFO] save_aspect_dataset:597 - writing to openasp-v1/analysis_valid.jsonl.gz
   2024-04-05 11:46:13 [INFO] save_aspect_dataset:597 - writing to openasp-v1/analysis_test.jsonl.gz
   2024-04-05 11:46:16 [INFO] save_aspect_dataset:605 - finished saving OpenAsp dataset into openasp-v1
   2024-04-05 11:46:16 [INFO] verify_dataset_integrity:613 - verifying dataset integrity
   2024-04-05 11:46:16 [INFO] verify_dataset_integrity:617 - verifying dataset sizes and guids
   2024-04-05 11:46:16 [INFO] verify_dataset_integrity:623 - verifying dataset content hashes
   2024-04-05 11:46:16 [INFO] verify_dataset_integrity:637 - all summaries content hashes matches
   2024-04-05 11:46:16 [INFO] run:689 - script completed successfully, dataset is available on: openasp-v1
   ```

</details>

7. load the dataset using [huggingface datasets](https://huggingface.co/docs/datasets/index)

```python
from glob import glob
import os
import gzip
import shutil
from datasets import load_dataset

openasp_files = os.path.join('openasp-v1', '*.jsonl.gz')

data_files = {
    os.path.basename(fname).split('.')[0]: fname
    for fname in glob(openasp_files)
}

for ftype, fname in data_files.copy().items():
    with gzip.open(fname, 'rb') as gz_file:
        with open(fname[:-3], 'wb') as output_file:
            shutil.copyfileobj(gz_file, output_file)
    data_files[ftype] = fname[:-3]

# load OpenAsp as huggingface's dataset
openasp = load_dataset('json', data_files=data_files)

# print first sample from every split
for split in ['train', 'valid', 'test']:
    sample = openasp[split][0]

    # print title, aspect_label, summary and documents for the sample
    title = sample['title']
    aspect_label = sample['aspect_label']
    summary = '\n'.join(sample['summary_text'])
    input_docs_text = ['\n'.join(d['text']) for d in sample['documents']]

    print('* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *')
    print(f'Sample from {split}\nSplit title={title}\nAspect label={aspect_label}')
    print(f'\naspect-based summary:\n {summary}')
    print('\ninput documents:\n')
    for i, doc_txt in enumerate(input_docs_text):
        print(f'---- doc #{i} ----')
        print(doc_txt[:256] + '...')
    print('* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *\n\n\n')
```

### Troubleshooting

1. **Dataset failed loading with `load_dataset()`** - you may want to delete huggingface datasets cache folder
2. **401 Client Error: Unauthorized -** you're DUC credentials are incorrect, please verify them (case sensitive, no extra spaces etc)
3. **Dataset created but prints a warning about content verification -** you may be using different version of `NLTK` or `spacy` model which affects the sentence tokenization process. You must use exact versions as pinned on `requirements.txt`.
4. **IndexError: list index out of range -** similar to (3), try to reinstall the requirements with exact package versions. 

For anything else, please open a [GitHub issue](https://github.com/liatschiff/OpenAsp/issues/new).


### Under The Hood

The `prepare_openasp_dataset.py` script downloads `DUC` and `Multi-News` source files, uses `sacrerouge` package to
prepare the datasets and uses the `openasp_v1_dataset_metadata.json` file to extract the relevant aspect summaries and compile the final OpenAsp dataset.


## License

This repository, including the `openasp_v1_dataset_metadata.json` and `prepare_openasp_dataset.py`, are released under [APACHE license](LICENSE).

`OpenAsp` dataset summary and source document for each sample, which are generated by running the script, are licensed under the respective generic summarization dataset - [Multi-News license](https://github.com/Alex-Fabbri/Multi-News/blob/master/LICENSE.txt) and [DUC license](https://duc.nist.gov/data.html).
